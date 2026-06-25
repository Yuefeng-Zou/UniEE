"""Domain-aware label loading.

Each fine domain ships labels in its own format:

* NoXi / NoXi-J / mpii  : per-frame ``engagement.annotation.csv`` with one
                          float-on-line (no header). For mpii the file
                          lives in a parallel ``engagement-annotations-*/``
                          directory rather than next to the features.
* NoXi-add / test sets  : no labels (zero-shot evaluation).
* PInSoRo               : two parallel CSVs:
                          ``{role}.task_engagement.annotation.csv``   (str class)
                          ``{role}.social_engagement.annotation.csv`` (str class)
                          plus duplicate ``.1.annotation.csv`` files (rater 2
                          which we currently ignore; future ablation could
                          average).

This module returns numpy arrays at the original feature frame count, with
an explicit ``label_mask`` of which frames have a valid label. Downstream
``build_session_npz.py`` is in charge of resampling to the 25 Hz grid.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


PINSORO_TASK_CLASSES = ("goaloriented", "aimless", "noplay", "adultseeking")
PINSORO_SOCIAL_CLASSES = ("solitary", "onlooker", "parallel",
                          "associative", "cooperative")

PINSORO_TASK_TO_IDX = {c: i for i, c in enumerate(PINSORO_TASK_CLASSES)}
PINSORO_SOCIAL_TO_IDX = {c: i for i, c in enumerate(PINSORO_SOCIAL_CLASSES)}

# Hand-picked priors mapping each PInSoRo class to a 0–1 engagement scalar.
# Used by the LearnableBridge (Phase 3) to supervise a continuous-value
# pseudo-target on PInSoRo, so the regression head still gets gradient
# updates on classification frames. NOT used by Phase 2.
#
# Ordering matches PINSORO_{TASK,SOCIAL}_CLASSES above.
#   task   : goaloriented > adultseeking > aimless > noplay
#   social : cooperative > associative > parallel > onlooker > solitary
# Final pseudo_cont = 0.5 * task_prior + 0.5 * social_prior, clipped to [0, 1].
PINSORO_TASK_CONT_PRIOR = (0.80, 0.40, 0.10, 0.55)
PINSORO_SOCIAL_CONT_PRIOR = (0.10, 0.30, 0.50, 0.70, 0.90)


@dataclass
class LabelBundle:
    """Per-frame label + mask at the SOURCE (pre-resample) rate.

    For continuous-label domains:
        labels_cont : (T,) float32 in [0, 1]
        label_mask  : (T,) bool — True where label is observed
        labels_task / labels_social : None
    For PInSoRo:
        labels_task : (T,) int64 in [0, len(TASK_CLASSES))
        labels_social : (T,) int64 in [0, len(SOCIAL_CLASSES))
        label_mask  : (T,) bool — True where both task & social are valid
        labels_cont : None
    For test sets that ship without labels:
        all label fields are None; label_mask is False everywhere with T=0
        (the caller decides what T to allocate when stitching).
    """
    labels_cont:   np.ndarray | None
    labels_task:   np.ndarray | None
    labels_social: np.ndarray | None
    label_mask:    np.ndarray
    src_fps:       float  # native rate of the label stream


# ── Continuous-label loader (NoXi family + mpii) ──────────────────────────

def load_continuous_csv(csv_path: Path, src_fps: float = 25.0) -> LabelBundle:
    """Load a frame-level engagement CSV (one float per line, no header).

    NaN / missing entries become ``False`` in ``label_mask``.
    """
    # The file has no header and a single float column. np.genfromtxt
    # tolerates empty lines and the (very rare) malformed row by yielding
    # NaN, which we mask out.
    arr = np.genfromtxt(csv_path, dtype=np.float32, missing_values="",
                        invalid_raise=False)
    if arr.ndim == 0:
        arr = arr[None]
    mask = np.isfinite(arr) & (arr >= 0.0) & (arr <= 1.0)
    arr = np.where(mask, arr, 0.0).astype(np.float32)
    return LabelBundle(
        labels_cont=arr,
        labels_task=None, labels_social=None,
        label_mask=mask, src_fps=src_fps,
    )


def find_continuous_label(session_dir: Path, role: str,
                          domain: str) -> Path | None:
    """Locate the engagement label CSV for one (session, role, domain)."""
    if domain == "mpiigi":
        # mpii val: features under .../precomputed-features-val/<sess>/
        # labels  under .../engagement-annotations-val/<sess>/<role>.engagement.annotation.csv
        feat_parent = session_dir.parent       # …/precomputed-features-val
        split_root = feat_parent.parent        # …/val or …/test
        # Find sibling engagement-annotations-* dir.
        for cand in split_root.iterdir():
            if cand.is_dir() and cand.name.startswith("engagement-annotations"):
                lbl = cand / session_dir.name / f"{role}.engagement.annotation.csv"
                if lbl.exists():
                    return lbl
        return None
    # Default: label sits next to features.
    lbl = session_dir / f"{role}.engagement.annotation.csv"
    return lbl if lbl.exists() else None


# ── PInSoRo categorical loader ────────────────────────────────────────────

def _read_class_csv(path: Path, mapping: dict[str, int]) -> tuple[np.ndarray, np.ndarray]:
    """Read a 1-column class-name CSV into (idx, valid_mask).

    PInSoRo main CSVs use EMPTY ROWS to mark frames where the two annotators
    disagree (readme: "frames where annotators agree are recorded ... with
    empty rows indicating disagreement"). We MUST preserve those rows as
    mask=False at the original line index — collapsing them would corrupt
    the time axis.
    """
    # splitlines() preserves empty lines as "" entries — unlike np.genfromtxt
    # which silently drops them and shrinks the array.
    tokens = path.read_text(encoding="utf-8", errors="replace").splitlines()
    out = np.zeros(len(tokens), dtype=np.int64)
    mask = np.zeros(len(tokens), dtype=bool)
    for i, t in enumerate(tokens):
        t = t.strip()
        if t in mapping:
            out[i] = mapping[t]
            mask[i] = True
    return out, mask


def load_pinsoro_labels(session_dir: Path, role: str,
                        src_fps: float = 30.0) -> LabelBundle | None:
    task_p   = session_dir / f"{role}.task_engagement.annotation.csv"
    social_p = session_dir / f"{role}.social_engagement.annotation.csv"
    if not task_p.exists() or not social_p.exists():
        return None
    task_idx,   task_mask   = _read_class_csv(task_p,   PINSORO_TASK_TO_IDX)
    social_idx, social_mask = _read_class_csv(social_p, PINSORO_SOCIAL_TO_IDX)
    # Align lengths — PInSoRo files are usually equal, but be defensive.
    T = min(len(task_idx), len(social_idx))
    task_idx, task_mask = task_idx[:T], task_mask[:T]
    social_idx, social_mask = social_idx[:T], social_mask[:T]
    return LabelBundle(
        labels_cont=None,
        labels_task=task_idx, labels_social=social_idx,
        label_mask=(task_mask & social_mask),
        src_fps=src_fps,
    )


# ── Unified entry point ───────────────────────────────────────────────────

def load_labels(session_dir: Path, role: str, domain: str,
                source_fps_hint: float = 25.0) -> LabelBundle | None:
    """Domain-dispatched loader. Returns None if no usable label exists."""
    if domain in ("pinsoro_cc", "pinsoro_cr"):
        return load_pinsoro_labels(session_dir, role, src_fps=30.0)
    if domain in ("noxi", "noxi_j", "mpiigi"):
        p = find_continuous_label(session_dir, role, domain)
        if p is None:
            return None
        return load_continuous_csv(p, src_fps=source_fps_hint)
    # noxi_add, NoXi test-base, mpii test, PInSoRo test → no label
    return None
