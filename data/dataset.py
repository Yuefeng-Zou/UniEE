"""SessionDataset — windowed reader over the per-(session, role) NPZ caches.

Each session_dir contains a flat directory of ``.npy`` files
(see ``data/feature_extractor/build_session_npz.py`` for the layout):

    T.npy                  int scalar
    label.npy              (T,) float32                                    # always present, 0 for unlabelled
    label_mask.npy         (T,) bool                                       # False where label is missing
    label_type.npy         object array of one string                      # "continuous" | "pinsoro" | "none"
    label_task.npy         (T,) int64    — only if label_type == "pinsoro"
    label_social.npy       (T,) int64    — only if label_type == "pinsoro"
    partner_roles.npy      object array — names of partner slots actually filled
    target_{feat}.npy      (T, dim)
    partner{i}_{feat}.npy  (T, dim)   for i in 0 .. n_partners-1
    _DONE                  marker file

The dataset:
  * mmaps everything per-session and caches handles
  * yields sliding windows of length ``window_len`` with stride ``stride``
  * z-scores features using the global stats file (see compute_feature_stats.py)
  * pads short sessions on the right (attention_mask = False on padding)
  * works for both regression (NoXi family / mpii) and classification (PInSoRo)
    by passing through whatever label fields exist
"""
from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class WindowSpec:
    session_key:   str        # "{domain}/{session_id}/{target_role}"
    session_dir:   str
    start:         int        # frame index
    end:           int        # exclusive; can exceed T (caller pads)
    valid_end:     int        # min(end, T) — frames before this are real
    domain:        str
    fine_domain:   str        # === domain for now; reserved for noxi_add language split


class _NPYCache:
    """LRU mmap cache. Each session's .npy files are opened once and reused."""
    def __init__(self, max_sessions: int = 8) -> None:
        self._data: OrderedDict[str, dict] = OrderedDict()
        self.max_sessions = int(max_sessions)

    def get(self, session_dir: str, names: list[str]) -> dict:
        if session_dir in self._data:
            self._data.move_to_end(session_dir)
            return self._data[session_dir]
        sd = Path(session_dir)
        data: dict[str, np.ndarray] = {}
        for name in names:
            p = sd / f"{name}.npy"
            if not p.exists():
                continue                                   # missing modality (e.g. mpii openface3)
            # Object-dtype arrays (label_type, partner_roles) can't be mmapped.
            if name in ("label_type", "partner_roles"):
                data[name] = np.load(p, allow_pickle=True)
            else:
                data[name] = np.load(p, mmap_mode="r")
        self._data[session_dir] = data
        if len(self._data) > self.max_sessions:
            self._data.popitem(last=False)
        return data


class SessionDataset(Dataset):
    """Sliding-window iterator over labelled (or labelled+unlabelled) sessions.

    Parameters
    ----------
    manifest_path
        Per-domain JSONL written by build_session_npz.py. Rows ignored if
        ``has_label`` is False AND ``drop_label_unavailable=True``.
    npz_root
        Same as build_session_npz.py's --out-root.
    features
        List of feature names to load (must be a subset of the build preset).
    window_len, stride
        Sliding window over each session. Train mode uses stride < window_len
        (overlap = 1 - stride/window_len). Eval uses smaller stride for
        dense overlap-add inference.
    max_partners
        Number of partner slots the model expects. Extra partners ignored;
        missing slots zero-filled with partner_present=False.
    feature_stats
        Path to .npz from compute_feature_stats.py. Per-channel mean/std for
        z-scoring at load time. Required when feature list includes wide-
        scale features (egemapsv2 ±3e4, etc.) to avoid bf16 NaN.
    normalize_clip
        Post-normalize clip range. ±5 covers ≥99.9999% under N(0,1).
    """
    def __init__(
        self,
        manifest_path: Path,
        npz_root: Path,
        features: list[str],
        window_len: int = 512,
        stride: int = 64,
        mode: str = "train",
        drop_label_unavailable: bool = True,
        max_partners: int = 1,
        cache_sessions: int = 8,
        feature_stats: Path | None = None,
        normalize_clip: float = 5.0,
        feature_dims: dict[str, int] | None = None,
    ) -> None:
        if mode not in ("train", "eval"):
            raise ValueError(f"mode must be 'train' or 'eval', got {mode}")
        self.features = list(features)
        self.window_len = int(window_len)
        self.stride = int(stride)
        self.mode = mode
        self.max_partners = int(max_partners)
        self._cache = _NPYCache(max_sessions=cache_sessions)
        # Per-feature target dim. If NPZ shape > declared dim (e.g. swin
        # was over-padded to 1024 but is really 768), we slice to the
        # first ``feature_dims[f]`` columns. None = use whatever NPZ has.
        self._feature_dims = dict(feature_dims) if feature_dims else {}

        # ── z-score normalization stats ─────────────────────────────────
        # Critical for numerical stability; see memory/feature_scale_nan.md
        self._feat_mean: dict[str, np.ndarray] = {}
        self._feat_std:  dict[str, np.ndarray] = {}
        self._normalize_clip = float(normalize_clip)
        if feature_stats is not None:
            sf = np.load(feature_stats, allow_pickle=True)
            for f in self.features:
                mk, sk = f"{f}_mean", f"{f}_std"
                if mk not in sf.files or sk not in sf.files:
                    raise ValueError(
                        f"feature_stats {feature_stats} missing entry for '{f}' "
                        f"(need '{mk}' and '{sk}')"
                    )
                self._feat_mean[f] = sf[mk].astype(np.float32)
                self._feat_std[f]  = np.maximum(sf[sk].astype(np.float32), 1e-6)

        # ── decide which .npy names to mmap each window ─────────────────
        self._array_names: list[str] = [
            "T", "label", "label_mask", "label_type", "partner_roles",
            "label_task", "label_social",            # PInSoRo-only; optional
            "label_pseudo_cont",                      # PInSoRo-only; Phase 3 bridge
        ]
        for m in self.features:
            self._array_names.append(f"target_{m}")
            for i in range(self.max_partners):
                self._array_names.append(f"partner{i}_{m}")

        # ── enumerate windows ───────────────────────────────────────────
        rows = [json.loads(l) for l in Path(manifest_path).read_text().splitlines() if l.strip()]
        self._windows: list[WindowSpec] = []
        n_kept = n_skipped = 0
        for row in rows:
            if drop_label_unavailable and not row.get("has_label", True):
                n_skipped += 1
                continue
            domain     = row["domain"]
            sess_id    = row["session_id"]
            target     = row["target_role"]
            sess_dir   = Path(row["out_dir"])
            if not (sess_dir / "_DONE").exists():
                n_skipped += 1
                continue
            T = int(np.load(sess_dir / "T.npy"))
            if T < self.window_len // 4:                  # too short to be useful
                n_skipped += 1
                continue
            session_key = f"{domain}/{sess_id}/{target}"
            # Stride-based windows.
            for start in range(0, max(1, T - self.window_len + 1), self.stride):
                end = start + self.window_len
                valid_end = min(end, T)
                self._windows.append(WindowSpec(session_key, str(sess_dir),
                                                start, end, valid_end,
                                                domain, domain))
            # Always include the tail window (frames after the last stride start).
            # NOTE: in eval mode we deliberately SKIP this to avoid the same
            # frame producing two predictions (the last stride window already
            # right-pads with attention_mask=False on missing frames; the tail
            # would re-cover the same frames with a different prediction,
            # double-counting them in the micro CCC concat).
            if mode != "eval":
                last_start = ((max(1, T - self.window_len + 1) - 1) // self.stride) * self.stride if T > self.window_len else 0
                tail_start = max(0, T - self.window_len)
                if tail_start > 0 and tail_start != last_start:
                    end = tail_start + self.window_len
                    valid_end = min(end, T)
                    self._windows.append(WindowSpec(session_key, str(sess_dir),
                                                    tail_start, end, valid_end,
                                                    domain, domain))
            n_kept += 1
        self.n_sessions = n_kept
        self.n_skipped_sessions = n_skipped

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, idx: int) -> dict:
        w = self._windows[idx]
        data = self._cache.get(w.session_dir, self._array_names)

        W = self.window_len
        valid = w.valid_end - w.start

        attn = np.zeros(W, dtype=bool)
        attn[:valid] = True

        def _slice_feat(key: str, dim_hint: int | None = None) -> np.ndarray | None:
            """Slice (W, D) from a mmap, z-score-normalize, nan_to_num.

            If self._feature_dims declares a target dim smaller than the
            NPZ-stored width, take the first N columns (used to undo a
            historical over-pad of swin from 768 → 1024).
            """
            arr = data.get(key)
            if arr is None:
                return None
            _, feat_name = key.split("_", 1)
            target_dim = self._feature_dims.get(feat_name)
            dim = arr.shape[1] if dim_hint is None else dim_hint
            if target_dim is not None and target_dim < dim:
                dim = target_dim
            out = np.zeros((W, dim), dtype=np.float32)
            out[:valid] = np.asarray(arr[w.start:w.valid_end, :dim])
            # PInSoRo egemapsv2 uses ±3.689e19 as a missing-value sentinel.
            # Replace BEFORE z-score so the normalization doesn't poison
            # whole frames; treat as NaN then 0-fill (= per-channel mean
            # after normalize).
            sentinel_mask = np.abs(out) > 1e10
            if sentinel_mask.any():
                out[sentinel_mask] = np.nan
            if self._feat_mean:
                if feat_name in self._feat_mean:
                    mean = self._feat_mean[feat_name][:dim]
                    std  = self._feat_std[feat_name][:dim]
                    out[:valid] -= mean
                    out[:valid] /= std
                    np.clip(out, -self._normalize_clip, self._normalize_clip, out=out)
            np.nan_to_num(out, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            return out

        # Target features.
        target_feats: dict[str, np.ndarray] = {}
        for m in self.features:
            arr = _slice_feat(f"target_{m}")
            if arr is not None:
                target_feats[m] = arr

        # Partner features. ``partner_roles`` tells us which slots are filled.
        partner_roles = list(data.get("partner_roles", []))
        partner_feats: list[dict[str, np.ndarray]] = []
        partner_present: list[bool] = []
        for i in range(self.max_partners):
            slot: dict[str, np.ndarray] = {}
            present = (i < len(partner_roles) and partner_roles[i])
            for m in self.features:
                arr = _slice_feat(f"partner{i}_{m}") if present else None
                if arr is None and target_feats.get(m) is not None:
                    # Zero-fill so partner_proj has the same shape as target.
                    arr = np.zeros_like(target_feats[m])
                if arr is not None:
                    slot[m] = arr
            partner_feats.append(slot)
            partner_present.append(bool(present))

        # Label (always exists; zero where mask is False).
        label = np.zeros(W, dtype=np.float32)
        label_mask = np.zeros(W, dtype=bool)
        if "label" in data:
            label[:valid] = np.asarray(data["label"][w.start:w.valid_end]).astype(np.float32)
            label_mask[:valid] = np.asarray(data["label_mask"][w.start:w.valid_end])

        # Optional categorical labels.
        out_dict: dict = {
            "target_feats":     {m: torch.from_numpy(v) for m, v in target_feats.items()},
            "partner_feats":    [{m: torch.from_numpy(v) for m, v in slot.items()} for slot in partner_feats],
            "partner_present":  partner_present,
            "label":            torch.from_numpy(label),
            "label_mask":       torch.from_numpy(label_mask),
            "attention_mask":   torch.from_numpy(attn),
            "session_key":      w.session_key,
            "window_start":     w.start,
            "valid_end":        w.valid_end,
            "domain":           w.domain,
        }
        if "label_task" in data:
            task = np.zeros(W, dtype=np.int64)
            task[:valid] = np.asarray(data["label_task"][w.start:w.valid_end]).astype(np.int64)
            out_dict["label_task"] = torch.from_numpy(task)
        if "label_social" in data:
            social = np.zeros(W, dtype=np.int64)
            social[:valid] = np.asarray(data["label_social"][w.start:w.valid_end]).astype(np.int64)
            out_dict["label_social"] = torch.from_numpy(social)
        if "label_pseudo_cont" in data:
            pc = np.zeros(W, dtype=np.float32)
            pc[:valid] = np.asarray(data["label_pseudo_cont"][w.start:w.valid_end]).astype(np.float32)
            out_dict["label_pseudo_cont"] = torch.from_numpy(pc)
        return out_dict


def collate(batch: list[dict]) -> dict:
    """Stack tensors along batch dim; keep metadata as lists.

    IMPORTANT: every item in `batch` must share the same ``domain`` (the
    DomainPromptPool selects exactly one prompt per forward). The
    DomainBalancedBatchSampler enforces this; the default RandomSampler
    breaks it. Always pair this dataset with our sampler.
    """
    out: dict = {}
    # Tensor-stacked fields.
    out["label"]          = torch.stack([b["label"]          for b in batch])
    out["label_mask"]     = torch.stack([b["label_mask"]     for b in batch])
    out["attention_mask"] = torch.stack([b["attention_mask"] for b in batch])

    # Per-modality stacking. Use intersection of modalities across batch
    # (handles sessions missing optional features like qwen3vl_emb).
    modalities = set(batch[0]["target_feats"].keys())
    for b in batch[1:]:
        modalities &= set(b["target_feats"].keys())
    modalities = sorted(modalities)
    out["target_feats"] = {
        m: torch.stack([b["target_feats"][m] for b in batch])
        for m in modalities
    }
    # Partners: stack per slot per modality.
    max_p = len(batch[0]["partner_feats"])
    out["partner_feats"] = []
    for i in range(max_p):
        slot_mods = set(batch[0]["partner_feats"][i].keys())
        for b in batch[1:]:
            slot_mods &= set(b["partner_feats"][i].keys())
        slot_mods = sorted(slot_mods)
        out["partner_feats"].append({
            m: torch.stack([b["partner_feats"][i][m] for b in batch])
            for m in slot_mods
        })
    # partner_present: collapse to per-slot bool (all items in batch should agree
    # by virtue of same-domain batches, but be defensive).
    out["partner_present"] = [
        bool(all(b["partner_present"][i] for b in batch))
        for i in range(max_p)
    ]

    # Domain — assert consistency.
    domains = {b["domain"] for b in batch}
    if len(domains) != 1:
        raise ValueError(f"collate batch contains multiple domains: {domains}. "
                         "Use DomainBalancedBatchSampler.")
    out["domain"] = next(iter(domains))

    # Categorical labels (optional).
    if "label_task" in batch[0]:
        out["label_task"]   = torch.stack([b["label_task"]   for b in batch])
        out["label_social"] = torch.stack([b["label_social"] for b in batch])
    if "label_pseudo_cont" in batch[0]:
        out["label_pseudo_cont"] = torch.stack([b["label_pseudo_cont"] for b in batch])

    # Metadata (kept as lists).
    out["session_key"]   = [b["session_key"]   for b in batch]
    out["window_start"]  = [b["window_start"]  for b in batch]
    out["valid_end"]     = [b["valid_end"]     for b in batch]
    return out
