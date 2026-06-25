"""Submission CSV writers — mirrors official baseline 4_TestingNN_fairness_per_session.py.

Output convention (per official format):
  Regression (NoXi / NoXi-add / NoXi-J / MPIGI):
    <out_dir>/<session>/<role>.engagement.prediction.csv  -- one %.6f / line
  Classification (PInSoRo cc / cr):
    <out_dir>/<session>/<role>.task_engagement.engagement.prediction.csv     -- one class label string / line
    <out_dir>/<session>/<role>.social_engagement.engagement.prediction.csv

Length convention: the file must have ONE line per frame for which the
corresponding feature stream had a valid annotation row (== feature row,
matching official baseline). Frames where annotation was 'nan' / '' /
'-nan(ind)' must NOT be written. The caller passes ``y_pred`` already
indexed for valid rows only.

The output directory layout matches the official submission ZIP structure:
  submission/
    noxi-base/<session>/<role>.engagement.prediction.csv
    noxi-additional/<session>/<role>.engagement.prediction.csv
    noxi-j/<session>/<role>.engagement.prediction.csv
    mpiigroupinteraction/<session>/<role>.engagement.prediction.csv
    pinsoro-cc/<session>/<role>.task_engagement.engagement.prediction.csv (+ social)
    pinsoro-cr/<session>/<role>.task_engagement.engagement.prediction.csv (+ social)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


# Domain name → official submission folder
DOMAIN_TO_FOLDER = {
    "noxi":       "noxi-base",
    "noxi_add":   "noxi-additional",
    "noxi_j":     "noxi-j",
    "mpiigi":     "mpiigroupinteraction",
    "pinsoro_cc": "pinsoro-cc",
    "pinsoro_cr": "pinsoro-cr",
}

# Official PInSoRo class inverse maps — must match dataset-config.json
PINSORO_TASK_INV = {0: "goaloriented", 1: "aimless", 2: "adultseeking", 3: "noplay"}
PINSORO_SOCIAL_INV = {
    0: "solitary", 1: "onlooker", 2: "parallel", 3: "associative", 4: "cooperative",
}


def submission_root(out_dir: str | Path, domain: str) -> Path:
    folder = DOMAIN_TO_FOLDER[domain]
    p = Path(out_dir) / folder
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_regression(out_dir: str | Path, domain: str, session_id: str,
                     role: str, y_pred: np.ndarray) -> Path:
    """Write per-frame continuous engagement predictions.

    ``y_pred`` is (N,) float in [0, 1]. Length must equal the count of valid
    annotation rows (== feature rows the model consumed).
    """
    if y_pred.ndim != 1:
        raise ValueError(f"y_pred must be 1-D, got shape {y_pred.shape}")
    y = np.clip(np.asarray(y_pred, dtype=np.float32), 0.0, 1.0)
    sess_dir = submission_root(out_dir, domain) / session_id
    sess_dir.mkdir(parents=True, exist_ok=True)
    out_path = sess_dir / f"{role}.engagement.prediction.csv"
    np.savetxt(out_path, y, fmt="%.6f")
    return out_path


def write_classification(out_dir: str | Path, domain: str, session_id: str,
                         role: str, task_pred: np.ndarray,
                         social_pred: np.ndarray) -> tuple[Path, Path]:
    """Write per-frame PInSoRo class predictions.

    ``task_pred`` / ``social_pred`` may be (N,) int class IDs or (N, C) logits/probs.
    """
    sess_dir = submission_root(out_dir, domain) / session_id
    sess_dir.mkdir(parents=True, exist_ok=True)

    task_ids   = _as_class_ids(task_pred,   n_classes=4)
    social_ids = _as_class_ids(social_pred, n_classes=5)

    task_labels   = [PINSORO_TASK_INV[i]   for i in task_ids]
    social_labels = [PINSORO_SOCIAL_INV[i] for i in social_ids]

    task_path   = sess_dir / f"{role}.task_engagement.engagement.prediction.csv"
    social_path = sess_dir / f"{role}.social_engagement.engagement.prediction.csv"
    task_path.write_text("\n".join(task_labels) + "\n", encoding="utf-8")
    social_path.write_text("\n".join(social_labels) + "\n", encoding="utf-8")
    return task_path, social_path


def _as_class_ids(arr: np.ndarray, n_classes: int) -> list[int]:
    arr = np.asarray(arr)
    if arr.ndim == 2:
        if arr.shape[1] != n_classes:
            raise ValueError(f"expected (N, {n_classes}), got shape {arr.shape}")
        return arr.argmax(axis=1).astype(int).tolist()
    if arr.ndim == 1:
        return arr.astype(int).tolist()
    raise ValueError(f"unsupported shape {arr.shape}")


def savgol_smooth(y: np.ndarray, window: int = 25, poly: int = 3) -> np.ndarray:
    """Optional smoothing for regression. ``window`` must be odd and ≤ len(y).
    Degrades gracefully on short sequences.
    """
    from scipy.signal import savgol_filter
    n = len(y)
    if n < 5:
        return y
    w = min(window, n if n % 2 == 1 else n - 1)
    w = max(w, 5)
    p = min(poly, w - 1)
    return savgol_filter(y, w, p, mode="nearest")
