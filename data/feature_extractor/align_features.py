"""Resample per-feature streams onto a 25 Hz timebase.

Per probe results the official release is NOT uniformly 25 Hz:

    feature             | sr by domain
    --------------------+---------------------------------------------------
    egemapsv2           | 25 Hz everywhere
    w2vbert2            | 40 Hz on NoXi / NoXi-add / mpii;  25 Hz on NoXi-J / PInSoRo
    xlmr                |  2 Hz on NoXi;                    25 Hz elsewhere
    openface2 / 3 /     |
    openpose / videomae | 25 Hz on NoXi / NoXi-J / NoXi-add / mpii;  30 Hz on PInSoRo
    swin / clip / dino  | same as above
    dino                | dim=768 on NoXi family / PInSoRo, dim=2304 on mpii

The training pipeline runs at a single 25 Hz grid, so this module is the
single place where every official stream gets resampled into ``(T_25, dim)``.

We use scipy linear interpolation for everything. Average-pool variants
(planned for 40→25 in v3 doc) are dropped — at 40 → 25 the ratio is 1.6,
which doesn't divide evenly, so linear interp is both simpler and more
accurate than a "pool then interpolate" pipeline.

dino mismatch (2304 vs 768) is dim-handled by ``align_feature_dim`` —
DINOv2-Large concatenates 3 ViT scales; the first 768 columns are the
patch-token mean from the smallest ViT, which is the most directly
comparable to the dim-768 NoXi family export. Same shape lets one
ModalityProjector handle every domain.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d


TARGET_FPS = 25.0


def resample_to_25hz(arr: np.ndarray, src_fps: float,
                     target_T: int | None = None) -> np.ndarray:
    """Resample ``arr`` of shape (T_src, D) from ``src_fps`` to 25 Hz.

    Parameters
    ----------
    arr : (T_src, D) float32
    src_fps : source frame rate (e.g. 40.0 for w2vbert2 on NoXi)
    target_T : if given, force output to exactly this length (truncate or
               extrapolate). Useful for clamping every stream to the same
               reference length (e.g. label length) before stacking.

    Notes
    -----
    Already-25Hz input is short-circuited (just optional truncation).
    Tiny rounding mismatches (sr=25 with 1-frame off) are handled by the
    truncation step.
    """
    if arr.ndim == 1:
        arr = arr[:, None]
    T_src, D = arr.shape

    if abs(src_fps - TARGET_FPS) < 1e-6:
        # already at target rate
        out = arr.astype(np.float32, copy=False)
    else:
        src_times = np.arange(T_src) / src_fps
        # Use the duration of the source stream as the resampling horizon;
        # we'll re-clamp to target_T below if needed.
        duration = T_src / src_fps
        T_new = int(round(duration * TARGET_FPS))
        if T_new < 1:
            T_new = 1
        tgt_times = np.arange(T_new) / TARGET_FPS
        f = interp1d(src_times, arr, axis=0, kind="linear",
                     bounds_error=False, fill_value="extrapolate")
        out = f(tgt_times).astype(np.float32)

    if target_T is not None:
        out = _to_length(out, target_T)
    return out


def broadcast_segments(seg_feats: np.ndarray,
                       seg_times: np.ndarray,
                       target_T: int) -> np.ndarray:
    """Broadcast segment-level features over the frames they cover.

    Used for the 2 Hz NoXi XLM-R stream where each value covers a sentence
    segment rather than a frame. ``seg_times`` is shape (N, 2) of (t0, t1)
    in seconds. Frames outside any segment are left at zero.
    """
    if seg_feats.ndim == 1:
        seg_feats = seg_feats[:, None]
    D = seg_feats.shape[1]
    out = np.zeros((target_T, D), dtype=np.float32)
    for (t0, t1), feat in zip(seg_times, seg_feats):
        i0 = max(0, int(round(t0 * TARGET_FPS)))
        i1 = min(target_T, int(round(t1 * TARGET_FPS)))
        if i1 > i0:
            out[i0:i1] = feat
    return out


def align_feature_dim(arr: np.ndarray, expected_dim: int) -> np.ndarray:
    """Truncate or pad-zero on the channel axis to hit ``expected_dim``.

    Specifically: mpii's dino is dim=2304 (DINOv2-Large concat of 3 ViT
    scales), while all other domains ship dim=768. We take the first 768
    columns from the mpii export — these are the smallest-ViT patch-mean
    embeddings and most directly comparable. Other dim mismatches we don't
    expect to see and would warrant their own handling.
    """
    T, D = arr.shape
    if D == expected_dim:
        return arr
    if D > expected_dim:
        return arr[:, :expected_dim].copy()
    out = np.zeros((T, expected_dim), dtype=arr.dtype)
    out[:, :D] = arr
    return out


def _to_length(arr: np.ndarray, target_T: int) -> np.ndarray:
    """Truncate or zero-pad on the time axis."""
    T, D = arr.shape
    if T == target_T:
        return arr
    if T > target_T:
        return arr[:target_T].copy()
    out = np.zeros((target_T, D), dtype=arr.dtype)
    out[:T] = arr
    return out
