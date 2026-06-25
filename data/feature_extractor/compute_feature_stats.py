"""Compute per-channel mean/std for each feature from training NPZ.

Output: a single .npz with arrays ``<feat>_mean`` / ``<feat>_std`` (shape
``[dim]``) for each feature. Loaded by SessionDataset(feature_stats=) to
z-score features at load time and clip to ±N to keep bf16 projections
numerically stable.

Aggregates over training manifests only (val/test never read), across
both target and partner roles (same normalization regardless of slot).

NaN-safe AND sentinel-aware:
  * openface3 has ~0.2% NaN cells from failed face detection
  * PInSoRo egemapsv2 has ~7% sentinel ±3.689e19 cells (missing-value)
  Both are masked out of the per-channel statistics.

Run example:
    python -m multimediate26.data.feature_extractor.compute_feature_stats \\
        --npz-root multimediate26/data_processed/npz_v3 \\
        --manifests multimediate26/manifests/noxi_train.jsonl \\
                    multimediate26/manifests/noxi_j_train.jsonl \\
        --features openface2,openface3,openpose,w2vbert2,egemapsv2,xlmr,swin,clip \\
        --out multimediate26/data_processed/npz_v3/__feature_stats.npz
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


# Same magnitude cutoff as dataset._slice_feat — values above this are
# treated as sentinel/missing.
SENTINEL_ABS = 1e10


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz-root",  type=Path, required=True)
    ap.add_argument("--manifests", nargs="+", type=Path, required=True,
                    help="training manifests to aggregate (omit val/test)")
    ap.add_argument("--features",  type=str, required=True,
                    help="comma-separated feature names")
    ap.add_argument("--out",       type=Path, required=True)
    ap.add_argument("--max-partners", type=int, default=1)
    args = ap.parse_args()

    features = [f.strip() for f in args.features.split(",") if f.strip()]

    # Collect session dirs from manifests (de-duped, only labeled).
    sess_dirs: list[Path] = []
    seen: set[Path] = set()
    for mf in args.manifests:
        rows = [json.loads(l) for l in mf.read_text().splitlines() if l.strip()]
        for r in rows:
            if not r.get("has_label", True):
                continue
            sd = Path(r["out_dir"])
            if sd in seen or not (sd / "_DONE").exists():
                continue
            seen.add(sd)
            sess_dirs.append(sd)
    print(f"Scanning {len(sess_dirs)} sessions × {len(features)} features × "
          f"(target + {args.max_partners} partner) roles ...", file=sys.stderr)

    role_keys = ["target"] + [f"partner{i}" for i in range(args.max_partners)]

    # Running stats in float64; nan-safe; sentinel-safe via finite-mask.
    sums:   dict[str, np.ndarray] = {}
    sumsqs: dict[str, np.ndarray] = {}
    counts: dict[str, np.ndarray] = {}

    for i, sd in enumerate(sess_dirs):
        for f in features:
            for role in role_keys:
                p = sd / f"{role}_{f}.npy"
                if not p.exists():
                    continue
                arr = np.load(p, mmap_mode="r").astype(np.float64, copy=False)
                if arr.ndim != 2:
                    raise ValueError(f"{p}: expected 2D, got {arr.shape}")
                # finite = NOT NaN/Inf AND magnitude < sentinel cutoff
                finite = np.isfinite(arr) & (np.abs(arr) < SENTINEL_ABS)
                arr_clean = np.where(finite, arr, 0.0)
                if f not in sums:
                    sums[f]   = arr_clean.sum(axis=0)
                    sumsqs[f] = (arr_clean * arr_clean).sum(axis=0)
                    counts[f] = finite.sum(axis=0).astype(np.int64)
                else:
                    sums[f]   += arr_clean.sum(axis=0)
                    sumsqs[f] += (arr_clean * arr_clean).sum(axis=0)
                    counts[f] += finite.sum(axis=0).astype(np.int64)
        if (i + 1) % 20 == 0 or (i + 1) == len(sess_dirs):
            print(f"  [{i + 1}/{len(sess_dirs)}]", file=sys.stderr)

    save_kwargs: dict[str, np.ndarray] = {}
    print(file=sys.stderr)
    print(f"{'feature':<14} {'dim':>5} {'n_min':>10} {'n_max':>10} "
          f"{'mean_min':>11} {'mean_max':>11} {'std_min':>11} {'std_max':>11}",
          file=sys.stderr)
    for f in features:
        if f not in sums:
            print(f"  WARN no data: {f}", file=sys.stderr)
            continue
        n = counts[f].astype(np.float64)
        n_safe = np.maximum(n, 1.0)
        mean = sums[f] / n_safe
        var  = sumsqs[f] / n_safe - mean * mean
        var  = np.maximum(var, 1e-12)
        std  = np.sqrt(var)
        save_kwargs[f"{f}_mean"] = mean.astype(np.float32)
        save_kwargs[f"{f}_std"]  = std.astype(np.float32)
        print(f"{f:<14} {mean.shape[0]:>5d} {int(n.min()):>10d} {int(n.max()):>10d} "
              f"{mean.min():>11.3e} {mean.max():>11.3e} "
              f"{std.min():>11.3e} {std.max():>11.3e}",
              file=sys.stderr)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **save_kwargs)
    print(f"\nSaved → {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
