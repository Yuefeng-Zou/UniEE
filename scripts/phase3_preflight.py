"""Run before launching Phase 3 — catches missing prereqs that would only
manifest after the trainer has spun up.

Checks (fail-fast):
  1. Phase 2 best.pt exists for each requested seed.
  2. Phase 3 feature_stats exists and has all 9 feature columns.
  3. PInSoRo NPZ has label_pseudo_cont.npy (bridge loss needs it).
  4. PInSoRo NPZ has label_task.npy + label_social.npy (CE loss needs them).
  5. Trainer module imports cleanly with bridge_ccc + ordinal weights enabled.
  6. At least 3 GPUs free (excluding GPU 2 watchdog).
  7. multimediate26.losses.ordinal_contrastive and models.heads.LearnableBridge
     are importable.
  8. base.yaml loss_weights has bridge_ccc + ordinal keys (so CLI override works).
"""
from __future__ import annotations

import argparse
import importlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import yaml


def check(label: str, cond: bool, hint: str = "") -> bool:
    mark = "✓" if cond else "✗"
    print(f"  {mark} {label}" + (f" — {hint}" if (hint and not cond) else ""))
    return cond


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase2-output-template",
                    default="multimediate26/output/phase2_whisper_noxi_noxij_mpiigi_seed{seed}/best.pt")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--feature-stats",
                    default="/ossfs/workspace/mm26_stats/feature_stats_phase3_whisper.npz")
    ap.add_argument("--npz-root", default="multimediate26/data_processed/npz_v3")
    ap.add_argument("--base-config", default="multimediate26/configs/base.yaml")
    args = ap.parse_args()

    ok = True
    print("=== Phase 3 preflight ===")

    # 1. Phase 2 best.pt
    print("\n[1] Phase 2 best.pt presence:")
    for s in args.seeds.split(","):
        s = s.strip()
        p = Path(args.phase2_output_template.format(seed=s))
        ok &= check(f"seed {s}: {p}", p.exists(),
                    "still training? launch Phase 3 after each seed finishes")

    # 2. Phase 3 feature stats
    print("\n[2] Phase 3 feature stats:")
    sp = Path(args.feature_stats)
    if not check(f"file: {sp}", sp.exists(), "run compute_feature_stats first"):
        ok = False
    else:
        stats = np.load(sp)
        expected = ["openface2", "openface3", "openpose", "w2vbert2",
                    "egemapsv2", "whisper", "xlmr", "swin", "clip"]
        for f in expected:
            ok &= check(f"  has {f}_mean", f"{f}_mean" in stats.files)

    # 3 + 4. PInSoRo NPZ has bridge + cls labels
    print("\n[3+4] PInSoRo NPZ label artifacts (sample 5 dirs):")
    pin_root = Path(args.npz_root)
    sample_dirs = []
    for sub in ("pinsoro_cc", "pinsoro_cr"):
        d = pin_root / sub
        if d.exists():
            for sess in sorted(d.iterdir())[:3]:
                if not sess.is_dir():
                    continue
                for role in sess.iterdir():
                    if role.is_dir() and (role / "_DONE").exists():
                        sample_dirs.append(role)
                        break
                if len(sample_dirs) >= 5:
                    break
        if len(sample_dirs) >= 5:
            break
    if not sample_dirs:
        check("no PInSoRo NPZ to check", False, "rebuild pinsoro NPZ")
        ok = False
    else:
        for sd in sample_dirs:
            for f in ("label_pseudo_cont.npy", "label_task.npy", "label_social.npy"):
                ok &= check(f"{sd.relative_to(pin_root)}/{f}",
                            (sd / f).exists(),
                            "rebuild pinsoro NPZ to regenerate")

    # 5. trainer imports
    print("\n[5] Trainer & loss imports:")
    try:
        importlib.import_module("multimediate26.train.trainer")
        ok &= check("multimediate26.train.trainer", True)
    except Exception as e:
        ok &= check("multimediate26.train.trainer", False, repr(e))
    try:
        from multimediate26.losses.ordinal_contrastive import ordinal_contrastive
        from multimediate26.models.heads import LearnableBridge
        ok &= check("ordinal_contrastive + LearnableBridge", True)
    except Exception as e:
        ok &= check("losses+heads import", False, repr(e))

    # 6. GPUs
    print("\n[6] GPU availability:")
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True,
        )
        free_gpus = []
        for line in result.stdout.strip().split("\n"):
            idx, mem = [s.strip() for s in line.split(",")]
            if int(idx) == 2:           # GPU 2 is reserved for system/watchdog
                continue
            if int(mem) < 5000:
                free_gpus.append(idx)
        ok &= check(f"≥3 free GPUs (got {len(free_gpus)}: {free_gpus})",
                    len(free_gpus) >= 3,
                    "kill Phase 2 trainers or wait for them to finish")
    except subprocess.CalledProcessError as e:
        ok &= check("nvidia-smi", False, str(e))

    # 7. base.yaml has the new loss_weights keys
    print("\n[7] base.yaml loss_weights:")
    cfg = yaml.safe_load(Path(args.base_config).read_text())
    lw = cfg.get("loss_weights", {})
    for k in ("bridge_ccc", "ordinal"):
        ok &= check(f"  key '{k}'", k in lw, "edit base.yaml to add")

    print(f"\n{'='*40}")
    print("READY ✓" if ok else "NOT READY ✗ — fix above before launching Phase 3")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
