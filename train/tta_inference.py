"""TTA + inference pipeline for regression domains.

For each seed checkpoint:
  1. Load model
  2. Per test domain: adapt LayerNorm + domain_prompt (3 epochs, consistency + smoothness)
  3. Run inference with adapted model
  4. Save predictions

Then ensemble all seeds' adapted predictions.

Usage:
    python -m multimediate26.train.tta_inference \
        --checkpoints multimediate26/output/phase2_v2arch_11feat_seed0/best.pt,multimediate26/output/phase2_v2arch_11feat_seed1/best.pt,multimediate26/output/phase2_v2arch_11feat_seed2/best.pt \
        --out-dir submission/_tta_ensemble \
        --gpu 0
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from scipy.signal import savgol_filter

from multimediate26.data.dataset import SessionDataset, collate
from multimediate26.data.sampler import DomainGroupedEvalSampler
from multimediate26.train.inference import (
    load_model, predict_session, write_regression_csv,
    DOMAIN_TEST_MANIFESTS, DOMAIN_TO_SUBMISSION_DIR, DOMAIN_FALLBACKS,
)
from torch.utils.data import DataLoader


def tta_adapt(model: nn.Module, loader: DataLoader, device: torch.device,
              domain: str, n_epochs: int = 3, lr: float = 1e-5,
              noise_std: float = 0.01) -> None:
    """Adapt LayerNorm + domain_prompt on unlabeled test data."""
    original_state = copy.deepcopy(model.state_dict())

    for name, p in model.named_parameters():
        p.requires_grad = (
            "norm" in name.lower() or "domain_prompt" in name
        )

    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        print(f"    No trainable params for TTA, skipping")
        return

    optimizer = torch.optim.AdamW(trainable, lr=lr)
    print(f"    TTA: {sum(p.numel() for p in trainable)} trainable params, {n_epochs} epochs")

    model.train()
    for ep in range(n_epochs):
        total_loss = 0
        n_batches = 0
        for batch in loader:
            batch = _to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(batch)
                # Consistency: perturbed input should give same output
                noisy_batch = _add_noise(batch, noise_std)
                out_noisy = model(noisy_batch)
                consistency = (out["reg"] - out_noisy["reg"]).pow(2).mean()
                # Smoothness
                smoothness = (out["reg"][:, 1:] - out["reg"][:, :-1]).abs().mean()
                loss = consistency + 0.05 * smoothness

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        print(f"    TTA epoch {ep}: loss={total_loss/max(1,n_batches):.6f}")

    # Restore requires_grad
    for p in model.parameters():
        p.requires_grad = True


def _add_noise(batch, std):
    noisy = dict(batch)
    noisy["target_feats"] = {
        k: v + torch.randn_like(v) * std for k, v in batch["target_feats"].items()
    }
    noisy["partner_feats"] = [
        {k: v + torch.randn_like(v) * std for k, v in p.items()}
        for p in batch["partner_feats"]
    ]
    return noisy


def _to_device(batch, device):
    out = dict(batch)
    out["target_feats"] = {k: v.to(device) for k, v in batch["target_feats"].items()}
    out["partner_feats"] = [{k: v.to(device) for k, v in slot.items()} for slot in batch["partner_feats"]]
    for k in ("label", "label_mask", "attention_mask"):
        if k in batch:
            out[k] = batch[k].to(device)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", type=str, required=True, help="Comma-separated checkpoint paths")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--base-config", type=Path, default=Path("multimediate26/configs/base.yaml"))
    ap.add_argument("--feature-specs", type=Path, default=Path("multimediate26/configs/feature_specs.yaml"))
    ap.add_argument("--features", type=str,
                    default="openface2,openface3,openpose,w2vbert2,egemapsv2,whisper,xlmr,videomae,dino,swin,clip")
    ap.add_argument("--feature-stats", type=Path, default=Path("experiments/_feature_stats/feature_stats_v4_whisper_full.npz"))
    ap.add_argument("--npz-root", type=Path, default=Path("multimediate26/data_processed/npz_v4"))
    ap.add_argument("--window-len", type=int, default=512)
    ap.add_argument("--max-partners", type=int, default=3)
    ap.add_argument("--tta-epochs", type=int, default=3)
    ap.add_argument("--tta-lr", type=float, default=1e-5)
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    features = [f.strip() for f in args.features.split(",")]
    feat_cfg = yaml.safe_load(args.feature_specs.read_text())
    feature_dims = {f: feat_cfg["feature_dims"][f] for f in features}
    ckpt_paths = [Path(p.strip()) for p in args.checkpoints.split(",")]

    domains = ["noxi", "noxi_add", "noxi_j", "mpiigi"]

    all_seed_preds = []

    for ci, ckpt_path in enumerate(ckpt_paths):
        print(f"\n=== Seed {ci}: {ckpt_path.name} ===")
        seed_preds = {}

        for domain in domains:
            print(f"  Domain: {domain}")
            model = load_model(ckpt_path, device, args.base_config, args.feature_specs,
                               features, use_group_fusion=True)

            # Create test dataset for TTA
            manifest_path = DOMAIN_TEST_MANIFESTS.get(domain)
            if not manifest_path or not Path(manifest_path).exists():
                continue

            # TTA: adapt on test data
            test_combined = args.out_dir / f"_tta_manifest_{domain}.jsonl"
            args.out_dir.mkdir(parents=True, exist_ok=True)
            test_combined.write_text(Path(manifest_path).read_text())

            tta_ds = SessionDataset(
                manifest_path=test_combined, npz_root=args.npz_root,
                features=features, window_len=args.window_len, stride=args.window_len // 2,
                mode="train", drop_label_unavailable=False,
                max_partners=args.max_partners, cache_sessions=4,
                feature_stats=args.feature_stats, feature_dims=feature_dims,
            )

            if len(tta_ds) > 0:
                tta_loader = DataLoader(
                    tta_ds, batch_size=16, shuffle=True, collate_fn=collate,
                    num_workers=0, drop_last=True,
                )
                tta_adapt(model, tta_loader, device, domain,
                          n_epochs=args.tta_epochs, lr=args.tta_lr)

            # Inference with adapted model
            with open(manifest_path) as f:
                rows = [json.loads(l) for l in f if l.strip()]

            sub_dir = DOMAIN_TO_SUBMISSION_DIR.get(domain, domain)

            for row in rows:
                sess_dir = Path(row["out_dir"])
                if not sess_dir.exists():
                    continue
                preds = predict_session(
                    model, sess_dir, features, feature_dims,
                    args.feature_stats, domain, row["target_role"],
                    row.get("partner_roles", []),
                    args.max_partners, args.window_len, device,
                )
                if not preds:
                    continue

                key = f"{sub_dir}/{row['session_id']}/{row['target_role']}"
                if key not in seed_preds:
                    seed_preds[key] = []
                seed_preds[key].append(preds["reg"])

                # Also save individual seed predictions
                seed_out = args.out_dir / f"seed{ci}" / sub_dir / row["session_id"]
                seed_out.mkdir(parents=True, exist_ok=True)
                np.savetxt(
                    seed_out / f"{row['target_role']}.engagement.prediction.csv",
                    preds["reg"], fmt="%.6f",
                )

        all_seed_preds.append(seed_preds)

    # Ensemble
    print(f"\n=== Ensemble {len(all_seed_preds)} seeds ===")
    ensemble_dir = args.out_dir / "ensemble"
    n_files = 0
    all_keys = set()
    for sp in all_seed_preds:
        all_keys.update(sp.keys())

    for key in sorted(all_keys):
        preds = []
        for sp in all_seed_preds:
            if key in sp:
                preds.extend(sp[key])
        if preds:
            avg = np.mean(preds, axis=0)
            avg = np.clip(avg, 0, 1)
            parts = key.split("/")
            out_path = ensemble_dir / parts[0] / parts[1] / f"{parts[2]}.engagement.prediction.csv"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.savetxt(out_path, avg, fmt="%.6f")
            n_files += 1

    print(f"  {n_files} ensemble files written → {ensemble_dir}")


if __name__ == "__main__":
    main()
