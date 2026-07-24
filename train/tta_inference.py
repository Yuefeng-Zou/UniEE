"""TTA + inference pipeline for regression domains.

For each test domain:
  1. Reload the same validation-selected checkpoint.
  2. Adapt LayerNorm and domain-prompt parameters on unlabeled test windows.
  3. Run inference with the adapted model.
  4. Save predictions directly in the official submission layout.

Usage:
    python -m multimediate26.train.tta_inference \
        --checkpoint multimediate26/output/paper_phase3_11feat/best.pt \
        --out-dir submission/paper_tta \
        --gpu 0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

from multimediate26.data.dataset import SessionDataset, collate
from multimediate26.train.inference import (
    load_model,
    predict_session,
    DOMAIN_TEST_MANIFESTS,
    DOMAIN_TO_SUBMISSION_DIR,
)
from torch.utils.data import DataLoader


def tta_adapt(model: nn.Module, loader: DataLoader, device: torch.device,
              domain: str, n_epochs: int = 3, lr: float = 1e-5,
              noise_std: float = 0.01) -> None:
    """Adapt LayerNorm + domain_prompt on unlabeled test data."""
    original_requires_grad = {
        id(param): param.requires_grad for param in model.parameters()
    }
    trainable = _configure_tta_parameters(model)
    if not trainable:
        print(f"    No trainable params for TTA, skipping")
        for param in model.parameters():
            param.requires_grad = original_requires_grad[id(param)]
        return

    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.0)
    print(f"    TTA: {sum(p.numel() for p in trainable)} trainable params, {n_epochs} epochs")

    # Keep dropout disabled so the consistency term measures input noise only.
    model.eval()
    try:
        for ep in range(n_epochs):
            total_loss = 0
            n_batches = 0
            for batch in loader:
                batch = _to_device(batch, device)
                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=device.type == "cuda",
                ):
                    out = model(batch)
                    noisy_batch = _add_noise(batch, noise_std)
                    out_noisy = model(noisy_batch)
                    valid = batch["attention_mask"]
                    consistency = (
                        (out["reg"] - out_noisy["reg"]).pow(2)
                        * valid.to(out["reg"].dtype)
                    ).sum() / valid.sum().clamp_min(1)
                    pair_valid = valid[:, 1:] & valid[:, :-1]
                    smoothness = (
                        (out["reg"][:, 1:] - out["reg"][:, :-1]).abs()
                        * pair_valid.to(out["reg"].dtype)
                    ).sum() / pair_valid.sum().clamp_min(1)
                    loss = consistency + 0.05 * smoothness

                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1

            print(f"    TTA epoch {ep}: loss={total_loss/max(1,n_batches):.6f}")
    finally:
        for param in model.parameters():
            param.requires_grad = original_requires_grad[id(param)]


def _configure_tta_parameters(model: nn.Module) -> list[nn.Parameter]:
    """Freeze the model except LayerNorm affine terms and domain prompts."""
    for param in model.parameters():
        param.requires_grad = False

    selected: list[nn.Parameter] = []
    selected_ids: set[int] = set()

    for module in model.modules():
        if isinstance(module, nn.LayerNorm):
            for param in module.parameters(recurse=False):
                param.requires_grad = True
                if id(param) not in selected_ids:
                    selected.append(param)
                    selected_ids.add(id(param))

    for name, param in model.named_parameters():
        if name.startswith("domain_prompt."):
            param.requires_grad = True
            if id(param) not in selected_ids:
                selected.append(param)
                selected_ids.add(id(param))
    return selected


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
    if "target_modality_present" in batch:
        out["target_modality_present"] = {
            k: v.to(device) for k, v in batch["target_modality_present"].items()
        }
    if "partner_modality_present" in batch:
        out["partner_modality_present"] = [
            {k: v.to(device) for k, v in slot.items()}
            for slot in batch["partner_modality_present"]
        ]
    for k in ("label", "label_mask", "attention_mask"):
        if k in batch:
            out[k] = batch[k].to(device)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True,
                    help="validation-selected checkpoint used to initialize every domain")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--base-config", type=Path, default=Path("multimediate26/configs/base.yaml"))
    ap.add_argument("--feature-specs", type=Path, default=Path("multimediate26/configs/feature_specs.yaml"))
    ap.add_argument("--features", type=str,
                    default="openface2,openface3,openpose,w2vbert2,egemapsv2,whisper,xlmr,videomae,dino,swin,clip")
    ap.add_argument(
        "--feature-stats",
        type=Path,
        default=Path(
            "multimediate26/experiments/_feature_stats/"
            "feature_stats_v4_whisper_full.npz"
        ),
    )
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

    domains = ["noxi", "noxi_add", "noxi_j", "mpiigi"]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    n_files = 0

    for domain in domains:
        print(f"\n=== Domain: {domain} ===")
        manifest_path = DOMAIN_TEST_MANIFESTS.get(domain)
        if not manifest_path or not Path(manifest_path).exists():
            print("  Test manifest not found, skipping")
            continue

        # Reloading guarantees that every domain starts from the same model.
        model = load_model(
            args.checkpoint,
            device,
            args.base_config,
            args.feature_specs,
            features,
            use_group_fusion=True,
        )

        tta_ds = SessionDataset(
            manifest_path=Path(manifest_path), npz_root=args.npz_root,
            features=features, window_len=args.window_len,
            stride=args.window_len // 2, mode="train",
            drop_label_unavailable=False, max_partners=args.max_partners,
            cache_sessions=4, feature_stats=args.feature_stats,
            feature_dims=feature_dims,
        )
        if len(tta_ds) > 0:
            tta_loader = DataLoader(
                tta_ds, batch_size=16, shuffle=True, collate_fn=collate,
                num_workers=0, drop_last=False,
            )
            tta_adapt(model, tta_loader, device, domain,
                      n_epochs=args.tta_epochs, lr=args.tta_lr)

        with open(manifest_path) as f:
            rows = [json.loads(line) for line in f if line.strip()]
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

            prediction = np.clip(preds["reg"], 0, 1)
            out_path = (
                args.out_dir
                / sub_dir
                / row["session_id"]
                / f"{row['target_role']}.engagement.prediction.csv"
            )
            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.savetxt(out_path, prediction, fmt="%.6f")
            n_files += 1

    print(f"\n  {n_files} prediction files written → {args.out_dir}")


if __name__ == "__main__":
    main()
