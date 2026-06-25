"""PInSoRo specialist trainer — lightweight model with Focal Loss + class weights.

Usage:
    CUDA_VISIBLE_DEVICES=0 python -m multimediate26.train.pinsoro_trainer \
        --output-dir multimediate26/output/pinsoro_v2_seed0
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from sklearn.metrics import cohen_kappa_score

from multimediate26.data.dataset import SessionDataset, collate
from multimediate26.data.sampler import DomainBalancedBatchSampler, DomainGroupedEvalSampler
from multimediate26.models.pinsoro_model import PInSoRoModel, PInSoRoConfig, FocalLoss


def compute_class_weights(dataset, n_task, n_social, domain_filter=None):
    """Compute inverse-freq class weights from training data."""
    task_counts = torch.zeros(n_task)
    social_counts = torch.zeros(n_social)
    for w in dataset._windows:
        if domain_filter and w.domain != domain_filter:
            continue
        sess_dir = Path(w.session_dir)
        try:
            task = np.load(sess_dir / "label_task.npy")
            social = np.load(sess_dir / "label_social.npy")
            mask = np.load(sess_dir / "label_mask.npy").astype(bool)
            for c in task[mask]:
                if 0 <= c < n_task:
                    task_counts[int(c)] += 1
            for c in social[mask]:
                if 0 <= c < n_social:
                    social_counts[int(c)] += 1
        except:
            pass
    task_w = task_counts.sum() / (n_task * task_counts.clamp(min=1))
    social_w = social_counts.sum() / (n_social * social_counts.clamp(min=1))
    return task_w, social_w


@torch.no_grad()
def evaluate(model, loader, device, eval_domains):
    model.eval()
    bins = {d: {"task_pred": [], "task_target": [], "social_pred": [], "social_target": [], "mask": []}
            for d in eval_domains}
    for batch in loader:
        batch = _to_device(batch, device)
        domain = batch["domain"]
        if domain not in bins:
            continue
        out = model(batch)
        if "task_logits" in out:
            bins[domain]["task_pred"].append(out["task_logits"].argmax(-1).cpu())
            bins[domain]["task_target"].append(batch["label_task"].cpu())
            bins[domain]["social_pred"].append(out["social_logits"].argmax(-1).cpu())
            bins[domain]["social_target"].append(batch["label_social"].cpu())
            bins[domain]["mask"].append(batch["label_mask"].cpu())

    metrics = {}
    for d in eval_domains:
        b = bins[d]
        if not b["task_pred"]:
            continue
        tp = torch.cat([x.reshape(-1) for x in b["task_pred"]]).numpy()
        tt = torch.cat([x.reshape(-1) for x in b["task_target"]]).numpy()
        sp = torch.cat([x.reshape(-1) for x in b["social_pred"]]).numpy()
        st = torch.cat([x.reshape(-1) for x in b["social_target"]]).numpy()
        m = torch.cat([x.reshape(-1) for x in b["mask"]]).numpy().astype(bool)
        if m.sum() >= 2:
            metrics[f"{d}/task_kappa"] = float(cohen_kappa_score(tt[m], tp[m]))
            metrics[f"{d}/social_kappa"] = float(cohen_kappa_score(st[m], sp[m]))

    kappa_vals = [v for k, v in metrics.items() if "_kappa" in k]
    if kappa_vals:
        metrics["combined_kappa"] = float(np.mean(kappa_vals))
    return metrics


def _to_device(batch, device):
    out = dict(batch)
    out["target_feats"] = {k: v.to(device) for k, v in batch["target_feats"].items()}
    out["partner_feats"] = [{k: v.to(device) for k, v in slot.items()} for slot in batch["partner_feats"]]
    for k in ("label", "label_mask", "attention_mask", "label_task", "label_social"):
        if k in batch:
            out[k] = batch[k].to(device)
    return out


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    feat_cfg = yaml.safe_load(Path(args.feature_specs).read_text())
    features = [f.strip() for f in args.features.split(",")]
    feature_dims = {f: feat_cfg["feature_dims"][f] for f in features}

    train_manifests = [Path(p) for p in args.train_manifests.split(",")]
    val_manifests = [Path(p) for p in args.val_manifests.split(",")]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_combined = args.output_dir / "_train_manifest.jsonl"
    val_combined = args.output_dir / "_val_manifest.jsonl"
    train_combined.write_text("\n".join(l for mp in train_manifests for l in Path(mp).read_text().splitlines()))
    val_combined.write_text("\n".join(l for mp in val_manifests for l in Path(mp).read_text().splitlines()))

    train_ds = SessionDataset(
        manifest_path=train_combined, npz_root=args.npz_root, features=features,
        window_len=args.window_len, stride=args.train_stride, mode="train",
        max_partners=args.max_partners, cache_sessions=args.cache_sessions,
        feature_stats=args.feature_stats, feature_dims=feature_dims,
    )
    val_ds = SessionDataset(
        manifest_path=val_combined, npz_root=args.npz_root, features=features,
        window_len=args.window_len, stride=args.window_len, mode="eval",
        max_partners=args.max_partners, cache_sessions=args.cache_sessions,
        feature_stats=args.feature_stats, feature_dims=feature_dims,
    )

    train_domains = sorted({w.domain for w in train_ds._windows})
    val_domains = sorted({w.domain for w in val_ds._windows})

    train_sampler = DomainBalancedBatchSampler(train_ds, args.batch_size, args.steps_per_epoch, alpha=0.5, seed=args.seed)
    val_sampler = DomainGroupedEvalSampler(val_ds, batch_size=args.val_batch_size)
    train_loader = DataLoader(train_ds, batch_sampler=train_sampler, collate_fn=collate, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_sampler=val_sampler, collate_fn=collate, num_workers=max(1, args.num_workers // 2), pin_memory=True)

    cfg = PInSoRoConfig(feature_dims=feature_dims, hidden_dim=args.hidden_dim,
                        lstm_layers=args.lstm_layers, dropout=args.dropout)
    model = PInSoRoModel(cfg).to(device)
    print(f"  model params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")

    # Class weights (precomputed inverse-freq from training data)
    cc_task_w = torch.tensor([0.39, 1.52, 1.42, 12.99])
    cc_social_w = torch.tensor([2.74, 3.37, 0.88, 1.50, 0.39])
    cr_task_w = torch.tensor([0.70, 4.88, 0.43, 16.95])
    cr_social_w = torch.tensor([0.49, 6.83, 0.59, 8.50, 1.0])

    cc_task_loss = FocalLoss(gamma=2.0, weight=cc_task_w.to(device))
    cc_social_loss = FocalLoss(gamma=2.0, weight=cc_social_w.to(device))
    cr_task_loss = FocalLoss(gamma=2.0, weight=cr_task_w.to(device))
    cr_social_loss = FocalLoss(gamma=2.0, weight=cr_social_w.to(device))

    print(f"  CC task weights: {cc_task_w.tolist()}")
    print(f"  CC social weights: {cc_social_w.tolist()}")
    print(f"  CR task weights: {cr_task_w.tolist()}")
    print(f"  CR social weights: {cr_social_w.tolist()}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * args.steps_per_epoch
    warmup = min(500, total_steps // 10)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min(s / max(1, warmup), 1.0) *
                                                   (0.5 * (1 + math.cos(math.pi * max(0, s - warmup) / max(1, total_steps - warmup)))))

    # EMA
    ema_decay = 0.999
    ema_shadow = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}

    best_kappa = -1.0
    best_path = args.output_dir / "best_kappa.pt"
    last_path = args.output_dir / "last.pt"
    log_path = args.output_dir / "train.log"
    log_fp = open(log_path, "a")

    def log(msg):
        print(msg, flush=True)
        log_fp.write(msg + "\n")
        log_fp.flush()

    if args.init_from:
        log(f"=== init from {args.init_from}")
        ckpt = torch.load(args.init_from, map_location=device, weights_only=False)
        state = ckpt.get("model", ckpt)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            log(f"    missing: {len(missing)} keys")
        if unexpected:
            log(f"    unexpected: {len(unexpected)} keys")

    log(f"=== start: epochs={args.epochs}, steps={total_steps}, device={device}")

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        epoch_losses = []
        for batch in train_loader:
            batch = _to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            domain = batch["domain"]

            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(batch)
                mask = batch["label_mask"]
                t_target = torch.where(mask, batch["label_task"], torch.full_like(batch["label_task"], -100))
                s_target = torch.where(mask, batch["label_social"], torch.full_like(batch["label_social"], -100))
                t_logits = out["task_logits"].reshape(-1, out["task_logits"].shape[-1])
                s_logits = out["social_logits"].reshape(-1, out["social_logits"].shape[-1])

                if domain == "pinsoro_cc":
                    loss = cc_task_loss(t_logits, t_target.reshape(-1)) + cc_social_loss(s_logits, s_target.reshape(-1))
                else:
                    loss = cr_task_loss(t_logits, t_target.reshape(-1)) + cr_social_loss(s_logits, s_target.reshape(-1))

            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            with torch.no_grad():
                for n, p in model.named_parameters():
                    if n in ema_shadow:
                        ema_shadow[n].mul_(ema_decay).add_(p.detach(), alpha=1 - ema_decay)
            epoch_losses.append(loss.item())

        # Eval with EMA
        backup = {}
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in ema_shadow:
                    backup[n] = p.detach().clone()
                    p.copy_(ema_shadow[n])

        metrics = evaluate(model, val_loader, device, val_domains)

        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in backup:
                    p.copy_(backup[n])

        avg_loss = np.mean(epoch_losses) if epoch_losses else 0
        metric_str = " ".join(f"{k}={v:.4f}" for k, v in sorted(metrics.items()))
        log(f"  epoch {epoch:3d}  lr={optimizer.param_groups[0]['lr']:.2e}  loss={avg_loss:.4f}  ({time.time()-t0:.0f}s)")
        log(f"             val: {metric_str}")

        ckpt = {"model": model.state_dict(), "ema": ema_shadow, "cfg": asdict(cfg), "epoch": epoch}
        torch.save(ckpt, last_path)

        sel = metrics.get("combined_kappa", -1.0)
        if sel > best_kappa:
            best_kappa = sel
            torch.save({**ckpt, "best_kappa": best_kappa}, best_path)
            log(f"             ✓ new best combined_kappa={sel:.4f} → {best_path.name}")

    log(f"=== done. best combined_kappa = {best_kappa:.4f}")
    log_fp.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-specs", type=Path, default=Path("multimediate26/configs/feature_specs.yaml"))
    ap.add_argument("--features", type=str, default="openface2,openface3,openpose,w2vbert2,egemapsv2,xlmr,videomae,dino,swin,clip")
    ap.add_argument("--train-manifests", type=str, required=True)
    ap.add_argument("--val-manifests", type=str, required=True)
    ap.add_argument("--npz-root", type=Path, default=Path("multimediate26/data_processed/npz_v4"))
    ap.add_argument("--feature-stats", type=Path, default=None)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--steps-per-epoch", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--val-batch-size", type=int, default=64)
    ap.add_argument("--window-len", type=int, default=512)
    ap.add_argument("--train-stride", type=int, default=64)
    ap.add_argument("--max-partners", type=int, default=3)
    ap.add_argument("--cache-sessions", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--hidden-dim", type=int, default=256)
    ap.add_argument("--lstm-layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--init-from", type=Path, default=None)
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
