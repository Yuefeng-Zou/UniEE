"""Trainer for MD-DAPA.

Pipelines per DAPA-paper + USTC-IAT'25 ablation findings:

  * Adam, lr 5e-5
  * Linear warmup 400 steps → cosine annealing T_max=10 epochs
  * EMA decay 0.999 over 40 epochs (DAPA paper)
  * bf16 autocast, gradient clip 1.0
  * 8x sliding window: window_len=512, stride=64 train; eval stride may differ
  * batch_size 32 train / 256 val
  * DomainBalancedBatchSampler — per-batch single domain (DAPA prompt rule)
  * Per-domain CCC eval (NoXi, NoXi-J, MPIGI, NoXi-add); Combined = mean
  * best.pt selected by val Combined CCC (or per_domain_mean when applicable)

PInSoRo classification heads are TURNED OFF in Phase 1 (regression-only).
Trainer.compute_loss switches per batch's domain.
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

# Use sklearn's cohen_kappa_score to exactly match the official evaluation
# pipeline (multimediate26_official/baseline/4_TestingNN_fairness_per_session.py).
# Default args (weights=None, sample_weight=None) — same call signature as the
# official baseline.
from sklearn.metrics import cohen_kappa_score

from multimediate26.data.dataset import SessionDataset, collate
from multimediate26.data.sampler import (
    DomainBalancedBatchSampler, DomainGroupedEvalSampler,
)
from multimediate26.losses.ccc_loss import (
    ccc, ccc_loss, mse_loss_masked, smoothness_loss,
)
from multimediate26.losses.ordinal_contrastive import ordinal_contrastive
from multimediate26.models.md_dapa import MDDAPA, MDDAPAConfig
from multimediate26.models.domain_prompt import HierarchicalDomainPrompt


# ── EMA ──────────────────────────────────────────────────────────────────

class EMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.999) -> None:
        self.decay = float(decay)
        self.shadow = {
            n: p.detach().clone()
            for n, p in model.named_parameters() if p.requires_grad
        }

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            self.shadow[n].mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)

    @torch.no_grad()
    def apply_to(self, model: torch.nn.Module) -> dict:
        """Swap model params with EMA shadow. Returns backup of original weights."""
        backup = {}
        for n, p in model.named_parameters():
            if n in self.shadow:
                backup[n] = p.detach().clone()
                p.copy_(self.shadow[n])
        return backup

    @torch.no_grad()
    def restore(self, model: torch.nn.Module, backup: dict) -> None:
        for n, p in model.named_parameters():
            if n in backup:
                p.copy_(backup[n])


# ── lr schedule ──────────────────────────────────────────────────────────

def make_lr_lambda(warmup_steps: int, total_steps: int, min_lr_ratio: float = 0.0):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr_lambda


# ── losses ──────────────────────────────────────────────────────────────

def build_param_groups(model: MDDAPA, lr_groups: dict[str, float],
                       weight_decay: float) -> list[dict]:
    """Map model parameters to lr groups by name prefix matching."""
    prefix_to_group = {
        "projector.":     "projector",
        "group_fusion.":  "group_fusion",
        "down_proj.":     "projector",
        "dapa_layers.":   "dapa_layers",
        "partner_pool.":  "partner_pool",
        "reg_head.":      "heads",
        "cls_heads.":     "heads",
        "bridge.":        "bridge",
        "domain_prompt.": "domain_prompt",
    }
    groups: dict[str, list[torch.nn.Parameter]] = {}
    for g in lr_groups:
        groups[g] = []
    unmatched: list[torch.nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        matched = False
        for prefix, group_name in prefix_to_group.items():
            if name.startswith(prefix) and group_name in groups:
                groups[group_name].append(param)
                matched = True
                break
        if not matched:
            unmatched.append(param)

    param_groups = []
    for group_name, params in groups.items():
        if params:
            param_groups.append({
                "params": params,
                "lr": lr_groups[group_name],
                "weight_decay": weight_decay,
            })
    if unmatched:
        default_lr = lr_groups.get("projector", 1e-4)
        param_groups.append({
            "params": unmatched,
            "lr": default_lr,
            "weight_decay": weight_decay,
        })
    return param_groups


# ── losses (original) ──────────────────────────────────────────────────

def compute_loss(output: dict, batch: dict, weights: dict, domain: str,
                 task_class_weights: torch.Tensor | None = None,
                 social_class_weights: torch.Tensor | None = None,
                 ) -> tuple[torch.Tensor, dict]:
    """Dispatch loss by domain. Returns (total, per-term dict).

    Continuous domains (noxi / noxi_j / mpiigi):
        L = w_ccc * (1 - CCC) + w_mse * MSE + w_smooth * |dy|

    PInSoRo (cc / cr): NOT used in Phase 1. Phase 2:
        L = w_task * CE(task) + w_social * CE(social) + (optional) bridge CCC
    """
    label_mask = batch["label_mask"]                 # (B, T) bool
    label = batch["label"]                           # (B, T) float
    loss_dict: dict[str, torch.Tensor] = {}

    if domain in ("noxi", "noxi_j", "noxi_add", "mpiigi"):
        # mpii has a tiny label set; mpiigi only triggers when we explicitly
        # include mpii val rows in train (Phase 2 setting).
        pred = output["reg"]
        if label_mask.sum() < 2:
            zero = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
            return zero, {}
        loss_dict["ccc"]    = weights.get("ccc", 1.0)   * ccc_loss(pred, label, label_mask)
        if weights.get("mse", 0.0) > 0:
            loss_dict["mse"]    = weights["mse"]        * mse_loss_masked(pred, label, label_mask)
        if weights.get("smooth", 0.0) > 0:
            loss_dict["smooth"] = weights["smooth"]     * smoothness_loss(pred, label_mask)

    elif domain in ("pinsoro_cc", "pinsoro_cr"):
        if "task_logits" not in output:
            return torch.tensor(0.0, device=label.device), {}
        task_lbl   = batch["label_task"]
        social_lbl = batch["label_social"]
        # CE over (B, T, C) vs (B, T). Mask invalid frames by setting label=-100.
        mask = label_mask
        t_target = torch.where(mask, task_lbl, torch.full_like(task_lbl, -100))
        s_target = torch.where(mask, social_lbl, torch.full_like(social_lbl, -100))
        loss_dict["task_ce"]   = weights.get("task_ce", 1.0) * F.cross_entropy(
            output["task_logits"].reshape(-1, output["task_logits"].shape[-1]),
            t_target.reshape(-1), weight=task_class_weights,
            label_smoothing=0.1, ignore_index=-100,
        )
        loss_dict["social_ce"] = weights.get("social_ce", 1.0) * F.cross_entropy(
            output["social_logits"].reshape(-1, output["social_logits"].shape[-1]),
            s_target.reshape(-1), weight=social_class_weights,
            label_smoothing=0.1, ignore_index=-100,
        )
        # Phase 3: bridge loss — supervise the regression head on PInSoRo
        # frames using a hand-tuned pseudo-continuous target (precomputed
        # by build_session_npz into label_pseudo_cont.npy and surfaced as
        # batch["label_pseudo_cont"]).
        w_bridge = weights.get("bridge_ccc", 0.0)
        if w_bridge > 0 and "bridged_reg" in output and "label_pseudo_cont" in batch:
            pseudo_cont = batch["label_pseudo_cont"]            # (B, T)
            bridged = output["bridged_reg"]
            if mask.sum() >= 2:
                loss_dict["bridge_ccc"] = w_bridge * ccc_loss(
                    bridged, pseudo_cont, mask,
                )
    else:
        raise ValueError(f"unknown domain '{domain}'")

    # Phase 3: ordinal contrastive — applied to ALL domains with a
    # continuous (or pseudo-continuous) target. Pulls hidden features of
    # frames with similar engagement together. Weight default 0.
    w_ord = weights.get("ordinal", 0.0)
    if w_ord > 0 and "features" in output and label_mask.sum() >= 4:
        # Use the pseudo-cont target on PInSoRo, the true label elsewhere.
        if domain in ("pinsoro_cc", "pinsoro_cr") and "label_pseudo_cont" in batch:
            ord_target = batch["label_pseudo_cont"]
        elif domain in ("noxi", "noxi_j", "noxi_add", "mpiigi"):
            ord_target = label
        else:
            ord_target = None
        if ord_target is not None:
            loss_dict["ordinal"] = w_ord * ordinal_contrastive(
                output["features"], ord_target, label_mask,
            )

    total = sum(loss_dict.values()) if loss_dict else torch.tensor(0.0, device=label.device)
    return total, {k: v.detach().item() for k, v in loss_dict.items()}


# ── eval ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device,
             eval_domains: list[str]) -> dict:
    """Run model on each val window; aggregate per-domain CCC / kappa."""
    model.eval()
    # Per domain: collect (pred, target, mask) over all val frames.
    bins: dict[str, dict[str, list]] = {
        d: {"pred": [], "target": [], "mask": [],
            "task_pred": [], "task_target": [],
            "social_pred": [], "social_target": []}
        for d in eval_domains
    }
    for batch in loader:
        batch = _to_device(batch, device)
        domain = batch["domain"]
        if domain not in bins:
            continue
        out = model(batch)
        if domain in ("noxi", "noxi_j", "noxi_add", "mpiigi"):
            bins[domain]["pred"].append(out["reg"].detach().cpu())
            bins[domain]["target"].append(batch["label"].cpu())
            bins[domain]["mask"].append(batch["label_mask"].cpu())
        elif domain in ("pinsoro_cc", "pinsoro_cr"):
            if "task_logits" in out:
                bins[domain]["task_pred"].append(out["task_logits"].argmax(-1).cpu())
                bins[domain]["task_target"].append(batch["label_task"].cpu())
                bins[domain]["social_pred"].append(out["social_logits"].argmax(-1).cpu())
                bins[domain]["social_target"].append(batch["label_social"].cpu())
                bins[domain]["mask"].append(batch["label_mask"].cpu())

    metrics: dict[str, float] = {}
    for d in eval_domains:
        b = bins[d]
        if b["pred"]:
            p = torch.cat([x.reshape(-1) for x in b["pred"]])
            t = torch.cat([x.reshape(-1) for x in b["target"]])
            m = torch.cat([x.reshape(-1) for x in b["mask"]])
            metrics[f"{d}/ccc"] = float(ccc(p, t, m).item()) if m.sum() >= 2 else float("nan")
        elif b["task_pred"]:
            tp = torch.cat([x.reshape(-1) for x in b["task_pred"]]).numpy()
            tt = torch.cat([x.reshape(-1) for x in b["task_target"]]).numpy()
            sp = torch.cat([x.reshape(-1) for x in b["social_pred"]]).numpy()
            st = torch.cat([x.reshape(-1) for x in b["social_target"]]).numpy()
            m  = torch.cat([x.reshape(-1) for x in b["mask"]]).numpy()
            mask = m.astype(bool)
            if mask.sum() >= 2:
                metrics[f"{d}/task_kappa"]   = float(cohen_kappa_score(tt[mask], tp[mask]))
                metrics[f"{d}/social_kappa"] = float(cohen_kappa_score(st[mask], sp[mask]))

    # Combined: mean over per-domain primary metric.
    ccc_vals = [v for k, v in metrics.items() if k.endswith("/ccc") and not math.isnan(v)]
    kappa_vals = [v for k, v in metrics.items() if k.endswith("_kappa") and not math.isnan(v)]
    if ccc_vals:
        metrics["combined_ccc"] = float(np.mean(ccc_vals))
    if kappa_vals:
        metrics["combined_kappa"] = float(np.mean(kappa_vals))
    return metrics


def _to_device(batch: dict, device: torch.device) -> dict:
    out = dict(batch)
    out["target_feats"] = {k: v.to(device) for k, v in batch["target_feats"].items()}
    out["partner_feats"] = [
        {k: v.to(device) for k, v in slot.items()}
        for slot in batch["partner_feats"]
    ]
    for k in ("label", "label_mask", "attention_mask", "label_task", "label_social",
              "label_pseudo_cont"):
        if k in batch:
            out[k] = batch[k].to(device)
    return out


# ── training loop ───────────────────────────────────────────────────────

def train(args) -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Configs from yaml.
    base_cfg = yaml.safe_load(Path(args.base_config).read_text())
    feat_cfg = yaml.safe_load(Path(args.feature_specs).read_text())

    features = [f.strip() for f in args.features.split(",") if f.strip()]
    feature_dims = {f: feat_cfg["feature_dims"][f] for f in features}

    # Datasets.
    train_manifests = [Path(p) for p in args.train_manifests.split(",")]
    val_manifests   = [Path(p) for p in args.val_manifests.split(",")]
    # We need a single SessionDataset per role (train vs val), but build_session_npz
    # writes one manifest per (domain, split). Merge them into one combined
    # JSONL per role.
    train_combined = args.output_dir / "_train_manifest.jsonl"
    val_combined   = args.output_dir / "_val_manifest.jsonl"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_combined.write_text("\n".join(
        l for mp in train_manifests for l in Path(mp).read_text().splitlines()
    ))
    val_combined.write_text("\n".join(
        l for mp in val_manifests for l in Path(mp).read_text().splitlines()
    ))

    train_ds = SessionDataset(
        manifest_path=train_combined,
        npz_root=args.npz_root,
        features=features,
        window_len=args.window_len,
        stride=args.train_stride,
        mode="train",
        max_partners=args.max_partners,
        cache_sessions=args.cache_sessions,
        feature_stats=args.feature_stats,
        feature_dims=feature_dims,
    )
    val_ds = SessionDataset(
        manifest_path=val_combined,
        npz_root=args.npz_root,
        features=features,
        window_len=args.window_len,
        # Non-overlapping eval windows. With overlap, the same frame's
        # prediction is appended multiple times in evaluate(), inflating
        # the empirical variance term in CCC and skewing the reported
        # score. Official baseline does per-frame MLP (no windows) so its
        # micro CCC has each frame counted once; we match that by setting
        # eval stride == window_len. SessionDataset's tail-window logic
        # still covers the last sub-window via a single appended tail.
        stride=args.window_len,
        mode="eval",
        max_partners=args.max_partners,
        cache_sessions=args.cache_sessions,
        feature_stats=args.feature_stats,
        feature_dims=feature_dims,
    )
    print(f"  train: {train_ds.n_sessions} sessions, {len(train_ds)} windows")
    print(f"  val:   {val_ds.n_sessions} sessions, {len(val_ds)} windows")

    train_domains = sorted({w.domain for w in train_ds._windows})
    val_domains   = sorted({w.domain for w in val_ds._windows})
    print(f"  train domains: {train_domains}")
    print(f"  val domains:   {val_domains}")

    train_sampler = DomainBalancedBatchSampler(
        train_ds, batch_size=args.batch_size,
        n_batches=args.steps_per_epoch, alpha=0.5, seed=args.seed,
    )
    val_sampler = DomainGroupedEvalSampler(val_ds, batch_size=args.val_batch_size)

    train_loader = DataLoader(
        train_ds, batch_sampler=train_sampler, collate_fn=collate,
        num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_sampler=val_sampler, collate_fn=collate,
        num_workers=max(1, args.num_workers // 2), pin_memory=True,
    )

    # Model.
    groups = feat_cfg.get("groups") if args.use_group_fusion else None
    model_cfg = MDDAPAConfig(
        feature_dims=feature_dims,
        hidden_dim=base_cfg["model"]["hidden_dim"],
        n_dapa_layers=base_cfg["model"]["n_dapa_layers"],
        n_heads=base_cfg["model"]["n_heads"],
        dropout=base_cfg["model"]["dropout"],
        n_prompts_coarse=base_cfg["model"].get("n_prompts_coarse", 4),
        n_prompts_fine=base_cfg["model"].get("n_prompts_fine", 8),
        max_partners=args.max_partners,
        domains=sorted(set(train_domains + val_domains)),
        groups=groups,
        fusion_layers=base_cfg["model"].get("fusion_layers", 1),
        enable_classification=any(d.startswith("pinsoro") for d in train_domains),
        enable_bridge=args.enable_bridge,
        use_flat_prompt=args.use_flat_prompt,
        use_sum_partner=args.use_sum_partner,
    )
    model = MDDAPA(model_cfg).to(device)
    print(f"  model params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")

    # Optimizer.
    if args.use_layerwise_lr and "lr_groups" in base_cfg:
        param_groups = build_param_groups(
            model, base_cfg["lr_groups"],
            weight_decay=base_cfg["train"]["weight_decay"],
        )
        optimizer = torch.optim.AdamW(param_groups)
        print(f"  layer-wise lr: {len(param_groups)} groups")
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=base_cfg["train"]["weight_decay"],
        )
    total_steps = args.epochs * args.steps_per_epoch
    warmup_steps = base_cfg["train"]["warmup_steps"]
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=make_lr_lambda(warmup_steps, total_steps),
    )

    # EMA.
    ema = EMA(model, decay=base_cfg["train"]["ema_decay"])

    loss_weights = dict(base_cfg["loss_weights"])
    # Apply CLI overrides (Phase 3 enables bridge_ccc / ordinal without forking yaml).
    for kv in getattr(args, "loss_weight", []):
        k, _, v = kv.partition("=")
        if not v:
            raise SystemExit(f"--loss-weight expects KEY=VALUE, got {kv!r}")
        loss_weights[k] = float(v)

    best_metric = -1.0
    best_kappa = -1.0
    best_path = args.output_dir / "best.pt"
    best_kappa_path = args.output_dir / "best_kappa.pt"
    last_path = args.output_dir / "last.pt"

    autocast_dtype = torch.bfloat16 if base_cfg["train"].get("use_bf16", True) else torch.float32

    global_step = 0
    start_epoch = 0
    log_path = args.output_dir / "train.log"
    log_fp = open(log_path, "a")

    def log(msg: str) -> None:
        print(msg, flush=True)
        log_fp.write(msg + "\n")
        log_fp.flush()

    # ── Resume from last.pt if requested ──────────────────────────────
    # We restore model + EMA + global_step + start_epoch. We deliberately
    # do NOT save/restore optimizer or LR scheduler state — keeping ckpt
    # small (~80 MB) and the resume cost is acceptable: AdamW momenta
    # rebuild within a few hundred steps, and the LR scheduler is
    # re-stepped global_step times to land at the same lr as before the
    # crash. The cosine curve depends on (warmup_steps, total_steps)
    # which we hold constant — so the lr trajectory continues smoothly.
    # Also restore best_metric from best.pt so future ckpts only
    # overwrite best.pt when they truly improve on the pre-crash best.
    if args.resume is not None and args.init_from is not None:
        raise SystemExit("--resume and --init-from are mutually exclusive")

    if args.init_from is not None:
        if not args.init_from.exists():
            raise SystemExit(f"--init-from path does not exist: {args.init_from}")
        log(f"=== init from {args.init_from} (model weights only, fresh optimizer)")
        ckpt = torch.load(args.init_from, map_location=device, weights_only=False)
        missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
        if missing:
            log(f"    missing keys ({len(missing)}): "
                + ", ".join(missing[:5]) + (" ..." if len(missing) > 5 else ""))
        if unexpected:
            log(f"    unexpected keys ({len(unexpected)}): "
                + ", ".join(unexpected[:5]) + (" ..." if len(unexpected) > 5 else ""))
        # Sync EMA to the loaded weights so the first eval has a meaningful
        # checkpoint to fall back on if step 0 happens to be the best.
        for n, p in model.named_parameters():
            if n in ema.shadow:
                ema.shadow[n].copy_(p.detach())
        log(f"    init done. start_epoch=0, global_step=0, fresh cosine schedule")

    if args.resume is not None and args.resume.exists():
        log(f"=== resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        if "ema" in ckpt:
            for n, p in ckpt["ema"].items():
                if n in ema.shadow:
                    ema.shadow[n].copy_(p.to(device))
        global_step = int(ckpt.get("step", 0))
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        # Fast-forward LR scheduler to the resumed step.
        for _ in range(global_step):
            scheduler.step()
        if best_path.exists():
            best_ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
            best_metric = float(best_ckpt.get("best_metric", -1.0))
            log(f"    restored best_metric={best_metric:.4f} from best.pt")
        log(f"    resume @ epoch {start_epoch} / global_step {global_step} / lr={optimizer.param_groups[0]['lr']:.2e}")

    log(f"=== start training: epochs={args.epochs}, total_steps={total_steps}, "
        f"warmup={warmup_steps}, device={device}, autocast={autocast_dtype}, "
        f"start_epoch={start_epoch}")

    save_epoch_set = set()
    if args.save_epochs:
        save_epoch_set = {int(e.strip()) for e in args.save_epochs.split(",")}

    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0 = time.time()
        epoch_losses: dict[str, list[float]] = {}
        for batch in train_loader:
            batch = _to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, dtype=autocast_dtype,
                                    enabled=device.type == "cuda"):
                out = model(batch)
                loss, parts = compute_loss(out, batch, loss_weights, batch["domain"])
            if not torch.isfinite(loss):
                log(f"  WARN non-finite loss at step {global_step} domain={batch['domain']} — skipping")
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           base_cfg["train"].get("grad_clip", 1.0))
            optimizer.step()
            scheduler.step()
            ema.update(model)
            global_step += 1
            for k, v in parts.items():
                epoch_losses.setdefault(k, []).append(v)
            epoch_losses.setdefault("total", []).append(float(loss.detach().item()))

        # Per-epoch eval (with EMA weights).
        backup = ema.apply_to(model)
        metrics = evaluate(model, val_loader, device, eval_domains=val_domains)
        ema.restore(model, backup)

        lr_now = optimizer.param_groups[0]["lr"]
        loss_str = " ".join(f"{k}={np.mean(v):.4f}" for k, v in epoch_losses.items())
        metric_str = " ".join(f"{k}={v:.4f}" for k, v in sorted(metrics.items()))
        log(f"  epoch {epoch:3d}  lr={lr_now:.2e}  {loss_str}  ({time.time()-t0:.0f}s)")
        log(f"             val: {metric_str}")

        # Selection metric.
        sel_ccc = metrics.get("combined_ccc", -1.0)
        sel_kappa = metrics.get("combined_kappa", -1.0)
        ckpt_payload = {"model": model.state_dict(),
                        "ema":   ema.shadow,
                        "cfg":   asdict(model_cfg),
                        "step":  global_step,
                        "epoch": epoch}
        torch.save(ckpt_payload, last_path)
        if save_epoch_set and epoch in save_epoch_set:
            ep_path = args.output_dir / f"epoch_{epoch}.pt"
            torch.save(ckpt_payload, ep_path)
            log(f"             ✓ saved epoch checkpoint → {ep_path.name}")
        if sel_ccc > best_metric:
            best_metric = sel_ccc
            torch.save({**ckpt_payload, "best_metric": best_metric}, best_path)
            log(f"             ✓ new best combined_ccc={sel_ccc:.4f} → {best_path.name}")
        if sel_kappa > best_kappa:
            best_kappa = sel_kappa
            torch.save({**ckpt_payload, "best_kappa": best_kappa}, best_kappa_path)
            log(f"             ✓ new best combined_kappa={sel_kappa:.4f} → {best_kappa_path.name}")

    log(f"=== done. best combined_ccc = {best_metric:.4f}, best combined_kappa = {best_kappa:.4f}")
    log_fp.close()
    return 0


def _main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-config",    type=Path, default=Path("multimediate26/configs/base.yaml"))
    ap.add_argument("--feature-specs",  type=Path, default=Path("multimediate26/configs/feature_specs.yaml"))
    ap.add_argument("--features",       type=str,  required=True,
                    help="comma-separated feature names (must subset feature_specs preset)")
    ap.add_argument("--train-manifests", type=str, required=True,
                    help="comma-separated JSONL manifests (per (domain, split))")
    ap.add_argument("--val-manifests",   type=str, required=True)
    ap.add_argument("--npz-root",       type=Path, required=True)
    ap.add_argument("--feature-stats",  type=Path, default=None,
                    help="per-channel z-score stats from compute_feature_stats.py")
    ap.add_argument("--output-dir",     type=Path, required=True)
    ap.add_argument("--seed",           type=int, default=0)
    ap.add_argument("--epochs",         type=int, default=40)
    ap.add_argument("--steps-per-epoch", type=int, default=200)
    ap.add_argument("--batch-size",     type=int, default=32)
    ap.add_argument("--val-batch-size", type=int, default=64)
    ap.add_argument("--window-len",     type=int, default=512)
    ap.add_argument("--train-stride",   type=int, default=64,
                    help="8x sliding window: stride=window_len/8 (USTC-IAT'25 finding)")
    ap.add_argument("--max-partners",   type=int, default=1)
    ap.add_argument("--cache-sessions", type=int, default=4)
    ap.add_argument("--num-workers",    type=int, default=1)
    ap.add_argument("--lr",             type=float, default=5e-5)
    ap.add_argument("--enable-bridge",  action="store_true", default=False,
                    help="Phase 2: enable LearnableBridge for PInSoRo → reg")
    ap.add_argument("--use-layerwise-lr", action="store_true", default=False,
                    help="Use per-module learning rates from base.yaml lr_groups")
    ap.add_argument("--use-group-fusion", action="store_true", default=False,
                    help="Use ModalityGroupFusion instead of concat+Linear down-proj")
    ap.add_argument("--use-flat-prompt", action="store_true", default=False,
                    help="Ablation: use flat DomainPromptPool instead of HierarchicalDomainPrompt")
    ap.add_argument("--use-sum-partner", action="store_true", default=False,
                    help="Ablation: use parameter-free sum instead of MultiPartnerPooling")
    ap.add_argument("--resume",         type=Path, default=None,
                    help="resume from a checkpoint (last.pt). Loads model+EMA, "
                    "sets start_epoch=ckpt['epoch']+1, advances scheduler to ckpt['step']. "
                    "Optimizer/scheduler state are NOT serialized — lr re-warms from "
                    "ckpt['step'] position in the cosine curve. Acceptable cost given "
                    "the unplanned crash and our 40-epoch horizon.")
    ap.add_argument("--init-from",      type=Path, default=None,
                    help="Phase 2: warm-start model weights from a Phase 1 best.pt. "
                    "Loads model state_dict ONLY (no EMA/step/epoch); training restarts "
                    "at epoch 0 with a fresh optimizer and full cosine schedule. "
                    "Missing/extra keys (e.g. new domain prompts, new partner slots) "
                    "are tolerated via strict=False. Mutually exclusive with --resume.")
    ap.add_argument("--save-epochs", type=str, default=None,
                    help="Comma-separated epoch numbers to save checkpoints, e.g. '2,7,9'. "
                    "Saves as epoch_{N}.pt in addition to best/last.")
    ap.add_argument("--loss-weight", action="append", default=[],
                    metavar="KEY=VALUE",
                    help="override a single loss_weights entry from base.yaml. "
                    "Repeatable. Example: --loss-weight bridge_ccc=0.3 "
                    "--loss-weight ordinal=0.1")
    args = ap.parse_args()
    return train(args)


if __name__ == "__main__":
    raise SystemExit(_main())
