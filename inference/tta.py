"""Test-Time Adaptation: per-session prompt tuning.

Freeze the entire model and only optimize the 8-token domain prompt for each
test session using self-supervised losses (temporal smoothness + entropy
minimization + range penalty).

Optimized for speed:
  - Builds a per-session manifest (single row) so SessionDataset only loads
    windows from that one session.
  - TTA runs a fixed number of gradient steps (not full epochs over all
    windows) — randomly samples mini-batches from the session's windows.
  - Uses num_workers=0 to avoid fork overhead for small datasets.
  - After TTA, runs inference with the adapted prompt, then restores the
    original prompt before moving to the next session.

Typical speed: ~5-15s per session (10 steps, batch=8, 512-window).
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from multimediate26.data.dataset import SessionDataset, collate as mm26_collate
from multimediate26.inference.run_test import (
    build_model, load_ckpt_into, run_session,
)
from multimediate26.losses.ccc_loss import smoothness_loss


def _to_device(batch, device):
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        elif isinstance(v, dict):
            out[k] = {kk: vv.to(device) for kk, vv in v.items()}
        elif isinstance(v, list) and v and isinstance(v[0], dict):
            out[k] = [{kk: vv.to(device) for kk, vv in d.items()} for d in v]
        else:
            out[k] = v
    return out


def _find_prompt_params(model, domain: str) -> list[torch.nn.Parameter]:
    """Find prompt parameters for the given domain (with fallback)."""
    params = []
    for name, p in model.named_parameters():
        if "domain_prompt" in name and domain in name:
            params.append(p)
    if not params:
        from multimediate26.models.md_dapa import DomainPromptPool
        fb = DomainPromptPool.DEFAULT_FALLBACKS.get(domain, domain)
        for name, p in model.named_parameters():
            if "domain_prompt" in name and fb in name:
                params.append(p)
    return params


def _make_single_session_manifest(row: dict, tmp_dir: Path) -> Path:
    """Write a 1-line JSONL manifest for a single (session, role)."""
    p = tmp_dir / f"tta_{row['session_id']}_{row['target_role']}.jsonl"
    p.write_text(json.dumps(row) + "\n")
    return p


def tta_adapt(model, loader, prompt_params, device, steps: int, lr: float):
    """Run TTA gradient steps on the prompt. Returns original state dict."""
    orig = {id(p): p.data.clone() for p in prompt_params}

    for p in model.parameters():
        p.requires_grad_(False)
    for p in prompt_params:
        p.requires_grad_(True)

    optimizer = torch.optim.Adam(prompt_params, lr=lr)
    model.train()

    batch_iter = iter(loader)
    for _ in range(steps):
        try:
            batch = next(batch_iter)
        except StopIteration:
            batch_iter = iter(loader)
            batch = next(batch_iter)
        batch = _to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(batch)
            reg = out["reg"]
            mask = batch["label_mask"]
            loss = smoothness_loss(reg, mask)
            if "task_logits" in out:
                for key in ("task_logits", "social_logits"):
                    probs = F.softmax(out[key], dim=-1)
                    ent = -(probs * probs.clamp_min(1e-8).log()).sum(-1)
                    loss = loss + 0.1 * ent[mask].mean()
            loss = loss + 0.01 * (F.relu(reg - 0.9).mean() + F.relu(0.1 - reg).mean())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(prompt_params, 0.5)
        optimizer.step()

    for p in model.parameters():
        p.requires_grad_(False)
    return orig


def tta_restore(prompt_params, orig):
    for p in prompt_params:
        if id(p) in orig:
            p.data.copy_(orig[id(p)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--feature-stats", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--npz-root", type=Path,
                    default=Path("multimediate26/data_processed/npz_v3"))
    ap.add_argument("--base-config", type=Path,
                    default=Path("multimediate26/configs/base.yaml"))
    ap.add_argument("--feature-specs", type=Path,
                    default=Path("multimediate26/configs/feature_specs.yaml"))
    ap.add_argument("--window-len", type=int, default=512)
    ap.add_argument("--stride", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-partners", type=int, default=3)
    ap.add_argument("--tta-steps", type=int, default=10)
    ap.add_argument("--tta-lr", type=float, default=5e-4)
    ap.add_argument("--out-dir", type=Path, default=Path("/tmp/tta_out"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feat_cfg = yaml.safe_load(Path(args.feature_specs).read_text())
    features = [f.strip() for f in args.features.split(",") if f.strip()]
    feature_dims = {f: feat_cfg["feature_dims"][f] for f in features}

    trained_domains = ["noxi", "noxi_j", "mpiigi", "pinsoro_cc", "pinsoro_cr"]
    model = build_model(args, feature_dims, trained_domains,
                        enable_cls=True).to(device)
    load_ckpt_into(model, args.ckpt, device)

    rows = [json.loads(l) for l in args.manifest.read_text().splitlines() if l.strip()]
    domain = rows[0]["domain"]
    is_cls = domain.startswith("pinsoro")
    prompt_params = _find_prompt_params(model, domain)
    print(f"TTA: {args.manifest.name}, {len(rows)} rows, domain={domain}, "
          f"prompt params={len(prompt_params)}, steps={args.tta_steps}", flush=True)

    if not prompt_params:
        print("  no prompt params found — skipping TTA", flush=True)
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix="tta_"))

    for idx, row in enumerate(rows):
        t0 = time.time()
        sess, role = row["session_id"], row["target_role"]

        # 1. Build single-session manifest + dataset
        single_manifest = _make_single_session_manifest(row, tmp_dir)
        ds = SessionDataset(
            manifest_path=single_manifest,
            npz_root=args.npz_root,
            features=features,
            feature_dims=feature_dims,
            window_len=args.window_len,
            stride=args.stride,
            max_partners=args.max_partners,
            feature_stats=args.feature_stats,
            drop_label_unavailable=False,
            mode="eval",
        )
        if len(ds) == 0:
            print(f"  [{idx+1}/{len(rows)}] {sess}/{role}: empty dataset, skip", flush=True)
            continue

        loader = DataLoader(ds, batch_size=args.batch_size,
                            num_workers=0, shuffle=True,
                            collate_fn=mm26_collate)

        # 2. TTA adapt (use shuffled loader for diverse gradients)
        orig = tta_adapt(model, loader, prompt_params, device,
                         steps=args.tta_steps, lr=args.tta_lr)

        # 3. Inference with adapted prompt — reuse same dataset, just
        #    switch to sequential order (DataLoader is cheap to recreate,
        #    the dataset's NPZ arrays stay in memory).
        model.eval()
        inf_loader = DataLoader(ds, batch_size=args.batch_size * 2,
                                num_workers=0, shuffle=False,
                                collate_fn=mm26_collate)
        preds = run_session(model, inf_loader, device, is_cls)

        # 4. Save
        for key, val in preds.items():
            out_path = args.out_dir / f"{key.replace('/', '_')}.npy"
            if isinstance(val, tuple):
                np.save(out_path.with_suffix(".task.npy"), val[0])
                np.save(out_path.with_suffix(".social.npy"), val[1])
            else:
                np.save(out_path, val)

        # 5. Restore
        tta_restore(prompt_params, orig)
        dt = time.time() - t0
        print(f"  [{idx+1}/{len(rows)}] {sess}/{role}: "
              f"{len(ds)} windows, {dt:.1f}s", flush=True)

    print(f"\n=== TTA done → {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
