"""Run sliding-window inference on test-split manifests and emit submission CSVs.

Pipeline per (domain, session, role):
  1. Build SessionDataset in eval mode (stride=window_len // 4 by default).
  2. Forward each window through model → (W, 1) regression or (W, C) logits.
  3. Reconstruct per-frame prediction by overlap-averaging (Hann-window for
     regression, max-vote for classification).
  4. Trim to actual T (drop pad frames at the tail).
  5. For 3-seed ensemble, the loop is wrapped: we run all seeds, average
     regression predictions per frame, sum-then-argmax classification logits.
  6. Write to submission/<official-folder>/<session>/<role>.{...}.prediction.csv

Output directory: --out-dir (default: multimediate26/submission/run_TIMESTAMP/)

Usage:
    python -m multimediate26.inference.run_test \
        --ckpts m26/output/phase3_5domain_whisper_seed{0,1,2}/best.pt \
        --features openface2,openface3,openpose,w2vbert2,egemapsv2,whisper,xlmr,swin,clip \
        --feature-stats /ossfs/workspace/mm26_stats/feature_stats_phase3_whisper.npz \
        --manifests multimediate26/manifests/noxi_test.jsonl,...,pinsoro_cr_test.jsonl \
        --out-dir multimediate26/submission/run_01
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

# Local
from multimediate26.data.dataset import SessionDataset, collate as mm26_collate
from multimediate26.models.md_dapa import MDDAPA, MDDAPAConfig
from multimediate26.submission.writer import (
    write_regression, write_classification, savgol_smooth,
)


def hann_weight(W: int) -> np.ndarray:
    """Cosine ramp peaking in the middle. Reduces seam artifacts on overlap."""
    if W < 2:
        return np.ones(W, dtype=np.float32)
    n = np.arange(W, dtype=np.float32)
    return 0.5 * (1.0 - np.cos(2.0 * np.pi * (n + 1) / (W + 1)))


def build_model(args, feature_dims: dict[str, int], domains: list[str],
                enable_cls: bool) -> MDDAPA:
    base_cfg = yaml.safe_load(Path(args.base_config).read_text())["model"]
    cfg = MDDAPAConfig(
        feature_dims=feature_dims,
        hidden_dim=base_cfg["hidden_dim"],
        n_dapa_layers=base_cfg["n_dapa_layers"],
        n_heads=base_cfg["n_heads"],
        dropout=0.0,                # eval — no dropout
        prompt_len=base_cfg["n_prompts_fine"],
        max_partners=args.max_partners,
        domains=domains,
        enable_classification=enable_cls,
        enable_bridge=enable_cls,   # only matters for cls; harmless when false
    )
    return MDDAPA(cfg)


def load_ckpt_into(model: torch.nn.Module, ckpt_path: Path, device) -> None:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("ema") or ckpt["model"]
    miss, unexp = model.load_state_dict(state, strict=False)
    if miss:
        print(f"  [warn] missing keys ({len(miss)}): {miss[:4]}...", file=sys.stderr)
    if unexp:
        print(f"  [warn] unexpected keys ({len(unexp)}): {unexp[:4]}...", file=sys.stderr)


def ckpt_val_score(ckpt_path: Path) -> float:
    """Best combined_ccc reported by the training run that produced this ckpt.

    Strategy:
      1. Prefer ``ckpt["best_metric"]`` if present (newer trainer writes it).
      2. Else parse the matching train.log under
         /ossfs/workspace/run_logs/<exp_name>/train.log for the highest
         'new best combined_ccc=' line.
      3. Else return 1.0 (uniform weight).

    Returns a positive float used as the un-normalized ensemble weight.
    """
    # 1. inline ckpt metric
    try:
        c = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if isinstance(c.get("best_metric"), (int, float)) and c["best_metric"] > 0:
            return float(c["best_metric"])
    except Exception:
        pass
    # 2. parse train.log next to the ckpt or under /ossfs/.../run_logs
    exp_name = ckpt_path.parent.name        # e.g. phase2_whisper_..._seed0
    candidates = [
        ckpt_path.parent / "train.log",
        Path("/ossfs/workspace/run_logs") / exp_name / "train.log",
    ]
    for log in candidates:
        if not log.exists():
            continue
        try:
            txt = log.read_text(errors="ignore")
        except Exception:
            continue
        best = 0.0
        for m in re.finditer(r"new best combined_ccc=([0-9.]+)", txt):
            v = float(m.group(1))
            if v > best:
                best = v
        if best > 0:
            return best
    # 3. fallback uniform
    return 1.0


def run_session(model, loader, device, is_classification: bool):
    """Returns per-session predictions:
        is_classification=False → dict[(domain, session, role)] = (pred_T, count_T)
        is_classification=True  → dict[(domain, session, role)] = (task_logits_T, social_logits_T, count_T)
    Accumulates with Hann weights over windows.
    """
    accum: dict[tuple, list] = defaultdict(lambda: None)
    W = None
    hann = None

    model.eval()
    with torch.no_grad():
        for batch in loader:
            target_feats = {k: v.to(device) for k, v in batch["target_feats"].items()}
            partner_feats = [
                {k: v.to(device) for k, v in slot.items()}
                for slot in batch["partner_feats"]
            ]
            model_batch = {
                "target_feats":   target_feats,
                "partner_feats":  partner_feats,
                "partner_present": batch["partner_present"],
                "attention_mask": batch["attention_mask"].to(device),
                "domain":         batch["domain"],
            }
            out = model(model_batch)

            if W is None:
                W = batch["attention_mask"].shape[1]
                hann = hann_weight(W)

            session_keys  = batch["session_key"]
            window_starts = batch["window_start"]
            valid_ends    = batch["valid_end"]

            if not is_classification:
                preds = out["reg"].detach().cpu().numpy()         # (B, W)
            else:
                task_logits   = out["task_logits"].detach().cpu().numpy()   # (B, W, 4)
                social_logits = out["social_logits"].detach().cpu().numpy() # (B, W, 5)

            B = len(session_keys)
            for i in range(B):
                key = session_keys[i]
                w_start = int(window_starts[i])
                valid_end = int(valid_ends[i])
                valid_len = valid_end - w_start
                if valid_len <= 0:
                    continue

                # Lazy-init accumulator per session — we need session T.
                if accum[key] is None:
                    # Probe T from npz dir (cheap, mmap).
                    # session_key looks like "{domain}/{sess}/{role}". We don't
                    # know NPZ root here; trust that session length is at
                    # least max(valid_end) seen across windows. So just track
                    # by 'max valid_end' as we go.
                    if not is_classification:
                        accum[key] = [np.zeros(valid_end, dtype=np.float64),
                                      np.zeros(valid_end, dtype=np.float64)]
                    else:
                        accum[key] = [np.zeros((valid_end, 4), dtype=np.float64),
                                      np.zeros((valid_end, 5), dtype=np.float64),
                                      np.zeros(valid_end, dtype=np.float64)]
                # Grow if a later window extends past current length.
                if not is_classification:
                    cur_T = len(accum[key][0])
                    if valid_end > cur_T:
                        for j in range(2):
                            accum[key][j] = np.concatenate([
                                accum[key][j], np.zeros(valid_end - cur_T, dtype=np.float64)
                            ])
                    w = hann[:valid_len]
                    accum[key][0][w_start:valid_end] += preds[i, :valid_len] * w
                    accum[key][1][w_start:valid_end] += w
                else:
                    cur_T = len(accum[key][2])
                    if valid_end > cur_T:
                        accum[key][0] = np.concatenate([
                            accum[key][0], np.zeros((valid_end - cur_T, 4), dtype=np.float64)])
                        accum[key][1] = np.concatenate([
                            accum[key][1], np.zeros((valid_end - cur_T, 5), dtype=np.float64)])
                        accum[key][2] = np.concatenate([
                            accum[key][2], np.zeros(valid_end - cur_T, dtype=np.float64)])
                    w = hann[:valid_len]
                    accum[key][0][w_start:valid_end] += task_logits[i, :valid_len]   * w[:, None]
                    accum[key][1][w_start:valid_end] += social_logits[i, :valid_len] * w[:, None]
                    accum[key][2][w_start:valid_end] += w

    # Normalize.
    out_preds: dict[tuple, np.ndarray] = {}
    for key, vals in accum.items():
        if vals is None:
            continue
        if not is_classification:
            pred, cnt = vals
            cnt = np.maximum(cnt, 1e-8)
            out_preds[key] = (pred / cnt).astype(np.float32)
        else:
            t_logits, s_logits, cnt = vals
            cnt = np.maximum(cnt, 1e-8)
            out_preds[key] = (
                (t_logits / cnt[:, None]).astype(np.float32),
                (s_logits / cnt[:, None]).astype(np.float32),
            )
    return out_preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", required=True,
                    help="comma-separated best.pt paths for seed-ensemble (any number)")
    ap.add_argument("--features", required=True)
    ap.add_argument("--feature-stats", type=Path, required=True)
    ap.add_argument("--manifests", required=True,
                    help="comma-separated test manifest jsonl paths")
    ap.add_argument("--npz-root", type=Path,
                    default=Path("multimediate26/data_processed/npz_v3"))
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--base-config", type=Path,
                    default=Path("multimediate26/configs/base.yaml"))
    ap.add_argument("--feature-specs", type=Path,
                    default=Path("multimediate26/configs/feature_specs.yaml"))
    ap.add_argument("--window-len", type=int, default=512)
    ap.add_argument("--stride",     type=int, default=128,
                    help="eval stride; small → more overlap, more compute")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-partners", type=int, default=3)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--smooth", action="store_true",
                    help="Savgol smooth regression outputs (window=25, poly=3)")
    ap.add_argument("--smooth-window", type=int, default=25)
    ap.add_argument("--ensemble-weighting", choices=("uniform", "val_ccc", "val_ccc_squared"),
                    default="val_ccc",
                    help="how to combine seeds. uniform = simple mean; "
                    "val_ccc = weight ∝ ckpt's best val combined_ccc; "
                    "val_ccc_squared = ∝ (combined_ccc)^2 (sharper towards top seed)")
    ap.add_argument("--enable-tta", action="store_true",
                    help="per-session TTA on PInSoRo (classification) domains only")
    ap.add_argument("--tta-steps", type=int, default=10)
    ap.add_argument("--tta-lr", type=float, default=5e-4)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpts = [Path(p) for p in args.ckpts.split(",") if p.strip()]
    manifests = [Path(p) for p in args.manifests.split(",") if p.strip()]
    feat_cfg = yaml.safe_load(Path(args.feature_specs).read_text())
    features = [f.strip() for f in args.features.split(",") if f.strip()]
    feature_dims = {f: feat_cfg["feature_dims"][f] for f in features}

    # Detect domains present across all manifests (so the model has all prompts).
    rows = []
    for mp in manifests:
        rows.extend(json.loads(l) for l in mp.read_text().splitlines() if l.strip())
    all_domains_test = sorted({r["domain"] for r in rows})
    # Domains the trained model knew about. Training-time was 5 (no noxi_add).
    trained_domains = ["noxi", "noxi_j", "mpiigi", "pinsoro_cc", "pinsoro_cr"]
    has_cls = any(d.startswith("pinsoro") for d in trained_domains)

    # Load feature dims; SessionDataset handles its own normalization via feature_stats path.
    print(f"=== inference: {len(ckpts)} ckpt(s) × {len(manifests)} manifest(s) → {args.out_dir}")
    print(f"  test domains : {all_domains_test}")
    print(f"  trained dom. : {trained_domains}")

    # For each ckpt, run inference for every manifest, accumulate ensemble.
    # ensemble[(domain, session, role)] → list of per-seed preds (regression)
    # or list of per-seed (task_logits, social_logits) (classification).
    ensemble: dict[tuple, list] = defaultdict(list)
    cls_keys: set[tuple] = set()
    reg_keys: set[tuple] = set()

    for ckpt_idx, ckpt_path in enumerate(ckpts):
        print(f"\n--- [{ckpt_idx+1}/{len(ckpts)}] {ckpt_path}")
        model = build_model(args, feature_dims, trained_domains,
                            enable_cls=has_cls).to(device)
        load_ckpt_into(model, ckpt_path, device)

        for mp in manifests:
            domain = json.loads(mp.read_text().splitlines()[0])["domain"]
            is_cls = domain.startswith("pinsoro")

            if is_cls and args.enable_tta:
                # Per-session TTA: iterate sessions individually so each
                # gets its own adapted prompt, then restore before the next.
                from multimediate26.inference.tta import (
                    tta_adapt, tta_restore, _find_prompt_params,
                    _make_single_session_manifest,
                )
                import tempfile
                prompt_params = _find_prompt_params(model, domain)
                tmp_dir = Path(tempfile.mkdtemp(prefix="tta_"))
                mp_rows = [json.loads(l)
                           for l in mp.read_text().splitlines() if l.strip()]
                n_tta = 0
                for row in mp_rows:
                    single_mp = _make_single_session_manifest(row, tmp_dir)
                    ds_single = SessionDataset(
                        manifest_path=single_mp,
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
                    if len(ds_single) == 0:
                        continue
                    tta_loader = DataLoader(
                        ds_single, batch_size=args.batch_size,
                        num_workers=0, shuffle=True,
                        collate_fn=mm26_collate)
                    orig = tta_adapt(model, tta_loader, prompt_params, device,
                                     steps=args.tta_steps, lr=args.tta_lr)
                    model.eval()
                    inf_loader = DataLoader(
                        ds_single, batch_size=args.batch_size * 2,
                        num_workers=0, shuffle=False,
                        collate_fn=mm26_collate)
                    preds = run_session(model, inf_loader, device, is_cls)
                    for key, val in preds.items():
                        ensemble[key].append(val)
                        cls_keys.add(key)
                    tta_restore(prompt_params, orig)
                    n_tta += 1
                print(f"    {mp.name}: {n_tta} sessions with TTA")
            else:
                ds = SessionDataset(
                    manifest_path=mp,
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
                    print(f"    [skip] {mp.name}: 0 windows")
                    continue
                loader = DataLoader(ds, batch_size=args.batch_size,
                                    num_workers=args.num_workers, shuffle=False,
                                    collate_fn=mm26_collate)
                preds = run_session(model, loader, device, is_cls)
                print(f"    {mp.name}: {len(preds)} (session, role) predicted")
                for key, val in preds.items():
                    ensemble[key].append(val)
                    (cls_keys if is_cls else reg_keys).add(key)

        # Free model before next ckpt loads.
        del model
        torch.cuda.empty_cache()

    # ── Compute per-ckpt ensemble weights (constant across all sessions/domains) ──
    raw_weights = [ckpt_val_score(p) for p in ckpts]
    if args.ensemble_weighting == "uniform":
        weights = [1.0] * len(ckpts)
    elif args.ensemble_weighting == "val_ccc_squared":
        weights = [w * w for w in raw_weights]
    else:                                                        # val_ccc
        weights = list(raw_weights)
    wsum = sum(weights)
    weights = [w / wsum for w in weights]
    print(f"\n=== ensemble weights ({args.ensemble_weighting}) ===")
    for c, raw, w in zip(ckpts, raw_weights, weights):
        print(f"  {c.parent.name:50s}  val_ccc={raw:.4f}  weight={w:.4f}")

    # ── Aggregate ensemble & write to submission CSVs ───────────────────
    print(f"\n=== writing submission to {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    n_reg = n_cls = 0
    for key in reg_keys:
        domain, sess, role = key.split("/")
        preds_list = ensemble[key]
        if not preds_list:
            continue
        # Each ckpt may produce a slightly different T (last-window padding);
        # trim to common min. Ensemble weight has the same index as ckpts list.
        min_T = min(p.shape[0] for p in preds_list)
        # ensemble[key] is a list ordered by ckpt_idx — so weights align.
        avg = np.zeros(min_T, dtype=np.float64)
        for w, pred in zip(weights[:len(preds_list)], preds_list):
            avg += w * pred[:min_T]
        # Re-normalize if some ckpts had no prediction for this key.
        active_w = sum(weights[:len(preds_list)])
        avg = (avg / max(active_w, 1e-9)).astype(np.float32)
        if args.smooth:
            avg = savgol_smooth(avg, window=args.smooth_window, poly=3)
        avg = np.clip(avg, 0.0, 1.0)
        write_regression(args.out_dir, domain, sess, role, avg)
        n_reg += 1
    for key in cls_keys:
        domain, sess, role = key.split("/")
        preds_list = ensemble[key]
        if not preds_list:
            continue
        min_T = min(p[0].shape[0] for p in preds_list)
        # Weighted average of logits, then argmax.
        avg_task   = np.zeros((min_T, 4), dtype=np.float64)
        avg_social = np.zeros((min_T, 5), dtype=np.float64)
        for w, (t, s) in zip(weights[:len(preds_list)], preds_list):
            avg_task   += w * t[:min_T]
            avg_social += w * s[:min_T]
        active_w = sum(weights[:len(preds_list)])
        avg_task   /= max(active_w, 1e-9)
        avg_social /= max(active_w, 1e-9)
        write_classification(args.out_dir, domain, sess, role,
                             task_pred=avg_task, social_pred=avg_social)
        n_cls += 1

    print(f"  wrote {n_reg} regression + {n_cls} classification CSV pairs → {args.out_dir}")


if __name__ == "__main__":
    main()
