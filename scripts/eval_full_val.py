"""Re-evaluate a trained best.pt on any val manifest combination.

Use cases:
  * mpiigi: training uses mpiigi_val_held.jsonl (7 rows) to avoid leakage,
    but we want to know the full mpiigi_val.jsonl (21 rows) score for
    comparison against DAPA's reported val number.
  * Smaller stride (=32) for finer overlap → tighter CCC estimate.
  * Combine multiple seeds and print individual + ensemble CCC.

Output: per-domain CCC / Kappa table + combined.

Example:
    python -m multimediate26.scripts.eval_full_val \
        --ckpts m26/output/phase1_whisper_noxi_noxij_seed{0,1,2}/best.pt \
        --features openface2,openface3,openpose,w2vbert2,egemapsv2,whisper,xlmr,swin,clip \
        --feature-stats /ossfs/.../feature_stats_phase1_whisper.npz \
        --manifests m26/manifests/noxi_val.jsonl,m26/manifests/noxi_j_val.jsonl \
        --stride 32
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import cohen_kappa_score
from torch.utils.data import DataLoader

from multimediate26.data.dataset import SessionDataset, collate as mm26_collate
from multimediate26.inference.run_test import build_model, load_ckpt_into, run_session


def ccc_score(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    """Lin's CCC on masked frames."""
    m = mask.astype(bool)
    if m.sum() < 2:
        return float("nan")
    p, t = pred[m], target[m]
    pm, tm = p.mean(), t.mean()
    ps2 = p.var()
    ts2 = t.var()
    pst = ((p - pm) * (t - tm)).mean()
    denom = ps2 + ts2 + (pm - tm) ** 2
    if denom < 1e-12:
        return float("nan")
    return float(2 * pst / denom)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", required=True,
                    help="comma-separated best.pt paths (1 = single, >1 = ensemble)")
    ap.add_argument("--features", required=True)
    ap.add_argument("--feature-stats", type=Path, required=True)
    ap.add_argument("--manifests", required=True,
                    help="comma-separated val manifest jsonl paths")
    ap.add_argument("--npz-root", type=Path,
                    default=Path("multimediate26/data_processed/npz_v3"))
    ap.add_argument("--base-config", type=Path,
                    default=Path("multimediate26/configs/base.yaml"))
    ap.add_argument("--feature-specs", type=Path,
                    default=Path("multimediate26/configs/feature_specs.yaml"))
    ap.add_argument("--window-len", type=int, default=512)
    ap.add_argument("--stride", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-partners", type=int, default=3)
    ap.add_argument("--num-workers", type=int, default=2)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpts = [Path(p) for p in args.ckpts.split(",") if p.strip()]
    manifests = [Path(p) for p in args.manifests.split(",") if p.strip()]
    feat_cfg = yaml.safe_load(Path(args.feature_specs).read_text())
    features = [f.strip() for f in args.features.split(",") if f.strip()]
    feature_dims = {f: feat_cfg["feature_dims"][f] for f in features}

    # Domains in the val set drive which prompts we load. Use the trained set.
    trained_domains = ["noxi", "noxi_j", "mpiigi", "pinsoro_cc", "pinsoro_cr"]
    has_cls = True

    print(f"=== full-val re-eval: {len(ckpts)} ckpt(s) × {len(manifests)} manifest(s)")

    # ckpt_idx → manifest → (session, role) → predictions array
    all_preds_per_ckpt: list[dict[str, dict[tuple, np.ndarray | tuple]]] = []
    # For loading targets (same across ckpts), we use the dataset directly.
    target_cache: dict[Path, dict[tuple, dict]] = {}

    for ckpt_idx, ckpt_path in enumerate(ckpts):
        print(f"\n--- ckpt {ckpt_idx+1}/{len(ckpts)}: {ckpt_path}")
        model = build_model(args, feature_dims, trained_domains,
                            enable_cls=has_cls).to(device)
        load_ckpt_into(model, ckpt_path, device)
        per_manifest: dict[str, dict[tuple, np.ndarray | tuple]] = {}

        for mp in manifests:
            import json
            domain = json.loads(mp.read_text().splitlines()[0])["domain"]
            is_cls = domain.startswith("pinsoro")
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
                print(f"  [skip] {mp.name}")
                continue
            loader = DataLoader(ds, batch_size=args.batch_size,
                                num_workers=args.num_workers, shuffle=False,
                                collate_fn=mm26_collate)
            preds = run_session(model, loader, device, is_cls)
            per_manifest[str(mp)] = preds
            print(f"  {mp.name}: {len(preds)} (session, role)")

            # Also collect ground-truth labels (only once across ckpts).
            if mp not in target_cache:
                target_cache[mp] = {}
                rows = [json.loads(l) for l in mp.read_text().splitlines() if l.strip()]
                for row in rows:
                    sess_dir = Path(row["out_dir"])
                    if not (sess_dir / "_DONE").exists():
                        continue
                    if is_cls:
                        target_cache[mp][(row["session_id"], row["target_role"])] = {
                            "label_task":   np.load(sess_dir / "label_task.npy"),
                            "label_social": np.load(sess_dir / "label_social.npy"),
                            "label_mask":   np.load(sess_dir / "label_mask.npy"),
                        }
                    else:
                        target_cache[mp][(row["session_id"], row["target_role"])] = {
                            "label":      np.load(sess_dir / "label.npy"),
                            "label_mask": np.load(sess_dir / "label_mask.npy"),
                        }

        all_preds_per_ckpt.append(per_manifest)
        del model
        torch.cuda.empty_cache()

    # ── Compute per-ckpt and ensemble metrics per manifest ──────────────
    print("\n=== metrics ===")
    headers = ["dataset"] + [f"seed{i}" for i in range(len(ckpts))] + ["ensemble"]
    rows = []

    for mp in manifests:
        import json
        domain = json.loads(mp.read_text().splitlines()[0])["domain"]
        is_cls = domain.startswith("pinsoro")
        if is_cls:
            # Task kappa + social kappa (mean) per ckpt + ensemble
            for metric_key in ("task", "social"):
                per_ckpt_kappa = []
                ens_task_preds = {}
                for ckpt_idx, per_manifest in enumerate(all_preds_per_ckpt):
                    preds = per_manifest.get(str(mp), {})
                    p_all, t_all, m_all = [], [], []
                    for key, val in preds.items():
                        if not isinstance(val, tuple):
                            continue
                        task_logits, social_logits = val
                        logits = task_logits if metric_key == "task" else social_logits
                        sess, role = key.split("/")[1], key.split("/")[2]
                        gt = target_cache[mp].get((sess, role))
                        if gt is None:
                            continue
                        lbl = gt[f"label_{metric_key}"]
                        msk = gt["label_mask"]
                        T = min(len(logits), len(lbl))
                        cls = logits[:T].argmax(-1)
                        p_all.append(cls)
                        t_all.append(lbl[:T])
                        m_all.append(msk[:T])
                        ens_task_preds.setdefault(key, []).append(logits[:T])
                    if p_all:
                        p = np.concatenate(p_all)
                        t = np.concatenate(t_all)
                        m = np.concatenate(m_all).astype(bool)
                        kappa = (float(cohen_kappa_score(t[m], p[m]))
                                 if m.sum() >= 2 else float("nan"))
                    else:
                        kappa = float("nan")
                    per_ckpt_kappa.append(kappa)
                # Ensemble: avg logits then argmax
                p_all, t_all, m_all = [], [], []
                for key, logits_list in ens_task_preds.items():
                    avg = np.mean(logits_list, axis=0)
                    sess, role = key.split("/")[1], key.split("/")[2]
                    gt = target_cache[mp].get((sess, role))
                    if gt is None:
                        continue
                    lbl = gt[f"label_{metric_key}"]
                    msk = gt["label_mask"]
                    T = min(len(avg), len(lbl))
                    p_all.append(avg[:T].argmax(-1))
                    t_all.append(lbl[:T])
                    m_all.append(msk[:T])
                if p_all:
                    p = np.concatenate(p_all)
                    t = np.concatenate(t_all)
                    m = np.concatenate(m_all).astype(bool)
                    ens_kappa = (float(cohen_kappa_score(t[m], p[m]))
                                 if m.sum() >= 2 else float("nan"))
                else:
                    ens_kappa = float("nan")
                rows.append([f"{domain}/{metric_key}_kappa", *per_ckpt_kappa, ens_kappa])
        else:
            per_ckpt_ccc = []
            ens_preds: dict[tuple, list[np.ndarray]] = {}
            for ckpt_idx, per_manifest in enumerate(all_preds_per_ckpt):
                preds = per_manifest.get(str(mp), {})
                p_all, t_all, m_all = [], [], []
                for key, pred in preds.items():
                    if isinstance(pred, tuple):
                        continue
                    sess, role = key.split("/")[1], key.split("/")[2]
                    gt = target_cache[mp].get((sess, role))
                    if gt is None:
                        continue
                    T = min(len(pred), len(gt["label"]))
                    p_all.append(pred[:T])
                    t_all.append(gt["label"][:T])
                    m_all.append(gt["label_mask"][:T])
                    ens_preds.setdefault(key, []).append(pred[:T])
                if p_all:
                    per_ckpt_ccc.append(
                        ccc_score(np.concatenate(p_all),
                                  np.concatenate(t_all),
                                  np.concatenate(m_all))
                    )
                else:
                    per_ckpt_ccc.append(float("nan"))

            # Ensemble: average per-session predictions then concat
            p_all, t_all, m_all = [], [], []
            for key, preds_list in ens_preds.items():
                Tmin = min(p.shape[0] for p in preds_list)
                avg = np.mean([p[:Tmin] for p in preds_list], axis=0)
                sess, role = key.split("/")[1], key.split("/")[2]
                gt = target_cache[mp].get((sess, role))
                if gt is None:
                    continue
                T = min(len(avg), len(gt["label"]))
                p_all.append(avg[:T])
                t_all.append(gt["label"][:T])
                m_all.append(gt["label_mask"][:T])
            if p_all:
                ens_ccc = ccc_score(np.concatenate(p_all),
                                    np.concatenate(t_all),
                                    np.concatenate(m_all))
            else:
                ens_ccc = float("nan")
            rows.append([f"{domain}/ccc", *per_ckpt_ccc, ens_ccc])

    # Print as table
    print()
    col_widths = [max(len(headers[i]), max(len(_fmt(r[i])) for r in rows) if rows else 0)
                  for i in range(len(headers))]
    print("  " + "  ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers)))
    print("  " + "  ".join("-" * w for w in col_widths))
    for r in rows:
        print("  " + "  ".join(_fmt(v).ljust(col_widths[i]) for i, v in enumerate(r)))

    # Combined metrics (per kind)
    print()
    ccc_vals = [r[-1] for r in rows if r[0].endswith("/ccc") and not math.isnan(r[-1])]
    kappa_vals = [r[-1] for r in rows if r[0].endswith("_kappa") and not math.isnan(r[-1])]
    if ccc_vals:
        print(f"  combined CCC   (ensemble): {np.mean(ccc_vals):.4f}")
    if kappa_vals:
        print(f"  combined Kappa (ensemble): {np.mean(kappa_vals):.4f}")
    return 0


def _fmt(v):
    if isinstance(v, str):
        return v
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "nan"
    return f"{v:.4f}"


if __name__ == "__main__":
    raise SystemExit(main())
