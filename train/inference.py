"""Generate test set predictions in official submission format.

For each test domain, loads the best checkpoint, runs sliding-window inference
with Hann-window overlap-add, and writes per-session CSV files matching the
official format:

  Regression (noxi/noxi_j/noxi_add/mpiigi):
    {out_dir}/{domain}/{session_id}/{role}.engagement.annotation.csv
    Content: one float per line, %.6f, T lines (all frames)

  Classification (pinsoro_cc/pinsoro_cr):
    {out_dir}/{domain}/{session_id}/{role}.task_engagement.engagement.prediction.csv
    {out_dir}/{domain}/{session_id}/{role}.social_engagement.engagement.prediction.csv
    Content: one string label per line, T lines
    PInSoRo CR: only purple role

Usage:
    python -m multimediate26.train.inference \\
        --checkpoint multimediate26/output/phase3_v2arch_11feat_seed0/best.pt \\
        --out-dir submission/v2arch_phase3_seed0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.signal import savgol_filter

from multimediate26.data.dataset import SessionDataset, collate
from multimediate26.models.md_dapa import MDDAPA, MDDAPAConfig


TASK_INV_MAP = {0: "goaloriented", 1: "aimless", 2: "adultseeking", 3: "noplay"}
SOCIAL_INV_MAP = {0: "solitary", 1: "onlooker", 2: "parallel", 3: "associative", 4: "cooperative"}

DOMAIN_TEST_MANIFESTS = {
    "noxi":       "multimediate26/manifests/noxi_test.jsonl",
    "noxi_add":   "multimediate26/manifests/noxi_add_test.jsonl",
    "noxi_j":     "multimediate26/manifests/noxi_j_test.jsonl",
    "mpiigi":     "multimediate26/manifests/mpiigi_test.jsonl",
    "pinsoro_cc": "multimediate26/manifests/pinsoro_cc_test.jsonl",
    "pinsoro_cr": "multimediate26/manifests/pinsoro_cr_test.jsonl",
}

DOMAIN_TO_SUBMISSION_DIR = {
    "noxi":       "noxi-base",
    "noxi_add":   "noxi-additional",
    "noxi_j":     "noxi-j",
    "mpiigi":     "mpiigroupinteraction",
    "pinsoro_cc": "pinsoro-cc",
    "pinsoro_cr": "pinsoro-cr",
}

DOMAIN_FALLBACKS = {"noxi_add": "noxi"}


def load_model(ckpt_path: Path, device: torch.device,
               base_config: Path, feature_specs: Path,
               features: list[str], use_group_fusion: bool,
               use_flat_prompt: bool = False,
               use_sum_partner: bool = False,
               ) -> MDDAPA:
    base_cfg = yaml.safe_load(base_config.read_text())
    feat_cfg = yaml.safe_load(feature_specs.read_text())
    feature_dims = {f: feat_cfg["feature_dims"][f] for f in features}
    groups = feat_cfg.get("groups") if use_group_fusion else None

    all_domains = ["noxi", "noxi_j", "noxi_add", "mpiigi", "pinsoro_cc", "pinsoro_cr"]
    cfg = MDDAPAConfig(
        feature_dims=feature_dims,
        hidden_dim=base_cfg["model"]["hidden_dim"],
        n_dapa_layers=base_cfg["model"]["n_dapa_layers"],
        n_heads=base_cfg["model"]["n_heads"],
        dropout=0.0,
        n_prompts_coarse=base_cfg["model"].get("n_prompts_coarse", 4),
        n_prompts_fine=base_cfg["model"].get("n_prompts_fine", 8),
        max_partners=base_cfg["model"].get("max_partners", 4),
        domains=all_domains,
        groups=groups,
        fusion_layers=base_cfg["model"].get("fusion_layers", 1),
        enable_classification=True,
        enable_bridge=True,
        use_flat_prompt=use_flat_prompt,
        use_sum_partner=use_sum_partner,
    )
    model = MDDAPA(cfg).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("ema", ckpt["model"])
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  missing keys ({len(missing)}): {missing[:5]}")
    if unexpected:
        print(f"  unexpected keys ({len(unexpected)}): {unexpected[:5]}")

    model.eval()
    return model


@torch.no_grad()
def predict_session(model: MDDAPA, session_dir: Path, features: list[str],
                    feature_dims: dict[str, int], feature_stats_path: Path | None,
                    domain: str, target_role: str, partner_roles: list[str],
                    max_partners: int, window_len: int, device: torch.device,
                    ) -> dict[str, np.ndarray]:
    """Run sliding-window inference on one session, return full-length predictions."""

    from multimediate26.data.dataset import SessionDataset

    T_path = session_dir / "T.npy"
    T = int(np.load(T_path)) if T_path.exists() else None

    target_feats = {}
    for feat in features:
        p = session_dir / f"target_{feat}.npy"
        if p.exists():
            arr = np.load(p, mmap_mode="r")
            target_feats[feat] = arr
    if not target_feats:
        return {}

    if T is None:
        T = next(iter(target_feats.values())).shape[0]

    partner_feats_list = []
    partner_present = []
    for i in range(max_partners):
        pf = {}
        has_any = False
        for feat in features:
            p = session_dir / f"partner{i}_{feat}.npy"
            if p.exists():
                pf[feat] = np.load(p, mmap_mode="r")
                has_any = True
        if has_any:
            partner_feats_list.append(pf)
            partner_present.append(True)
        else:
            dummy = {feat: np.zeros((T, d), dtype=np.float32) for feat, d in feature_dims.items() if feat in target_feats}
            partner_feats_list.append(dummy)
            partner_present.append(False)

    if not any(partner_present):
        partner_feats_list = [{feat: np.zeros((T, d), dtype=np.float32) for feat, d in feature_dims.items() if feat in target_feats}]
        partner_present = [True]

    stats = None
    if feature_stats_path and feature_stats_path.exists():
        stats = dict(np.load(feature_stats_path))

    def normalize(feats_dict):
        out = {}
        for fname, arr in feats_dict.items():
            arr = arr.astype(np.float32)
            arr = np.nan_to_num(arr, nan=0.0)
            arr = np.clip(arr, -1e10, 1e10)
            if stats and f"{fname}_mean" in stats:
                mean = stats[f"{fname}_mean"]
                std = stats[f"{fname}_std"]
                std = np.where(std < 1e-8, 1.0, std)
                arr = (arr - mean) / std
                arr = np.clip(arr, -10, 10)
            out[fname] = arr
        return out

    target_feats = normalize(target_feats)
    partner_feats_list = [normalize(pf) for pf in partner_feats_list]

    stride = window_len // 2
    reg_acc = np.zeros(T, dtype=np.float64)
    reg_cnt = np.zeros(T, dtype=np.float64)
    task_acc = np.zeros((T, 4), dtype=np.float64)
    social_acc = np.zeros((T, 5), dtype=np.float64)
    cls_cnt = np.zeros(T, dtype=np.float64)
    hann = np.hanning(window_len)

    for start in range(0, T, stride):
        end = min(start + window_len, T)
        wl = end - start

        chunk_t = {k: torch.tensor(v[start:end]).unsqueeze(0).to(device) for k, v in target_feats.items()}
        chunk_ps = [
            {k: torch.tensor(v[start:end]).unsqueeze(0).to(device) for k, v in pf.items()}
            for pf in partner_feats_list
        ]
        mask = torch.ones(1, wl, dtype=torch.bool, device=device)

        fallback = DOMAIN_FALLBACKS.get(domain)
        batch = {
            "target_feats": chunk_t,
            "partner_feats": chunk_ps,
            "partner_present": partner_present,
            "attention_mask": mask,
            "domain": domain if domain != "noxi_add" else "noxi_add",
        }

        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(batch)

        w = hann[:wl] if wl == window_len else np.hanning(wl)

        pred_reg = out["reg"].squeeze(0).cpu().float().numpy()
        reg_acc[start:end] += pred_reg * w
        reg_cnt[start:end] += w

        if "task_logits" in out:
            task_probs = out["task_logits"].squeeze(0).cpu().float().numpy()
            social_probs = out["social_logits"].squeeze(0).cpu().float().numpy()
            task_acc[start:end] += task_probs * w[:, None]
            social_acc[start:end] += social_probs * w[:, None]
            cls_cnt[start:end] += w

    reg_pred = reg_acc / np.maximum(reg_cnt, 1e-8)
    sg_wl = min(25, T // 2 * 2 - 1)
    if sg_wl >= 5:
        reg_pred = savgol_filter(reg_pred, sg_wl, 3)
    reg_pred = np.clip(reg_pred, 0.0, 1.0)

    result = {"reg": reg_pred}

    if cls_cnt.sum() > 0:
        task_avg = task_acc / np.maximum(cls_cnt[:, None], 1e-8)
        social_avg = social_acc / np.maximum(cls_cnt[:, None], 1e-8)
        result["task_class"] = task_avg.argmax(axis=1)
        result["social_class"] = social_avg.argmax(axis=1)

    return result


def write_regression_csv(out_dir: Path, session_id: str, role: str,
                         predictions: np.ndarray) -> Path:
    sess_dir = out_dir / session_id
    sess_dir.mkdir(parents=True, exist_ok=True)
    out_path = sess_dir / f"{role}.engagement.prediction.csv"
    np.savetxt(out_path, predictions, fmt="%.6f")
    return out_path


def write_classification_csv(out_dir: Path, session_id: str, role: str,
                             task_classes: np.ndarray, social_classes: np.ndarray,
                             ) -> tuple[Path, Path]:
    sess_dir = out_dir / session_id
    sess_dir.mkdir(parents=True, exist_ok=True)

    task_labels = [TASK_INV_MAP.get(int(c), "noplay") for c in task_classes]
    social_labels = [SOCIAL_INV_MAP.get(int(c), "solitary") for c in social_classes]

    task_path = sess_dir / f"{role}.task_engagement.prediction.csv"
    social_path = sess_dir / f"{role}.social_engagement.prediction.csv"

    with open(task_path, "w") as f:
        f.write("\n".join(task_labels) + "\n")
    with open(social_path, "w") as f:
        f.write("\n".join(social_labels) + "\n")

    return task_path, social_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--base-config", type=Path, default=Path("multimediate26/configs/base.yaml"))
    ap.add_argument("--feature-specs", type=Path, default=Path("multimediate26/configs/feature_specs.yaml"))
    ap.add_argument("--features", type=str,
                    default="openface2,openface3,openpose,w2vbert2,egemapsv2,whisper,xlmr,videomae,dino,swin,clip")
    ap.add_argument("--feature-stats", type=Path, default=None)
    ap.add_argument("--npz-root", type=Path, default=Path("multimediate26/data_processed/npz_v4"))
    ap.add_argument("--window-len", type=int, default=512)
    ap.add_argument("--max-partners", type=int, default=3)
    ap.add_argument("--use-group-fusion", action="store_true", default=False)
    ap.add_argument("--use-flat-prompt", action="store_true", default=False)
    ap.add_argument("--use-sum-partner", action="store_true", default=False)
    ap.add_argument("--domains", type=str, default="noxi,noxi_add,noxi_j,mpiigi,pinsoro_cc,pinsoro_cr")
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    features = [f.strip() for f in args.features.split(",")]
    feat_cfg = yaml.safe_load(args.feature_specs.read_text())
    feature_dims = {f: feat_cfg["feature_dims"][f] for f in features}
    domains = [d.strip() for d in args.domains.split(",")]

    print(f"Loading model from {args.checkpoint}...")
    model = load_model(args.checkpoint, device, args.base_config, args.feature_specs,
                       features, args.use_group_fusion, args.use_flat_prompt, args.use_sum_partner)

    for domain in domains:
        manifest_path = DOMAIN_TEST_MANIFESTS.get(domain)
        if not manifest_path or not Path(manifest_path).exists():
            print(f"  Skipping {domain}: no test manifest")
            continue

        with open(manifest_path) as f:
            rows = [json.loads(l) for l in f if l.strip()]

        is_cls = domain.startswith("pinsoro")
        cr_purple_only = (domain == "pinsoro_cr")
        domain_out = args.out_dir / DOMAIN_TO_SUBMISSION_DIR.get(domain, domain)
        n_written = 0

        print(f"\n=== {domain}: {len(rows)} sessions ===")

        for row in rows:
            session_id = row["session_id"]
            target_role = row["target_role"]

            if cr_purple_only and target_role != "purple":
                continue

            sess_dir = Path(row["out_dir"])
            if not sess_dir.exists():
                print(f"  SKIP {session_id}/{target_role}: npz dir not found")
                continue

            preds = predict_session(
                model, sess_dir, features, feature_dims,
                args.feature_stats, domain, target_role,
                row.get("partner_roles", []),
                args.max_partners, args.window_len, device,
            )
            if not preds:
                print(f"  SKIP {session_id}/{target_role}: no features")
                continue

            if is_cls and "task_class" in preds:
                write_classification_csv(
                    domain_out, session_id, target_role,
                    preds["task_class"], preds["social_class"],
                )
            else:
                write_regression_csv(domain_out, session_id, target_role, preds["reg"])

            n_written += 1

        print(f"  {domain}: {n_written} prediction files written → {domain_out}")

    print(f"\nDone. Submission files in {args.out_dir}")


if __name__ == "__main__":
    main()
