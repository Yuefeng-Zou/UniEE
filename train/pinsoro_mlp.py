"""PInSoRo per-frame MLP classifier — mirrors official baseline approach.

Per-modality independent MLPs, per-frame prediction, MinMaxScaler.
CC and CR trained separately with class-weighted Focal Loss.

Usage:
    python -m multimediate26.train.pinsoro_mlp \
        --domain pinsoro_cc \
        --output-dir multimediate26/output/pinsoro_mlp_cc_seed0
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import cohen_kappa_score
from sklearn.preprocessing import MinMaxScaler


TASK_MAP = {"goaloriented": 0, "aimless": 1, "adultseeking": 2, "noplay": 3}
SOCIAL_MAP = {"solitary": 0, "onlooker": 1, "parallel": 2, "associative": 3, "cooperative": 4}
TASK_INV = {v: k for k, v in TASK_MAP.items()}
SOCIAL_INV = {v: k for k, v in SOCIAL_MAP.items()}


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce)
        return (((1 - pt) ** self.gamma) * ce).mean()


class FrameMLP(nn.Module):
    def __init__(self, in_dim, n_classes, hidden=256, dropout=0.35):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def load_frames(manifest_path, npz_root, features, feature_dims):
    """Load all frames as (X, task_labels, social_labels, mask)."""
    X_all, task_all, social_all = [], [], []
    with open(manifest_path) as f:
        rows = [json.loads(l) for l in f if l.strip()]

    for row in rows:
        sess_dir = Path(row["out_dir"])
        feats = []
        T = None
        for feat in features:
            p = sess_dir / f"target_{feat}.npy"
            if p.exists():
                arr = np.load(p).astype(np.float32)
                arr = np.nan_to_num(arr, nan=0.0)
                arr = np.clip(arr, -1e10, 1e10)
                feats.append(arr)
                if T is None:
                    T = arr.shape[0]
            else:
                feats.append(np.zeros((T or 1, feature_dims[feat]), dtype=np.float32))

        if T is None:
            continue

        X = np.concatenate(feats, axis=1)

        task_path = sess_dir / "label_task.npy"
        social_path = sess_dir / "label_social.npy"
        mask_path = sess_dir / "label_mask.npy"

        if task_path.exists() and social_path.exists() and mask_path.exists():
            task = np.load(task_path).astype(np.int64)
            social = np.load(social_path).astype(np.int64)
            mask = np.load(mask_path).astype(bool)

            valid = mask[:len(X)]
            X_valid = X[:len(mask)][valid]
            task_valid = task[valid]
            social_valid = social[valid]

            X_all.append(X_valid)
            task_all.append(task_valid)
            social_all.append(social_valid)

    return np.concatenate(X_all), np.concatenate(task_all), np.concatenate(social_all)


def load_test_frames(manifest_path, npz_root, features, feature_dims):
    """Load test frames per session (no labels)."""
    sessions = []
    with open(manifest_path) as f:
        rows = [json.loads(l) for l in f if l.strip()]

    for row in rows:
        sess_dir = Path(row["out_dir"])
        feats = []
        T = None
        for feat in features:
            p = sess_dir / f"target_{feat}.npy"
            if p.exists():
                arr = np.load(p).astype(np.float32)
                arr = np.nan_to_num(arr, nan=0.0)
                arr = np.clip(arr, -1e10, 1e10)
                feats.append(arr)
                if T is None:
                    T = arr.shape[0]
            else:
                feats.append(np.zeros((T or 1, feature_dims[feat]), dtype=np.float32))

        if T is None:
            continue
        X = np.concatenate(feats, axis=1)
        sessions.append({
            "session_id": row["session_id"],
            "target_role": row["target_role"],
            "X": X,
            "T": T,
        })
    return sessions


def compute_class_weights(labels, n_classes):
    counts = np.bincount(labels, minlength=n_classes).astype(np.float32)
    counts = np.maximum(counts, 1)
    total = counts.sum()
    return torch.tensor(total / (n_classes * counts))


def train_head(X_train, y_train, X_val, y_val, n_classes, class_weights,
               head_name, epochs=100, lr=1e-3, batch_size=4096, device="cuda"):
    in_dim = X_train.shape[1]
    model = FrameMLP(in_dim, n_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = FocalLoss(gamma=2.0, weight=class_weights.to(device))

    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.long)
    X_v = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_v = torch.tensor(y_val, dtype=torch.long).to(device)

    best_kappa = -1.0
    best_state = None
    n = len(X_t)

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        losses = []
        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            xb = X_t[idx].to(device)
            yb = y_t[idx].to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        scheduler.step()

        model.eval()
        with torch.no_grad():
            pred = model(X_v).argmax(dim=1).cpu().numpy()
            gt = y_v.cpu().numpy()
            kappa = cohen_kappa_score(gt, pred)

        if kappa > best_kappa:
            best_kappa = kappa
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (ep + 1) % 20 == 0:
            print(f"  {head_name} ep{ep+1}: loss={np.mean(losses):.4f} val_kappa={kappa:.4f} best={best_kappa:.4f}")

    model.load_state_dict(best_state)
    return model, best_kappa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", type=str, required=True, choices=["pinsoro_cc", "pinsoro_cr"])
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--features", type=str,
                    default="openface2,openface3,openpose,w2vbert2,egemapsv2,xlmr,videomae,dino,swin,clip")
    ap.add_argument("--feature-specs", type=Path, default=Path("multimediate26/configs/feature_specs.yaml"))
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()

    import yaml
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    feat_cfg = yaml.safe_load(args.feature_specs.read_text())
    features = [f.strip() for f in args.features.split(",")]
    feature_dims = {f: feat_cfg["feature_dims"][f] for f in features}

    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_manifest = f"multimediate26/manifests/{args.domain}_train.jsonl"
    val_manifest = f"multimediate26/manifests/{args.domain}_val.jsonl"

    print(f"Loading {args.domain} train data...")
    X_train, task_train, social_train = load_frames(train_manifest, None, features, feature_dims)
    print(f"  train: {X_train.shape[0]} frames, {X_train.shape[1]} dims")

    print(f"Loading {args.domain} val data...")
    X_val, task_val, social_val = load_frames(val_manifest, None, features, feature_dims)
    print(f"  val: {X_val.shape[0]} frames")

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    n_task = 4
    n_social = 5
    task_w = compute_class_weights(task_train, n_task)
    social_w = compute_class_weights(social_train, n_social)
    print(f"  task weights: {task_w.tolist()}")
    print(f"  social weights: {social_w.tolist()}")

    print(f"\nTraining task head...")
    task_model, task_kappa = train_head(
        X_train, task_train, X_val, task_val, n_task, task_w,
        "task", args.epochs, args.lr, device=device,
    )

    print(f"\nTraining social head...")
    social_model, social_kappa = train_head(
        X_train, social_train, X_val, social_val, n_social, social_w,
        "social", args.epochs, args.lr, device=device,
    )

    print(f"\n=== {args.domain} results ===")
    print(f"  task_kappa={task_kappa:.4f}, social_kappa={social_kappa:.4f}")
    print(f"  combined_kappa={(task_kappa + social_kappa) / 2:.4f}")

    torch.save({
        "task_model": task_model.state_dict(),
        "social_model": social_model.state_dict(),
        "scaler_mean": scaler.data_min_,
        "scaler_scale": scaler.scale_,
        "task_kappa": task_kappa,
        "social_kappa": social_kappa,
        "features": features,
        "feature_dims": feature_dims,
        "domain": args.domain,
    }, args.output_dir / "best.pt")

    # Inference on test
    test_manifest = f"multimediate26/manifests/{args.domain.replace('pinsoro_', 'pinsoro_')}_test.jsonl"
    if Path(test_manifest).exists():
        print(f"\nRunning test inference...")
        test_sessions = load_test_frames(test_manifest, None, features, feature_dims)
        domain_sub = "pinsoro-cc" if args.domain == "pinsoro_cc" else "pinsoro-cr"
        is_cr = args.domain == "pinsoro_cr"

        for sess in test_sessions:
            if is_cr and sess["target_role"] != "purple":
                continue
            X_test = scaler.transform(sess["X"])
            X_t = torch.tensor(X_test, dtype=torch.float32).to(device)

            task_model.eval()
            social_model.eval()
            with torch.no_grad():
                task_pred = task_model(X_t).argmax(dim=1).cpu().numpy()
                social_pred = social_model(X_t).argmax(dim=1).cpu().numpy()

            out_dir = args.output_dir / "predictions" / domain_sub / sess["session_id"]
            out_dir.mkdir(parents=True, exist_ok=True)

            task_labels = [TASK_INV.get(int(c), "noplay") for c in task_pred]
            social_labels = [SOCIAL_INV.get(int(c), "solitary") for c in social_pred]

            with open(out_dir / f"{sess['target_role']}.task_engagement.prediction.csv", "w") as f:
                f.write("\n".join(task_labels) + "\n")
            with open(out_dir / f"{sess['target_role']}.social_engagement.prediction.csv", "w") as f:
                f.write("\n".join(social_labels) + "\n")

        n_files = sum(1 for _ in (args.output_dir / "predictions").rglob("*.csv"))
        print(f"  {n_files} prediction files written")

    print("\nDone.")


if __name__ == "__main__":
    main()
