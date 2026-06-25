"""Ordinal contrastive loss (TechPlan v3 §5).

Encourages frames with similar continuous engagement scores to have similar
hidden features, and frames with very different scores to be pushed apart.
Acts as a soft regularizer over the dyadic representation so the
continuous-task and classification-task domains live in a comparable space.

Only frames with valid labels participate. For PInSoRo (no continuous label
directly) we use the pseudo-continuous target built by build_session_npz
from the class-prior table.

Loss form (margin-based, simpler than NT-Xent):
    For an in-batch pair (i, j) with valid labels y_i, y_j and features f_i, f_j:
        target_dist  = |y_i - y_j|            ∈ [0, 1]
        actual_dist  = ||normalize(f_i) - normalize(f_j)||  ∈ [0, 2]
    Loss = mean over all valid pairs of (actual_dist/2 - target_dist)^2

  * If labels are close, actual_dist should be 0 — pulls features together
  * If labels are far  (≈1), actual_dist should be 2 — pushes apart
  * The /2 maps [0, 2] norm into [0, 1] to compare with label diff directly

We subsample N=512 valid frames per batch to keep the pairwise cost bounded
(O(N^2)). Per the plan, weight is 0.1.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def ordinal_contrastive(features: torch.Tensor,
                        continuous_labels: torch.Tensor,
                        mask: torch.Tensor,
                        max_pairs_per_batch: int = 512,
                        eps: float = 1e-6) -> torch.Tensor:
    """Frame-level ordinal contrastive loss.

    Parameters
    ----------
    features : (B, T, D)  hidden states from the model
    continuous_labels : (B, T) float — true CCC label OR pseudo-cont for PInSoRo
    mask : (B, T) bool — valid label mask
    max_pairs_per_batch : cap on sampled frames (gives ~N^2/2 pairs); 512 → 130k
    """
    B, T, D = features.shape
    # Flatten and select only valid frames
    feat_flat = features.reshape(B * T, D)
    lbl_flat  = continuous_labels.reshape(B * T)
    mask_flat = mask.reshape(B * T)
    valid_idx = mask_flat.nonzero(as_tuple=True)[0]
    n_valid = valid_idx.numel()
    if n_valid < 2:
        return torch.tensor(0.0, device=features.device, dtype=features.dtype)
    # Sub-sample
    if n_valid > max_pairs_per_batch:
        perm = torch.randperm(n_valid, device=features.device)[:max_pairs_per_batch]
        valid_idx = valid_idx[perm]
        n_valid = max_pairs_per_batch
    f = feat_flat[valid_idx]                                  # (N, D)
    y = lbl_flat[valid_idx]                                   # (N,)
    f = F.normalize(f, dim=-1, eps=eps)                       # unit-norm

    # Pairwise distances
    label_dist = (y.unsqueeze(0) - y.unsqueeze(1)).abs().clamp_(0, 1)   # (N, N) in [0,1]
    # Use SQUARED distance instead of sqrt — sqrt(x) at x→0 has gradient
    # 1/(2√x) which explodes for near-identical unit vectors (very common
    # within a batch) and produces NaN gradients under bf16 autocast.
    # squared_dist = ||a - b||² = 2 - 2 a·b for unit vectors, ∈ [0, 4]
    # Normalize to [0, 1] and compare to label_dist directly.
    cos_sim = f @ f.t()                                       # (N, N) in [-1, 1]
    feat_sq_dist = (2.0 - 2.0 * cos_sim).clamp_min(0)         # ∈ [0, 4]
    feat_dist_norm = (feat_sq_dist / 4.0).clamp(0, 1)         # ∈ [0, 1]

    # Drop self-pairs
    diag = torch.eye(n_valid, dtype=torch.bool, device=features.device)
    diff = (feat_dist_norm - label_dist) ** 2
    diff = diff.masked_fill(diag, 0.0)
    n_pairs = n_valid * (n_valid - 1)
    return diff.sum() / max(n_pairs, 1)
