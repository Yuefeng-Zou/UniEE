"""Ordinal distance-matching loss used by UniEE.

For normalized frame representations ``u_i`` and scalar engagement targets
``y_i``, the loss matches representation distance to ordinal label distance:

    d_f(i, j) = (2 - 2 * u_i^T u_j) / 4
    d_y(i, j) = clip(|y_i - y_j|, 0, 1)
    L_ord     = mean_{i != j} (d_f(i, j) - d_y(i, j))^2

This is a pairwise distance-regression objective. It does not use positive
pairs, negatives, a margin, or the NT-Xent softmax. PInSoRo contributes
pseudo-continuous targets derived from the fixed task and social class priors.
At most 512 valid frames are sampled so the O(N^2) pairwise cost stays bounded.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def ordinal_distance_matching_loss(
    features: torch.Tensor,
    continuous_labels: torch.Tensor,
    mask: torch.Tensor,
    max_frames_per_batch: int = 512,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute frame-level ordinal distance matching.

    Parameters
    ----------
    features : (B, T, D) hidden states from the model
    continuous_labels : (B, T) true or pseudo-continuous targets
    mask : (B, T) valid-label mask
    max_frames_per_batch : sampled-frame cap; 512 gives 261,632 ordered pairs
    """
    B, T, D = features.shape
    feat_flat = features.reshape(B * T, D)
    lbl_flat = continuous_labels.reshape(B * T)
    mask_flat = mask.reshape(B * T)
    valid_idx = mask_flat.nonzero(as_tuple=True)[0]
    n_valid = valid_idx.numel()
    if n_valid < 2:
        return torch.tensor(
            0.0, device=features.device, dtype=features.dtype,
        )

    if n_valid > max_frames_per_batch:
        perm = torch.randperm(
            n_valid, device=features.device,
        )[:max_frames_per_batch]
        valid_idx = valid_idx[perm]
        n_valid = max_frames_per_batch

    features_valid = F.normalize(
        feat_flat[valid_idx], dim=-1, eps=eps,
    )
    labels_valid = lbl_flat[valid_idx]

    label_dist = (
        labels_valid.unsqueeze(0) - labels_valid.unsqueeze(1)
    ).abs().clamp_(0, 1)

    cosine_similarity = features_valid @ features_valid.t()
    feature_sq_dist = (
        2.0 - 2.0 * cosine_similarity
    ).clamp_min(0)
    feature_dist = (feature_sq_dist / 4.0).clamp(0, 1)

    self_pairs = torch.eye(
        n_valid, dtype=torch.bool, device=features.device,
    )
    pairwise_error = (feature_dist - label_dist).square()
    pairwise_error = pairwise_error.masked_fill(self_pairs, 0.0)
    n_pairs = n_valid * (n_valid - 1)
    return pairwise_error.sum() / max(n_pairs, 1)
