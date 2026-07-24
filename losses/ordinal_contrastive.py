"""Compatibility wrapper for the former ordinal-contrastive module name."""
from __future__ import annotations

import torch

from .ordinal_distance_matching import ordinal_distance_matching_loss


def ordinal_contrastive(
    features: torch.Tensor,
    continuous_labels: torch.Tensor,
    mask: torch.Tensor,
    max_pairs_per_batch: int = 512,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Backward-compatible alias for historical imports.

    ``max_pairs_per_batch`` historically named the sampled-frame cap. It is
    forwarded unchanged to the canonical implementation.
    """
    return ordinal_distance_matching_loss(
        features,
        continuous_labels,
        mask,
        max_frames_per_batch=max_pairs_per_batch,
        eps=eps,
    )
