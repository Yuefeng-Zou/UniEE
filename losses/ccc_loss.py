"""CCC loss + smoothness regularizer (DAPA Eq 11 / Lin 1989).

DAPA paper uses pure ``1 - CCC`` as the only loss. We add an optional MSE
auxiliary (anchors absolute level — useful early in training when CCC
gradient is unstable) and a smoothness penalty (suppresses frame-to-frame
jitter that hurts CCC on noisy sessions). Both are off by default; the
trainer's loss_weights config controls them.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def ccc(pred: torch.Tensor, target: torch.Tensor,
        mask: torch.Tensor | None = None,
        eps: float = 1e-8) -> torch.Tensor:
    """Concordance Correlation Coefficient as a scalar tensor.

    pred, target: (B, T) or (N,). mask: same shape, bool. Masked-out values
    are excluded from the population statistics.

    Returns CCC in (-1, 1]. The caller computes ``1 - ccc`` as the loss.
    """
    if mask is not None:
        pred = pred[mask]
        target = target[mask]
    if pred.numel() < 2:
        return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
    p_mean, t_mean = pred.mean(), target.mean()
    p_var = pred.var(unbiased=False)
    t_var = target.var(unbiased=False)
    cov = ((pred - p_mean) * (target - t_mean)).mean()
    return 2 * cov / (p_var + t_var + (p_mean - t_mean) ** 2 + eps)


def ccc_loss(pred: torch.Tensor, target: torch.Tensor,
             mask: torch.Tensor | None = None) -> torch.Tensor:
    """Loss = 1 - CCC. Bounded in [0, 2]."""
    return 1.0 - ccc(pred, target, mask)


def mse_loss_masked(pred: torch.Tensor, target: torch.Tensor,
                    mask: torch.Tensor | None = None) -> torch.Tensor:
    if mask is None:
        return F.mse_loss(pred, target)
    if mask.sum() == 0:
        return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
    return F.mse_loss(pred[mask], target[mask])


def smoothness_loss(pred: torch.Tensor,
                    mask: torch.Tensor | None = None) -> torch.Tensor:
    """L1 of first temporal difference. pred: (B, T)."""
    diff = (pred[:, 1:] - pred[:, :-1]).abs()
    if mask is None:
        return diff.mean()
    # Only count diffs where both frames are valid.
    valid = mask[:, 1:] & mask[:, :-1]
    if valid.sum() == 0:
        return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
    return diff[valid].mean()
