"""Parameter-free multi-partner sum (ablation baseline).

Simple sum aggregation. Used for ablation comparison against MultiPartnerPooling.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class MultiPartnerSum(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

    def forward(self, partners: list[torch.Tensor],
                partner_mask: list[bool] | None = None) -> torch.Tensor:
        if not partners:
            raise ValueError("at least 1 partner expected")
        if partner_mask is None:
            partner_mask = [True] * len(partners)
        real = [p for p, m in zip(partners, partner_mask) if m]
        if not real:
            return torch.zeros_like(partners[0])
        return torch.stack(real, dim=0).sum(dim=0)
