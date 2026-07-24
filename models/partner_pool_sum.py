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

    def forward(
        self,
        partners: list[torch.Tensor],
        partner_mask: list[bool] | torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not partners:
            raise ValueError("at least 1 partner expected")
        B = partners[0].shape[0]
        if partner_mask is None:
            mask = torch.ones(
                B, len(partners), dtype=torch.bool, device=partners[0].device,
            )
        elif isinstance(partner_mask, torch.Tensor):
            mask = partner_mask.to(partners[0].device, dtype=torch.bool)
        else:
            mask = torch.tensor(
                partner_mask, dtype=torch.bool, device=partners[0].device,
            ).unsqueeze(0).expand(B, -1)
        stacked = torch.stack(partners, dim=2)
        return (
            stacked
            * mask[:, None, :, None].to(stacked.dtype)
        ).sum(dim=2)
