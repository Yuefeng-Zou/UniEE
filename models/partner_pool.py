"""Attention-based multi-partner pooling.

For single-partner sessions (NoXi, NoXi-J) this is a pass-through.
For multi-partner sessions (MPIIGI 3-4 person) a learnable query attends
over the partner representations to produce a weighted summary.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class MultiPartnerPooling(nn.Module):
    def __init__(self, hidden_dim: int, n_heads: int = 4,
                 max_partners: int = 4) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_partners = max_partners
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.attn = nn.MultiheadAttention(
            hidden_dim, n_heads, batch_first=True,
        )

    def forward(
        self,
        partners: list[torch.Tensor],
        partner_mask: list[bool] | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """partners: list of (B, T, D), one per partner slot.

        partner_mask: optional (B, N) mask or a legacy list of slot booleans.
        """
        if not partners:
            raise ValueError("at least 1 partner expected")
        partners = partners[:self.max_partners]
        B, T, D = partners[0].shape
        n_partners = len(partners)

        if partner_mask is None:
            mask = torch.ones(
                B, n_partners, dtype=torch.bool, device=partners[0].device,
            )
        elif isinstance(partner_mask, torch.Tensor):
            mask = partner_mask[:, :n_partners].to(
                device=partners[0].device, dtype=torch.bool,
            )
        else:
            mask = torch.tensor(
                partner_mask[:n_partners],
                dtype=torch.bool,
                device=partners[0].device,
            ).unsqueeze(0).expand(B, -1)

        if mask.shape != (B, n_partners):
            raise ValueError(
                f"partner_mask must have shape {(B, n_partners)}, got {tuple(mask.shape)}"
            )
        if n_partners == 1:
            return partners[0] * mask[:, :1, None].to(partners[0].dtype)

        stacked = torch.stack(partners, dim=2)      # (B, T, N, D)
        reshaped = stacked.reshape(B * T, n_partners, D)
        query = self.query.expand(B * T, -1, -1)    # (B*T, 1, D)

        # MultiheadAttention cannot accept rows where every key is masked.
        # Temporarily expose a zero-filled slot, then zero the pooled result.
        has_partner = mask.any(dim=1)
        safe_mask = mask.clone()
        safe_mask[~has_partner, 0] = True
        frame_mask = safe_mask[:, None, :].expand(B, T, n_partners)
        key_padding_mask = ~frame_mask.reshape(B * T, n_partners)

        pooled, _ = self.attn(
            query,
            reshaped,
            reshaped,
            key_padding_mask=key_padding_mask,
        )
        pooled = pooled.squeeze(1).reshape(B, T, D)
        return pooled * has_partner[:, None, None].to(pooled.dtype)
