"""Attention-based multi-partner pooling (v3 §4.5).

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

    def forward(self, partners: list[torch.Tensor],
                partner_mask: list[bool] | None = None) -> torch.Tensor:
        """partners: list of (B, T, D), one per partner slot.

        partner_mask: optional list of bool — True for real partner.
        """
        if not partners:
            raise ValueError("at least 1 partner expected")
        if partner_mask is None:
            partner_mask = [True] * len(partners)

        real = [p for p, m in zip(partners, partner_mask) if m]
        if not real:
            return torch.zeros_like(partners[0])
        if len(real) == 1:
            return real[0]

        if len(real) > self.max_partners:
            idx = torch.randperm(len(real))[:self.max_partners]
            real = [real[i] for i in idx]

        B, T, D = real[0].shape
        stacked = torch.stack(real, dim=2)          # (B, T, N, D)
        reshaped = stacked.reshape(B * T, len(real), D)
        query = self.query.expand(B * T, -1, -1)    # (B*T, 1, D)
        pooled, _ = self.attn(query, reshaped, reshaped)
        return pooled.squeeze(1).reshape(B, T, D)
