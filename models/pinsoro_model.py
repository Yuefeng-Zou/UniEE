"""PInSoRo specialist model — lightweight BiLSTM + Cross-Attention for classification.

Designed for small PInSoRo dataset (64 sessions):
  - Light BiLSTM encoder (2 layers, shared for target/partner)
  - 1-layer cross-attention for interaction modeling
  - 4 independent classification heads (CC/CR × task/social)
  - Focal Loss + inverse-freq class weights
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PInSoRoConfig:
    feature_dims: dict[str, int]
    hidden_dim: int = 256
    lstm_layers: int = 2
    n_heads: int = 4
    dropout: float = 0.3
    cc_task_classes: int = 4
    cc_social_classes: int = 5
    cr_task_classes: int = 4
    cr_social_classes: int = 5


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None,
                 label_smoothing: float = 0.05, ignore_index: int = -100):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.label_smoothing = label_smoothing
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, weight=self.weight,
                             label_smoothing=self.label_smoothing,
                             ignore_index=self.ignore_index, reduction='none')
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce
        return focal.mean()


class PInSoRoModel(nn.Module):
    def __init__(self, cfg: PInSoRoConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.projs = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(dim, cfg.hidden_dim),
                nn.LayerNorm(cfg.hidden_dim),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
            )
            for name, dim in cfg.feature_dims.items()
        })

        K = len(cfg.feature_dims)
        self.down_proj = nn.Sequential(
            nn.Linear(K * cfg.hidden_dim, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )

        self.bilstm = nn.LSTM(
            input_size=cfg.hidden_dim,
            hidden_size=cfg.hidden_dim // 2,
            num_layers=cfg.lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=cfg.dropout if cfg.lstm_layers > 1 else 0,
        )

        self.cross_attn_t2p = nn.MultiheadAttention(
            cfg.hidden_dim, cfg.n_heads, dropout=cfg.dropout, batch_first=True,
        )
        self.cross_attn_p2t = nn.MultiheadAttention(
            cfg.hidden_dim, cfg.n_heads, dropout=cfg.dropout, batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(cfg.hidden_dim)
        self.cross_drop = nn.Dropout(cfg.dropout)

        head_in = cfg.hidden_dim * 2

        self.cc_task_head = nn.Sequential(
            nn.Linear(head_in, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.cc_task_classes),
        )
        self.cc_social_head = nn.Sequential(
            nn.Linear(head_in, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.cc_social_classes),
        )
        self.cr_task_head = nn.Sequential(
            nn.Linear(head_in, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.cr_task_classes),
        )
        self.cr_social_head = nn.Sequential(
            nn.Linear(head_in, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.cr_social_classes),
        )

        self._modality_order = sorted(cfg.feature_dims.keys())

    def _project(self, feats: dict[str, torch.Tensor]) -> torch.Tensor:
        parts = []
        ref = next(iter(feats.values()))
        B, T, _ = ref.shape
        for name in self._modality_order:
            if name in feats:
                parts.append(self.projs[name](feats[name]))
            else:
                parts.append(torch.zeros(B, T, self.cfg.hidden_dim,
                                         device=ref.device, dtype=ref.dtype))
        return self.down_proj(torch.cat(parts, dim=-1))

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.bilstm(x)
        return out

    def forward(self, batch: dict) -> dict[str, torch.Tensor]:
        target_x = self._project(batch["target_feats"])
        partners = batch["partner_feats"]
        partner_present = batch.get("partner_present", [True] * len(partners))

        real = [self._project(pf) for pf, m in zip(partners, partner_present) if m]
        if real:
            partner_x = torch.stack(real, dim=0).sum(dim=0)
        else:
            partner_x = torch.zeros_like(target_x)

        h_t = self._encode(target_x)
        h_p = self._encode(partner_x)

        a_t, _ = self.cross_attn_t2p(h_t, h_p, h_p)
        h_t = self.cross_norm(h_t + self.cross_drop(a_t))

        a_p, _ = self.cross_attn_p2t(h_p, h_t, h_t)
        h_p = self.cross_norm(h_p + self.cross_drop(a_p))

        dyad = torch.cat([h_t, h_p], dim=-1)

        domain = batch["domain"]
        out: dict[str, torch.Tensor] = {"features": dyad}

        if domain == "pinsoro_cc":
            out["task_logits"] = self.cc_task_head(dyad)
            out["social_logits"] = self.cc_social_head(dyad)
        elif domain == "pinsoro_cr":
            out["task_logits"] = self.cr_task_head(dyad)
            out["social_logits"] = self.cr_social_head(dyad)
        else:
            out["task_logits"] = self.cc_task_head(dyad)
            out["social_logits"] = self.cc_social_head(dyad)

        return out
