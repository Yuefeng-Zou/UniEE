"""Continuous and categorical engagement prediction heads.

Engagement = regression on NoXi/NoXi-J/NoXi-add/MPIGI (continuous 0-1 with
CCC metric) AND classification on PInSoRo cc/cr (task_engagement: 4
classes, social_engagement: 5 classes, Cohen's kappa metric).

The regression head is an MLP followed by a sigmoid. The two categorical
heads predict PInSoRo task and social engagement. In Phase 3, the bridge
maps their probabilities to a pseudo-continuous scalar in [0, 1].
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class RegressionHead(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)              # (B, T)


class ClassificationHeads(nn.Module):
    """Two linear heads for PInSoRo: task (4 classes) + social (5 classes)."""
    def __init__(self, hidden_dim: int,
                 n_task_classes: int = 4,
                 n_social_classes: int = 5) -> None:
        super().__init__()
        self.task = nn.Linear(hidden_dim, n_task_classes)
        self.social = nn.Linear(hidden_dim, n_social_classes)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "task_logits":   self.task(x),          # (B, T, 4)
            "social_logits": self.social(x),        # (B, T, 5)
        }


class LearnableBridge(nn.Module):
    """Maps PInSoRo (task_softmax, social_softmax) → pseudo-continuous [0,1].

    In Phase 3 it gives PInSoRo batches a pseudo-continuous supervision path
    (the ``bridge_ccc`` loss term against a
    fixed per-class prior). Initialized so MLP output ≈ ``target_mean``
    out of the gate, preventing it from yanking the regression head off
    during the first few steps.
    """
    def __init__(self, n_task_classes: int = 4,
                 n_social_classes: int = 5,
                 hidden: int = 32,
                 target_mean: float = 0.5) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(n_task_classes + n_social_classes, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )
        with torch.no_grad():
            for m in self.mlp.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=0.01)
            # Bias the pre-sigmoid output so sigmoid(bias) == target_mean.
            bias_logit = math.log(target_mean / (1.0 - target_mean + 1e-8))
            self.mlp[-2].bias.fill_(bias_logit)

    def forward(self, task_logits: torch.Tensor,
                social_logits: torch.Tensor) -> torch.Tensor:
        task_p = task_logits.softmax(dim=-1)
        social_p = social_logits.softmax(dim=-1)
        x = torch.cat([task_p, social_p], dim=-1)
        return self.mlp(x).squeeze(-1)
