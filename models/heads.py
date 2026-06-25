"""Dual-task prediction heads.

Engagement = regression on NoXi/NoXi-J/NoXi-add/MPIGI (continuous 0-1 with
CCC metric) AND classification on PInSoRo cc/cr (task_engagement: 4
classes, social_engagement: 5 classes, Cohen's kappa metric).

DAPA paper uses MLP + Sigmoid for regression. We follow that — KAN was
considered (USTC-IAT'25 used it for +0.025 CCC) but the user opted out for
this iteration to simplify the parameter count and training.

The bridge is OFF by default — turn it on in Phase 2 once both heads are
trained; it lets PInSoRo's categorical supervision contribute a soft
target to the regression head via a learnable mapping from class
probabilities to a [0, 1] engagement scalar. Initialized so the initial
bridge prediction = the training-data mean engagement (NoXi ≈ 0.5).
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

    For Phase 2 we use it on PInSoRo batches to give the regression head a
    second source of supervision (``bridge_ccc`` loss term against a
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
