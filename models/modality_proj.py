"""ModalityProjector — per-feature linear projection to a shared hidden dim.

For each feature ``f`` of native dim ``D_f`` we learn a separate projection
``Linear(D_f → D)`` followed by LayerNorm + GELU + Dropout. The output is a
single tensor of shape ``(B, T, D)`` formed by *concatenating along the
channel axis* the projected modalities. This is the DAPA recipe ("each
modality projected then concatenated into a frame-level sequence X ∈
R^{N x D_in}", DAPA §3.3.1) — much simpler than v3-plan's 6-group fusion.

Two practical notes:

1. **z-score normalization happens in SessionDataset, not here.** Wide-scale
   features (egemapsv2 ±3e4, openface2 with a 19k frame-index column) would
   overflow bf16 here without dataset-side normalize + clip. That's a known
   land-mine; see memory/feature_scale_nan.md.

2. ``init_new_modality`` is for Phase 3 VLM injection — when adding the
   ``qwen3vl_emb`` projector after Phase 2 ckpt load, init its weights with
   small std so the new modality doesn't kick existing activations.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ModalityProjector(nn.Module):
    def __init__(self, in_dims: dict[str, int], hidden_dim: int,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.modalities: list[str] = list(in_dims.keys())
        self.projs = nn.ModuleDict()
        for m, d in in_dims.items():
            self.projs[m] = nn.Sequential(
                nn.Linear(int(d), self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )

    @torch.no_grad()
    def init_new_modality(self, name: str, small_std: float = 0.01) -> None:
        proj = self.projs[name]
        for m in proj.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=small_std)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, feats: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """feats: dict[name -> (B, T, D_f)] → dict[name -> (B, T, hidden_dim)]"""
        return {name: self.projs[name](feats[name])
                for name in self.modalities if name in feats}


class ModalityGroupFusion(nn.Module):
    """Two-level modality fusion (v3 §4.2).

    Intra-group: per-group TransformerEncoder fuses modalities within the same
    semantic group (e.g. the three audio features attend to each other).
    Inter-group: shared TransformerEncoder fuses group-level representations.
    """

    def __init__(self, groups: dict[str, list[str]], hidden_dim: int,
                 n_heads: int = 4, n_layers: int = 1,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.groups = dict(groups)
        self.hidden_dim = hidden_dim
        self.intra = nn.ModuleDict({
            g: nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=hidden_dim, nhead=n_heads,
                    dim_feedforward=hidden_dim * 2,
                    dropout=dropout, batch_first=True, norm_first=True,
                ),
                num_layers=n_layers,
            ) for g in groups
        })
        self.inter = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=n_heads,
                dim_feedforward=hidden_dim * 2,
                dropout=dropout, batch_first=True, norm_first=True,
            ),
            num_layers=1,
        )

    def forward(self, projected: dict[str, torch.Tensor]) -> torch.Tensor:
        """projected: dict[modality_name -> (B, T, D)] → (B, T, D)"""
        group_outs = []
        for group_name, modalities in self.groups.items():
            avail = [projected[m] for m in modalities if m in projected]
            if not avail:
                continue
            B, T, D = avail[0].shape
            if len(avail) == 1:
                stacked = avail[0].reshape(B * T, 1, D)
            else:
                stacked = torch.stack(avail, dim=2).reshape(B * T, len(avail), D)
            fused = self.intra[group_name](stacked).mean(dim=1).reshape(B, T, D)
            group_outs.append(fused)

        if not group_outs:
            raise ValueError("No modality groups had available features")

        B, T, D = group_outs[0].shape
        n_groups = len(group_outs)
        stacked = torch.stack(group_outs, dim=2).reshape(B * T, n_groups, D)
        out = self.inter(stacked).mean(dim=1).reshape(B, T, D)
        return out
