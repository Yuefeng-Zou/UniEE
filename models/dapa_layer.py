"""DAPA Layer — Parallel Reactive-Anticipatory Cross-Attention (DAPA §3.4).

For each layer:

  1. Encode target and partner sequences each with a BiLSTM. The output
     concatenates forward + backward hidden states. We split them back into:
        Reactive states  H→  =  forward  (h_t depends on past+current only)
        Anticipatory states H← =  backward (h_t depends on future only)
     A subtle but critical detail in the DAPA paper: BiLSTM outputs are
     conceptually decomposed, NOT summed/concatenated as a single vector.
     The Reactive half captures in-the-moment reactions, the Anticipatory
     half captures the future-informed global perspective.

  2. Two parallel cross-attention pathways:
        Reactive Alignment:    A^R_{T←P}  =  Attn( H→T, H→P, H→P )
                               A^R_{P←T}  =  Attn( H→P, H→T, H→T )
        Anticipatory Alignment: A^A_{T←P} =  Attn( H←T, H←P, H←P )
                                A^A_{P←T} =  Attn( H←P, H←T, H←T )
     Both pathways run simultaneously for both target and partner — that's
     "parallel" (DAPA Fig 1). Earlier DAT-style approaches used a single
     unidirectional cross-attention; DAPA's ablation shows +0.015-0.038 CCC
     from this bidirectional split (Tab 1).

  3. Concatenate reactive + anticipatory alignment for each participant:
        H'_T = Concat[A^R_{T←P}, A^A_{T←P}]                 (DAPA Eq 7)
        H'_P = Concat[A^R_{P←T}, A^A_{P←T}]                 (DAPA Eq 8)
     Note this concat puts the participant back to ``hidden_dim`` width
     (each pathway emits hidden_dim/2 channels).

The whole layer is a residual block: input shape = output shape = (B, T, D),
so layers stack without dimension matching.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class DAPALayer(nn.Module):
    def __init__(self, hidden_dim: int, n_heads: int = 4,
                 dropout: float = 0.1) -> None:
        super().__init__()
        if hidden_dim % 2 != 0:
            raise ValueError(f"hidden_dim must be even (got {hidden_dim}) — "
                             "BiLSTM splits it into halves.")
        self.hidden_dim = hidden_dim
        half = hidden_dim // 2

        # BiLSTM with hidden=half/direction => output = hidden_dim
        self.bilstm_target = nn.LSTM(
            input_size=hidden_dim, hidden_size=half,
            batch_first=True, bidirectional=True,
        )
        self.bilstm_partner = nn.LSTM(
            input_size=hidden_dim, hidden_size=half,
            batch_first=True, bidirectional=True,
        )

        # Cross-attention. Each pathway operates in half-dim space.
        # n_heads must divide half — caller's responsibility.
        if half % n_heads != 0:
            raise ValueError(
                f"n_heads={n_heads} must divide half-hidden={half} "
                f"(hidden_dim={hidden_dim})."
            )
        self.react_TfromP = nn.MultiheadAttention(half, n_heads, dropout=dropout,
                                                  batch_first=True)
        self.react_PfromT = nn.MultiheadAttention(half, n_heads, dropout=dropout,
                                                  batch_first=True)
        self.antic_TfromP = nn.MultiheadAttention(half, n_heads, dropout=dropout,
                                                  batch_first=True)
        self.antic_PfromT = nn.MultiheadAttention(half, n_heads, dropout=dropout,
                                                  batch_first=True)

        # Post-attention layernorms + residual to encoder output. The DAPA
        # paper figure shows the parallel attentions feed directly into the
        # next layer; we add LN + residual here for training stability with
        # bf16 (paper used fp32 throughout).
        self.norm_target = nn.LayerNorm(hidden_dim)
        self.norm_partner = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _split_directions(bilstm_out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(B, T, hidden_dim) → ( forward, backward ), each (B, T, hidden_dim/2 )."""
        D = bilstm_out.shape[-1]
        return bilstm_out[..., : D // 2], bilstm_out[..., D // 2 :]

    def forward(self, target: torch.Tensor, partner: torch.Tensor,
                attn_mask: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """target, partner: (B, T, hidden_dim).

        attn_mask: optional (B, T) bool, True for valid frames. Used as
        key_padding_mask on all 4 cross-attentions; same mask applies to
        both participants because partner shares the same valid window.
        """
        # 1. BiLSTM encoders.
        t_out, _ = self.bilstm_target(target)            # (B, T, D)
        p_out, _ = self.bilstm_partner(partner)          # (B, T, D)
        t_react, t_antic = self._split_directions(t_out)
        p_react, p_antic = self._split_directions(p_out)

        # Pre-build key_padding_mask. MultiheadAttention expects True == PAD.
        kpm = None if attn_mask is None else ~attn_mask  # invert

        # 2. Four parallel cross-attentions.
        rT, _ = self.react_TfromP(t_react, p_react, p_react, key_padding_mask=kpm)
        rP, _ = self.react_PfromT(p_react, t_react, t_react, key_padding_mask=kpm)
        aT, _ = self.antic_TfromP(t_antic, p_antic, p_antic, key_padding_mask=kpm)
        aP, _ = self.antic_PfromT(p_antic, t_antic, t_antic, key_padding_mask=kpm)

        # 3. Concat reactive + anticipatory → hidden_dim wide again.
        new_t = torch.cat([rT, aT], dim=-1)
        new_p = torch.cat([rP, aP], dim=-1)

        # Residual + LN. Original input is the participant's feature, BiLSTM
        # is in the path so residual goes to pre-LSTM input.
        target_out = self.norm_target(target + self.dropout(new_t))
        partner_out = self.norm_partner(partner + self.dropout(new_p))
        return target_out, partner_out
