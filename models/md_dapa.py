"""MD-DAPA v2 — Multi-Domain Domain-Adaptive Parallel Attention.

v2 architecture upgrades over v1:
  - ModalityGroupFusion replaces concat+Linear down-projection
  - HierarchicalDomainPrompt replaces flat DomainPromptPool
  - MultiPartnerPooling replaces parameter-free sum
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from .modality_proj import ModalityProjector, ModalityGroupFusion
from .domain_prompt import HierarchicalDomainPrompt
from .dapa_layer import DAPALayer
from .partner_pool import MultiPartnerPooling
from .heads import RegressionHead, ClassificationHeads, LearnableBridge


@dataclass
class MDDAPAConfig:
    feature_dims: dict[str, int]
    hidden_dim: int = 384
    n_dapa_layers: int = 3
    n_heads: int = 4
    dropout: float = 0.15
    n_prompts_coarse: int = 4
    n_prompts_fine: int = 8
    max_partners: int = 4
    domains: list[str] | None = None
    groups: dict[str, list[str]] | None = None
    fusion_layers: int = 1
    enable_classification: bool = True
    enable_bridge: bool = True
    n_task_classes: int = 4
    n_social_classes: int = 5
    target_mean: float = 0.5
    use_flat_prompt: bool = False
    use_sum_partner: bool = False


class MDDAPA(nn.Module):
    def __init__(self, cfg: MDDAPAConfig) -> None:
        super().__init__()
        self.cfg = cfg
        if cfg.domains is None:
            raise ValueError("MDDAPAConfig.domains must be provided.")

        self.projector = ModalityProjector(
            in_dims=cfg.feature_dims, hidden_dim=cfg.hidden_dim,
            dropout=cfg.dropout,
        )

        if cfg.groups is not None:
            active_groups = {
                g: [m for m in mods if m in cfg.feature_dims]
                for g, mods in cfg.groups.items()
            }
            active_groups = {g: mods for g, mods in active_groups.items() if mods}
            self.group_fusion = ModalityGroupFusion(
                groups=active_groups,
                hidden_dim=cfg.hidden_dim,
                n_heads=cfg.n_heads,
                n_layers=cfg.fusion_layers,
                dropout=cfg.dropout,
            )
            self._use_group_fusion = True
        else:
            K = len(cfg.feature_dims)
            self.down_proj = nn.Sequential(
                nn.Linear(K * cfg.hidden_dim, cfg.hidden_dim),
                nn.LayerNorm(cfg.hidden_dim),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
            )
            self._use_group_fusion = False

        fine_domains = [d for d in sorted(set(cfg.domains))
                        if d not in HierarchicalDomainPrompt.DEFAULT_FALLBACKS]
        if cfg.use_flat_prompt:
            from .domain_prompt_flat import DomainPromptPool
            self.domain_prompt = DomainPromptPool(
                domains=sorted(set(cfg.domains)),
                hidden_dim=cfg.hidden_dim,
                prompt_len=cfg.n_prompts_fine,
            )
            self._flat_prompt = True
        else:
            self.domain_prompt = HierarchicalDomainPrompt(
                hidden_dim=cfg.hidden_dim,
                n_prompts_coarse=cfg.n_prompts_coarse,
                n_prompts_fine=cfg.n_prompts_fine,
                fine_domains=fine_domains,
            )
            self._flat_prompt = False

        if cfg.use_sum_partner:
            from .partner_pool_sum import MultiPartnerSum
            self.partner_pool = MultiPartnerSum(hidden_dim=cfg.hidden_dim)
        else:
            self.partner_pool = MultiPartnerPooling(
                hidden_dim=cfg.hidden_dim,
                n_heads=cfg.n_heads,
                max_partners=cfg.max_partners,
            )

        self.dapa_layers = nn.ModuleList([
            DAPALayer(hidden_dim=cfg.hidden_dim, n_heads=cfg.n_heads,
                      dropout=cfg.dropout)
            for _ in range(cfg.n_dapa_layers)
        ])

        head_in = 2 * cfg.hidden_dim
        self.reg_head = RegressionHead(head_in, dropout=cfg.dropout)
        if cfg.enable_classification:
            self.cls_heads = ClassificationHeads(
                head_in,
                n_task_classes=cfg.n_task_classes,
                n_social_classes=cfg.n_social_classes,
            )
        if cfg.enable_classification and cfg.enable_bridge:
            self.bridge = LearnableBridge(
                n_task_classes=cfg.n_task_classes,
                n_social_classes=cfg.n_social_classes,
                target_mean=cfg.target_mean,
            )

        self._modality_order = list(cfg.feature_dims.keys())

    def _project_one_role(self, feats: dict[str, torch.Tensor]) -> torch.Tensor:
        if not feats:
            raise ValueError("at least one modality must be provided.")

        projected = self.projector(feats)

        if self._use_group_fusion:
            return self.group_fusion(projected)
        else:
            ref = next(iter(feats.values()))
            B, T, _ = ref.shape
            H = self.cfg.hidden_dim
            device, dtype = ref.device, ref.dtype
            parts: list[torch.Tensor] = []
            for name in self._modality_order:
                if name in projected:
                    parts.append(projected[name])
                else:
                    parts.append(torch.zeros(B, T, H, device=device, dtype=dtype))
            concat = torch.cat(parts, dim=-1)
            return self.down_proj(concat)

    def forward(self, batch: dict) -> dict[str, torch.Tensor]:
        t_proj = self._project_one_role(batch["target_feats"])
        partner_projs = [self._project_one_role(pf)
                         for pf in batch["partner_feats"]]
        partner_mask = batch.get("partner_present",
                                 [True] * len(partner_projs))
        partner_proj = self.partner_pool(partner_projs, partner_mask)

        B, T, D = t_proj.shape

        if self._flat_prompt:
            from .domain_prompt_flat import DomainPromptPool
            fallback = DomainPromptPool.DEFAULT_FALLBACKS.get(batch["domain"])
        else:
            fallback = HierarchicalDomainPrompt.DEFAULT_FALLBACKS.get(batch["domain"])
        prompt = self.domain_prompt.get_prompt(
            batch["domain"],
            batch_size=B,
            fallback_domain=fallback,
        )
        L_p = prompt.shape[1]
        t_seq = torch.cat([prompt, t_proj], dim=1)
        p_seq = torch.cat([prompt, partner_proj], dim=1)

        attn_mask = batch["attention_mask"]
        prompt_mask = torch.ones(B, L_p, dtype=torch.bool, device=attn_mask.device)
        full_mask = torch.cat([prompt_mask, attn_mask], dim=1)

        for layer in self.dapa_layers:
            t_seq, p_seq = layer(t_seq, p_seq, attn_mask=full_mask)

        t_frames = t_seq[:, L_p:, :]
        p_frames = p_seq[:, L_p:, :]

        dyad = torch.cat([t_frames, p_frames], dim=-1)
        out: dict[str, torch.Tensor] = {
            "reg":      self.reg_head(dyad),
            "features": dyad,
        }
        if self.cfg.enable_classification:
            cls = self.cls_heads(dyad)
            out["task_logits"]   = cls["task_logits"]
            out["social_logits"] = cls["social_logits"]
            if self.cfg.enable_bridge:
                out["bridged_reg"] = self.bridge(
                    cls["task_logits"], cls["social_logits"],
                )
        return out
