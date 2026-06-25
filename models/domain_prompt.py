"""Hierarchical domain prompt (v3 §4.3).

Two-level prompt: coarse (shared within coarse group) + fine (per-domain).
Coarse groups cluster domains by interaction type so related domains share
a structural prior while retaining domain-specific adaptation via fine prompts.

Unseen-language fallback (NoXi-add: ar/it/id/es) gets the 'conversational_adult'
coarse prompt automatically (same group as noxi/noxi_j) and falls back to the
'noxi' fine prompt since no noxi_add fine prompt is trained.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class HierarchicalDomainPrompt(nn.Module):

    COARSE_GROUPS = {
        "noxi":       "conversational_adult",
        "noxi_j":     "conversational_adult",
        "noxi_add":   "conversational_adult",
        "mpiigi":     "group_discussion",
        "pinsoro_cc": "child_freeplay",
        "pinsoro_cr": "child_freeplay",
    }

    DEFAULT_FALLBACKS = {"noxi_add": "noxi"}

    def __init__(self, hidden_dim: int,
                 n_prompts_coarse: int = 4,
                 n_prompts_fine: int = 8,
                 fine_domains: list[str] | None = None,
                 init_std: float = 0.02) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.n_prompts_coarse = n_prompts_coarse
        self.n_prompts_fine = n_prompts_fine

        if fine_domains is None:
            fine_domains = ["noxi", "noxi_j", "mpiigi", "pinsoro_cc", "pinsoro_cr"]
        self.fine_domains = list(fine_domains)

        coarse_names = sorted(set(self.COARSE_GROUPS.values()))
        self.coarse_prompts = nn.ParameterDict({
            cg: nn.Parameter(torch.randn(n_prompts_coarse, hidden_dim) * init_std)
            for cg in coarse_names
        })
        self.fine_prompts = nn.ParameterDict({
            fd: nn.Parameter(torch.randn(n_prompts_fine, hidden_dim) * init_std)
            for fd in self.fine_domains
        })

    def get_prompt(self, domain: str, batch_size: int,
                   fallback_domain: str | None = None) -> torch.Tensor:
        """Returns (B, n_coarse + n_fine, D)."""
        coarse_name = self.COARSE_GROUPS.get(domain)
        if coarse_name is None:
            raise KeyError(
                f"Unknown domain '{domain}' — not in COARSE_GROUPS. "
                f"Known: {list(self.COARSE_GROUPS)}"
            )
        coarse = self.coarse_prompts[coarse_name]

        if domain in self.fine_prompts:
            fine = self.fine_prompts[domain]
        elif fallback_domain is not None and fallback_domain in self.fine_prompts:
            fine = self.fine_prompts[fallback_domain]
        else:
            raise KeyError(
                f"No fine prompt for '{domain}' and no valid fallback. "
                f"Known fine domains: {self.fine_domains}. "
                f"For NoXi-add, pass fallback_domain='noxi'."
            )

        prompt = torch.cat([coarse, fine], dim=0)
        return prompt.unsqueeze(0).expand(batch_size, -1, -1)
