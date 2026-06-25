"""Flat domain prompt pool (ablation baseline).

Single-level: one (L_p, D) prompt per domain. Used for ablation comparison
against HierarchicalDomainPrompt.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class DomainPromptPool(nn.Module):
    DEFAULT_FALLBACKS = {"noxi_add": "noxi"}

    def __init__(self, domains: list[str], hidden_dim: int,
                 prompt_len: int = 8, init_std: float = 0.02) -> None:
        super().__init__()
        self.domains = list(domains)
        self.hidden_dim = int(hidden_dim)
        self.prompt_len = int(prompt_len)
        self.prompts = nn.ParameterDict({
            d: nn.Parameter(torch.randn(self.prompt_len, self.hidden_dim) * init_std)
            for d in self.domains
        })

    def get_prompt(self, domain: str, batch_size: int,
                   fallback_domain: str | None = None) -> torch.Tensor:
        if domain not in self.prompts:
            if fallback_domain is not None and fallback_domain in self.prompts:
                domain = fallback_domain
            else:
                raise KeyError(f"unknown domain '{domain}'")
        p = self.prompts[domain]
        return p.unsqueeze(0).expand(batch_size, -1, -1)
