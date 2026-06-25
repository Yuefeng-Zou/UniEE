"""Test-Time Adaptation (v3 §8).

Adapts only LayerNorm + domain prompt parameters per test domain:
  - Regression: input perturbation consistency + smoothness
  - Classification: entropy minimization
"""
from __future__ import annotations

import copy
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class TestTimeAdapter:
    def __init__(self, model: nn.Module, lr: float = 1e-5):
        self.model = model
        self.original_state = copy.deepcopy(model.state_dict())

        for name, p in model.named_parameters():
            p.requires_grad = (
                "LayerNorm" in name or "layer_norm" in name
                or "norm" in name or "domain_prompt" in name
            )

        trainable = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(trainable, lr=lr)

    def adapt(self, loader: DataLoader, device: torch.device,
              n_epochs: int = 3, noise_std: float = 0.01) -> None:
        self.model.train()
        for _ in range(n_epochs):
            for batch in loader:
                batch = _to_device(batch, device)
                self.optimizer.zero_grad(set_to_none=True)

                domain = batch["domain"]
                if domain in ("pinsoro_cc", "pinsoro_cr"):
                    output = self.model(batch)
                    if "task_logits" not in output:
                        continue
                    p_t = output["task_logits"].softmax(-1)
                    p_s = output["social_logits"].softmax(-1)
                    ent_t = -(p_t * (p_t + 1e-8).log()).sum(-1).mean()
                    ent_s = -(p_s * (p_s + 1e-8).log()).sum(-1).mean()
                    loss = ent_t + ent_s
                else:
                    output = self.model(batch)
                    noisy_batch = _add_noise(batch, noise_std)
                    output_noisy = self.model(noisy_batch)
                    consistency = (
                        output["reg"] - output_noisy["reg"]
                    ).pow(2).mean()
                    smoothness = (
                        output["reg"][:, 1:] - output["reg"][:, :-1]
                    ).abs().mean()
                    loss = consistency + 0.05 * smoothness

                loss.backward()
                self.optimizer.step()

    def rollback(self) -> None:
        self.model.load_state_dict(self.original_state)

    def save_adapted(self, path: Path) -> None:
        torch.save({"model": self.model.state_dict()}, path)


def _add_noise(batch: dict, std: float) -> dict:
    noisy = dict(batch)
    noisy["target_feats"] = {
        k: v + torch.randn_like(v) * std for k, v in batch["target_feats"].items()
    }
    noisy["partner_feats"] = [
        {k: v + torch.randn_like(v) * std for k, v in p.items()}
        for p in batch["partner_feats"]
    ]
    return noisy


def _to_device(batch: dict, device: torch.device) -> dict:
    out = dict(batch)
    out["target_feats"] = {k: v.to(device) for k, v in batch["target_feats"].items()}
    out["partner_feats"] = [
        {k: v.to(device) for k, v in slot.items()}
        for slot in batch["partner_feats"]
    ]
    for k in ("label", "label_mask", "attention_mask", "label_task", "label_social",
              "label_pseudo_cont"):
        if k in batch:
            out[k] = batch[k].to(device)
    return out
