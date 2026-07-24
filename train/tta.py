"""Reusable regression test-time adaptation.

The paper protocol adapts LayerNorm affine parameters and hierarchical domain
prompts with input-noise consistency and temporal smoothness. Categorical
entropy minimization is intentionally not part of this class.
"""
from __future__ import annotations

import copy
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class TestTimeAdapter:
    def __init__(self, model: nn.Module, lr: float = 1e-5):
        self.model = model
        self.original_state = copy.deepcopy(model.state_dict())

        for param in model.parameters():
            param.requires_grad = False
        trainable = []
        selected_ids = set()
        for module in model.modules():
            if isinstance(module, nn.LayerNorm):
                for param in module.parameters(recurse=False):
                    param.requires_grad = True
                    if id(param) not in selected_ids:
                        trainable.append(param)
                        selected_ids.add(id(param))
        for name, param in model.named_parameters():
            if name.startswith("domain_prompt."):
                param.requires_grad = True
                if id(param) not in selected_ids:
                    trainable.append(param)
                    selected_ids.add(id(param))
        self.optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.0)

    def adapt(self, loader: DataLoader, device: torch.device,
              n_epochs: int = 3, noise_std: float = 0.01) -> None:
        self.model.eval()
        for _ in range(n_epochs):
            for batch in loader:
                batch = _to_device(batch, device)
                self.optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=device.type == "cuda",
                ):
                    output = self.model(batch)
                    noisy_batch = _add_noise(batch, noise_std)
                    output_noisy = self.model(noisy_batch)
                    valid = batch["attention_mask"]
                    consistency = (
                        (output["reg"] - output_noisy["reg"]).pow(2)
                        * valid.to(output["reg"].dtype)
                    ).sum() / valid.sum().clamp_min(1)
                    pair_valid = valid[:, 1:] & valid[:, :-1]
                    smoothness = (
                        output["reg"][:, 1:] - output["reg"][:, :-1]
                    ).abs()
                    smoothness = (
                        smoothness * pair_valid.to(smoothness.dtype)
                    ).sum() / pair_valid.sum().clamp_min(1)
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
    if "target_modality_present" in batch:
        out["target_modality_present"] = {
            k: v.to(device) for k, v in batch["target_modality_present"].items()
        }
    if "partner_modality_present" in batch:
        out["partner_modality_present"] = [
            {k: v.to(device) for k, v in slot.items()}
            for slot in batch["partner_modality_present"]
        ]
    for k in ("label", "label_mask", "attention_mask", "label_task", "label_social",
              "label_pseudo_cont"):
        if k in batch:
            out[k] = batch[k].to(device)
    return out
