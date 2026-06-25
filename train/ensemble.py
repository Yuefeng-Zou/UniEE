"""Multi-seed / multi-window ensemble + submission writer (v3 §9-10)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.signal import savgol_filter


class EnsemblePredictor:
    def __init__(self, models: list[nn.Module],
                 window_lens: tuple[int, ...] = (256, 512, 1024)):
        self.models = models
        self.window_lens = window_lens

    @torch.no_grad()
    def predict_session(self, feats: dict[str, torch.Tensor],
                        partners_feats: list[dict[str, torch.Tensor]],
                        partner_present: list[bool],
                        domain: str, device: torch.device,
                        task: str = "reg") -> np.ndarray:
        preds = []
        for model in self.models:
            model.eval()
            for wl in self.window_lens:
                p = self._sliding_predict(
                    model, feats, partners_feats, partner_present,
                    wl, domain, device, task,
                )
                preds.append(p)
        ensemble = np.stack(preds).mean(axis=0)
        if task == "reg":
            wl = min(25, len(ensemble) // 2 * 2 - 1)
            if wl >= 5:
                ensemble = savgol_filter(ensemble, wl, 3)
            ensemble = np.clip(ensemble, 0.0, 1.0)
        return ensemble

    def _sliding_predict(self, model: nn.Module,
                         feats: dict[str, torch.Tensor],
                         partners_feats: list[dict[str, torch.Tensor]],
                         partner_present: list[bool],
                         window_len: int, domain: str,
                         device: torch.device,
                         task: str) -> np.ndarray:
        ref = next(iter(feats.values()))
        T = ref.shape[0]
        stride = window_len // 2
        accumulator = np.zeros(T, dtype=np.float64)
        counter = np.zeros(T, dtype=np.float64)
        weight = np.hanning(window_len)

        for start in range(0, T, stride):
            end = min(start + window_len, T)
            actual_wl = end - start
            chunk = {k: v[start:end].unsqueeze(0).to(device) for k, v in feats.items()}
            chunk_partners = [
                {k: v[start:end].unsqueeze(0).to(device) for k, v in p.items()}
                for p in partners_feats
            ]
            mask = torch.ones(1, actual_wl, dtype=torch.bool, device=device)
            batch = {
                "target_feats": chunk,
                "partner_feats": chunk_partners,
                "partner_present": partner_present,
                "attention_mask": mask,
                "domain": domain,
            }
            out = model(batch)
            if task == "reg":
                pred = out["reg"].squeeze(0).cpu().numpy()
            elif task == "task":
                pred = out["task_logits"].argmax(-1).squeeze(0).cpu().numpy().astype(float)
            elif task == "social":
                pred = out["social_logits"].argmax(-1).squeeze(0).cpu().numpy().astype(float)
            else:
                raise ValueError(f"Unknown task: {task}")

            w = weight[:actual_wl] if actual_wl == window_len else np.hanning(actual_wl)
            accumulator[start:end] += pred * w
            counter[start:end] += w

        return accumulator / np.maximum(counter, 1e-8)


def write_submission(session_id: str, predictions: dict,
                     domain: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{session_id}.csv"
    if domain.startswith("pinsoro"):
        df = pd.DataFrame({
            "frame": np.arange(len(predictions["task"])),
            "task_engagement": predictions["task"].astype(int),
            "social_engagement": predictions["social"].astype(int),
        })
    else:
        df = pd.DataFrame({
            "frame": np.arange(len(predictions["reg"])),
            "engagement": predictions["reg"],
        })
    df.to_csv(out_path, index=False)
    return out_path
