"""Regression tests for the implementation contract stated in the paper."""
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
MAIN_FEATURES = (
    "w2vbert2",
    "egemapsv2",
    "whisper",
    "xlmr",
    "openface2",
    "openface3",
    "openpose",
    "videomae",
    "dino",
    "swin",
    "clip",
)


def _load_module(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {relative_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class StaticPaperContractTests(unittest.TestCase):
    def test_main_feature_contract(self):
        config = yaml.safe_load((ROOT / "configs/feature_specs.yaml").read_text())
        dims = config["feature_dims"]
        preset = tuple(config["presets"]["mm26_whisper_full"])
        self.assertEqual(set(preset), set(MAIN_FEATURES))
        self.assertEqual(sum(dims[name] for name in preset), 7490)

        groups = config["groups"]
        grouped = {
            name
            for group_name, names in groups.items()
            if group_name != "vlm"
            for name in names
        }
        self.assertEqual(grouped, set(MAIN_FEATURES))

    def test_pinsoro_class_order_and_priors(self):
        labels = _load_module("uniee_label_loader", "data/label_loader.py")
        self.assertEqual(
            labels.PINSORO_TASK_CLASSES,
            ("goaloriented", "aimless", "noplay", "adultseeking"),
        )
        self.assertEqual(labels.PINSORO_TASK_FROM_IDX[2], "noplay")
        self.assertEqual(labels.PINSORO_TASK_FROM_IDX[3], "adultseeking")
        self.assertEqual(labels.PINSORO_TASK_CONT_PRIOR, (0.80, 0.40, 0.10, 0.55))
        self.assertEqual(labels.PINSORO_SOCIAL_CONT_PRIOR,
                         (0.10, 0.30, 0.50, 0.70, 0.90))

    def test_pinsoro_domain_config_has_four_task_classes(self):
        config = yaml.safe_load((ROOT / "configs/domain_config.yaml").read_text())
        for domain in ("pinsoro_cc", "pinsoro_cr"):
            self.assertEqual(config["label_space"][domain]["task_classes"], 4)
            self.assertEqual(config["label_space"][domain]["social_classes"], 5)

    def test_main_model_parameter_contract(self):
        feature_config = yaml.safe_load(
            (ROOT / "configs/feature_specs.yaml").read_text()
        )
        model_config = yaml.safe_load((ROOT / "configs/base.yaml").read_text())
        dims = feature_config["feature_dims"]
        d = model_config["model"]["hidden_dim"]
        n_features = len(MAIN_FEATURES)
        n_groups = 5

        projector = sum(dims[name] * d + 3 * d for name in MAIN_FEATURES)
        transformer_layer = 8 * d * d + 11 * d
        group_fusion = (n_groups + 1) * transformer_layer
        prompts = (3 * 4 + 5 * 8) * d
        partner_pool = 4 * d * d + 5 * d

        half = d // 2
        one_bilstm = 2 * (4 * half * d + 4 * half * half + 8 * half)
        four_half_dim_attentions = 4 * (4 * half * half + 4 * half)
        one_interaction_layer = (
            2 * one_bilstm + four_half_dim_attentions + 4 * d
        )
        interactions = (
            model_config["model"]["n_dapa_layers"] * one_interaction_layer
        )

        head_in = 2 * d
        regression_head = head_in * d + d + d + 1
        classification_heads = head_in * 4 + 4 + head_in * 5 + 5
        bridge = 9 * 32 + 32 + 32 + 1

        total = (
            projector
            + group_fusion
            + prompts
            + partner_pool
            + interactions
            + regression_head
            + classification_heads
            + bridge
        )
        self.assertEqual(n_features, 11)
        self.assertEqual(total, 18_016_875)


try:
    import torch
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed in this environment")
class TorchPaperContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if "multimediate26" not in sys.modules:
            spec = importlib.util.spec_from_file_location(
                "multimediate26",
                ROOT / "__init__.py",
                submodule_search_locations=[str(ROOT)],
            )
            package = importlib.util.module_from_spec(spec)
            sys.modules["multimediate26"] = package
            spec.loader.exec_module(package)

    def test_ordinal_distance_formula(self):
        from multimediate26.losses.ordinal_distance_matching import (
            ordinal_distance_matching_loss,
        )

        features = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
        mask = torch.ones(1, 2, dtype=torch.bool)
        matched = ordinal_distance_matching_loss(
            features, torch.tensor([[0.0, 0.5]]), mask,
        )
        separated = ordinal_distance_matching_loss(
            features, torch.tensor([[0.0, 1.0]]), mask,
        )
        self.assertAlmostEqual(float(matched), 0.0, places=6)
        self.assertAlmostEqual(float(separated), 0.25, places=6)

    def test_single_partner_mask_is_per_sample(self):
        from multimediate26.models.partner_pool import MultiPartnerPooling

        pool = MultiPartnerPooling(hidden_dim=4, n_heads=1, max_partners=1)
        partner = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
        mask = torch.tensor([[True], [False]])
        output = pool([partner], mask)
        self.assertTrue(torch.equal(output[0], partner[0]))
        self.assertTrue(torch.equal(output[1], torch.zeros_like(partner[1])))

    def test_collate_preserves_modality_union_and_partner_masks(self):
        from multimediate26.data.dataset import collate

        def item(modalities, partner_present):
            feats = {
                name: torch.ones(2, dim)
                for name, dim in modalities.items()
            }
            return {
                "target_feats": feats,
                "partner_feats": [dict(feats)],
                "partner_present": [partner_present],
                "label": torch.zeros(2),
                "label_mask": torch.ones(2, dtype=torch.bool),
                "attention_mask": torch.ones(2, dtype=torch.bool),
                "session_key": "noxi/s/r",
                "window_start": 0,
                "valid_end": 2,
                "domain": "noxi",
            }

        batch = collate([
            item({"audio": 3}, True),
            item({"audio": 3, "visual": 2}, False),
        ])
        self.assertEqual(set(batch["target_feats"]), {"audio", "visual"})
        self.assertTrue(torch.equal(
            batch["target_feats"]["visual"][0],
            torch.zeros(2, 2),
        ))
        self.assertTrue(torch.equal(
            batch["target_modality_present"]["visual"],
            torch.tensor([False, True]),
        ))
        self.assertTrue(torch.equal(
            batch["partner_present"],
            torch.tensor([[True], [False]]),
        ))


if __name__ == "__main__":
    unittest.main()
