from __future__ import annotations

import unittest
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from core.progress import ETAEstimator, format_duration
from PIL import Image
from torchvision.transforms import functional as TF

from ..bag_store import PackedBagStore
from ..cohort import EXPECTED_AUDIT, build_audit, load_cohort
from ..config import load_config, resolved_paths
from ..manifests import build_pair_manifest, build_split_manifest, validate_split_manifest
from ..model import CommonStage2
from ..encoders.base import FrozenRegionEncoder
from ..encoders.topodise import TopoDiSEStage1NativeZSemFrozen
from ..feature_cache import build_patient_bags, prepare_feature_cache
from ..pretrain_simclr import collect_pretrain_images
from ..trainer import run_outer_fold
from topodise.encoder import EncoderConfig, TopoDiSEEncoder


class _DummyFrozenEncoder(FrozenRegionEncoder):
    encoder_name = "dummy"
    raw_dim = 3
    checkpoint_path = Path(__file__)
    checkpoint_sha256 = "dummy-checkpoint"
    preprocessing_name = "dummy-rgb-16"

    def __init__(self):
        super().__init__()
        self.transform = self._transform

    @staticmethod
    def _transform(image: Image.Image) -> torch.Tensor:
        return TF.to_tensor(TF.resize(image.convert("RGB"), [16, 16]))

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        return self.transform(image)

    def encode_region(self, images: torch.Tensor, region: str) -> torch.Tensor:
        return images.mean(dim=(-2, -1))


class _DummyFrozenPairEncoder(_DummyFrozenEncoder):
    encoder_name = "dummy_native_pair"
    feature_mode = "native_pair"
    checkpoint_sha256 = "dummy-native-pair-checkpoint"

    def encode_pair(
        self, central_images: torch.Tensor, whorl_images: torch.Tensor
    ) -> torch.Tensor:
        return 0.5 * (
            central_images.mean(dim=(-2, -1)) + whorl_images.mean(dim=(-2, -1))
        )


class ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = load_config(
            "experiments/unified_stage2_encoder_comparison/configs/protocol_v1.yaml"
        )
        cls.paths = resolved_paths(cls.cfg)
        cls.records, cls.exclusions = load_cohort(cls.paths["dataset_json"], cls.paths["dataset_root"])

    def test_locked_cohort_counts(self):
        audit = build_audit(self.paths["dataset_json"], self.records, self.exclusions)
        observed = {key: audit[key] for key in EXPECTED_AUDIT}
        self.assertEqual(observed, EXPECTED_AUDIT)

    def test_pair_manifest_is_deterministic_and_capped(self):
        first = build_pair_manifest(self.records)
        second = build_pair_manifest(self.records)
        self.assertEqual(first["manifest_sha256"], second["manifest_sha256"])
        self.assertTrue(all(len(p["selected_pairs"]) <= 8 for p in first["patients"]))

    def test_patient_splits_have_no_leakage(self):
        for task in self.cfg["tasks"]:
            manifest = build_split_manifest(self.records, task, 42)
            validate_split_manifest(manifest)

    def test_common_model_contract(self):
        model = CommonStage2(raw_dim=384, num_classes=3)
        logits, attention = model(torch.randn(8, 384), torch.randn(8, 384), torch.randn(21))
        self.assertEqual(tuple(logits.shape), (1, 3))
        self.assertEqual(tuple(attention.shape), (8,))
        self.assertAlmostEqual(float(attention.detach().sum()), 1.0, places=5)

    def test_native_pair_model_bypasses_pair_composer(self):
        model = CommonStage2(raw_dim=16, num_classes=3, input_mode="native_pair")
        logits, attention = model(torch.randn(8, 16), None, torch.randn(21))
        self.assertIsNone(model.pair_composer)
        self.assertEqual(tuple(logits.shape), (1, 3))
        self.assertEqual(tuple(attention.shape), (8,))

    def test_native_topodise_checkpoint_loads_fusion_and_projection(self):
        config = EncoderConfig(
            base_channels=8,
            channel_mult=(1, 1),
            blocks_per_stage=(1, 1),
            z_dim=16,
            image_size=32,
        )
        source = TopoDiSEEncoder(config)
        parent_config = {
            "encoder_base_channels": config.base_channels,
            "encoder_channel_mult": config.channel_mult,
            "encoder_blocks_per_stage": config.blocks_per_stage,
            "z_dim": config.z_dim,
            "image_size": config.image_size,
        }
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "ema_best.pt"
            torch.save({
                "parent_config": parent_config,
                "ema_encoder": source.state_dict(),
            }, checkpoint)
            frozen = TopoDiSEStage1NativeZSemFrozen(checkpoint)
            with torch.inference_mode():
                z_sem = frozen.encode_pair(
                    torch.randn(2, 1, 32, 32), torch.randn(2, 1, 32, 32)
                )
        self.assertEqual(tuple(z_sem.shape), (2, 16))
        self.assertFalse(frozen.load_missing_keys)
        self.assertFalse(frozen.load_unexpected_keys)
        self.assertTrue(all(not parameter.requires_grad for parameter in frozen.parameters()))

    def test_batched_mask_matches_single_patient_eval(self):
        model = CommonStage2(raw_dim=16, num_classes=2, attention_dropout=0.0).eval()
        central = torch.randn(3, 16)
        whorl = torch.randn(3, 16)
        morphology = torch.randn(21)
        with torch.inference_mode():
            logits_single, _ = model(central, whorl, morphology)
            padded_c = torch.cat([central, torch.zeros(2, 16)]).unsqueeze(0)
            padded_w = torch.cat([whorl, torch.zeros(2, 16)]).unsqueeze(0)
            mask = torch.tensor([[True, True, True, False, False]])
            logits_batch, attention = model(
                padded_c, padded_w, morphology.unsqueeze(0), mask
            )
        self.assertTrue(torch.allclose(logits_single, logits_batch, atol=1e-6))
        self.assertEqual(float(attention[0, 3:].sum()), 0.0)

    def test_packed_store_keeps_variable_bag_masks(self):
        bags = {
            "A": {
                "central": torch.randn(2, 4), "whorl": torch.randn(2, 4),
                "morphology": torch.randn(21), "label": 0,
            },
            "B": {
                "central": torch.randn(4, 4), "whorl": torch.randn(4, 4),
                "morphology": torch.randn(21), "label": 1,
            },
        }
        store = PackedBagStore.from_bags(bags, torch.device("cpu"))
        *_, labels, mask = store.batch(["B", "A"])
        self.assertEqual(labels.tolist(), [1, 0])
        self.assertEqual(mask.sum(dim=1).tolist(), [4, 2])

    def test_packed_store_supports_native_pair_features(self):
        bags = {
            "A": {
                "pair_features": torch.randn(2, 4),
                "morphology": torch.randn(21), "label": 0,
            },
            "B": {
                "pair_features": torch.randn(4, 4),
                "morphology": torch.randn(21), "label": 1,
            },
        }
        store = PackedBagStore.from_bags(bags, torch.device("cpu"))
        pair_features, whorl, _, labels, mask = store.batch(["B", "A"])
        self.assertEqual(store.feature_mode, "native_pair")
        self.assertIsNone(whorl)
        self.assertEqual(tuple(pair_features.shape), (2, 4, 4))
        self.assertEqual(labels.tolist(), [1, 0])
        self.assertEqual(mask.sum(dim=1).tolist(), [4, 2])

    def test_batched_training_gradient_matches_eight_patient_accumulation(self):
        torch.manual_seed(9)
        batch_size, pairs, raw_dim = 8, 3, 4
        batched_model = CommonStage2(
            raw_dim=raw_dim, num_classes=2, common_dim=8, adapter_rank=2,
            attention_hidden=4, attention_dropout=0.0, morphology_dim=2,
        )
        sequential_model = deepcopy(batched_model)
        central = torch.randn(batch_size, pairs, raw_dim)
        whorl = torch.randn(batch_size, pairs, raw_dim)
        morphology = torch.randn(batch_size, 21)
        mask = torch.ones(batch_size, pairs, dtype=torch.bool)
        targets = torch.tensor([0, 1, 0, 1, 1, 1, 0, 1])
        weights = torch.tensor([1.5, 0.5])

        logits, _ = batched_model(central, whorl, morphology, mask)
        torch.nn.functional.cross_entropy(
            logits, targets, weight=weights, reduction="none"
        ).sum().div(batch_size).backward()
        for index in range(batch_size):
            logits_one, _ = sequential_model(
                central[index], whorl[index], morphology[index]
            )
            torch.nn.functional.cross_entropy(
                logits_one, targets[index:index + 1], weight=weights, reduction="none"
            ).sum().div(batch_size).backward()
        for batched_parameter, sequential_parameter in zip(
            batched_model.parameters(), sequential_model.parameters()
        ):
            self.assertTrue(torch.allclose(
                batched_parameter.grad, sequential_parameter.grad, atol=2e-6, rtol=1e-5
            ))

    def test_parallel_feature_loader_fuses_morphology_decode(self):
        pair_manifest = build_pair_manifest(self.records[:2])
        progress_events = []
        with TemporaryDirectory() as directory:
            payload, cache_dir = prepare_feature_cache(
                _DummyFrozenEncoder(), pair_manifest, Path(directory), torch.device("cpu"),
                batch_size=2, amp=False, num_workers=2, pin_memory=False,
                channels_last=True, progress_callback=progress_events.append,
            )
            selected = [
                pair
                for patient in pair_manifest["patients"]
                for pair in patient["selected_pairs"]
            ]
            expected_central = {pair["central"] for pair in selected}
            expected_whorl = {pair["whorl"] for pair in selected}
            self.assertEqual(set(payload["central"]), expected_central)
            self.assertEqual(set(payload["whorl"]), expected_whorl)
            self.assertEqual(set(payload["morph"]), expected_central | expected_whorl)
            self.assertTrue((cache_dir / "features.pt").is_file())
            self.assertTrue(any(row["kind"] == "feature_progress" for row in progress_events))

    def test_native_pair_feature_cache_uses_manifest_pairs(self):
        pair_manifest = build_pair_manifest(self.records[:2])
        progress_events = []
        with TemporaryDirectory() as directory:
            payload, cache_dir = prepare_feature_cache(
                _DummyFrozenPairEncoder(), pair_manifest, Path(directory),
                torch.device("cpu"), batch_size=2, amp=False, num_workers=0,
                pin_memory=False, channels_last=True,
                progress_callback=progress_events.append,
            )
            expected_ids = {
                pair["rank_sha256"]
                for patient in pair_manifest["patients"]
                for pair in patient["selected_pairs"]
            }
            self.assertEqual(set(payload["pair"]), expected_ids)
            self.assertEqual(set(payload["pair_morphology"]), expected_ids)
            labels = {patient["patient_id"]: 0 for patient in pair_manifest["patients"]}
            bags = build_patient_bags(pair_manifest, payload, labels)
            self.assertTrue(all("pair_features" in bag for bag in bags.values()))
            self.assertTrue((cache_dir / "features.pt").is_file())
            self.assertTrue(any(
                row.get("region") == "native_pair" for row in progress_events
            ))

    def test_native_pair_outer_fold_smoke(self):
        generator = torch.Generator().manual_seed(17)
        bags = {}
        for index in range(16):
            label = index % 2
            bags[f"N{index:02d}"] = {
                "pair_features": torch.randn(3, 8, generator=generator) + label * 0.2,
                "morphology": torch.randn(21, generator=generator),
                "label": label,
            }
        fold = {
            "fold": 0,
            "outer_train": [f"N{i:02d}" for i in range(12)],
            "inner_train": [f"N{i:02d}" for i in range(8)],
            "inner_val": [f"N{i:02d}" for i in range(8, 12)],
            "outer_test": [f"N{i:02d}" for i in range(12, 16)],
        }
        with TemporaryDirectory() as directory:
            summary = run_outer_fold(
                raw_dim=8,
                num_classes=2,
                model_cfg={
                    "common_dim": 16,
                    "adapter_rank": 2,
                    "attention_hidden": 4,
                    "attention_dropout": 0.0,
                    "morphology_dim": 2,
                    "input_mode": "native_pair",
                },
                optim_cfg={
                    "epochs": 1,
                    "warmup_epochs": 0,
                    "lr": 1e-3,
                    "min_lr": 1e-6,
                    "weight_decay": 1e-2,
                    "grad_clip": 1.0,
                    "accumulate_patients": 4,
                    "patient_batch_size": 4,
                    "eval_patient_batch_size": 8,
                },
                bags=bags,
                fold_spec=fold,
                task="pd_binary",
                encoder_name="topodise_native_zsem",
                split_seed=17,
                device=torch.device("cpu"),
                amp=False,
                run_dir=Path(directory),
                provenance={
                    "protocol_version": "test",
                    "protocol_config_sha256": "test",
                },
            )
            self.assertEqual(summary["stage2_input_mode"], "native_pair")
            self.assertEqual(summary["outer_metrics"]["n"], 4)
            self.assertTrue((Path(directory) / "DONE.json").is_file())

    def test_eta_estimator_uses_recent_completed_units(self):
        estimator = ETAEstimator(10, window=5)
        estimator.update(1, interval_seconds=2.0)
        estimator.update(2, interval_seconds=4.0)
        state = estimator.update(3, interval_seconds=3.0)
        self.assertEqual(state["rolling_unit_seconds"], 3.0)
        self.assertEqual(state["eta_seconds"], 21.0)
        self.assertEqual(state["eta_quality"], "usable")
        self.assertEqual(format_duration(state["eta_seconds"]), "21s")

    def test_pretraining_sources_are_data_matched_and_label_free(self):
        topo = self.cfg["encoders"]["topodise_stage1"]["pretrain"]
        simclr = self.cfg["encoders"]["simclr_resnet18"]["pretrain"]
        self.assertEqual(topo["stage0"]["topo_target"], "hgrad")
        self.assertFalse(topo["stage0"]["amp"])
        self.assertFalse(topo["stage0"]["channels_last"])
        self.assertIsNone(topo["stage1"]["region_pair_manifest"])
        self.assertEqual(topo["stage1"]["parent_encoder_weights"], "raw")
        self.assertFalse(topo["stage1"]["amp"])
        self.assertFalse(topo["stage1"]["allow_dm_disease"])
        self.assertFalse(simclr["initialization"])
        self.assertEqual(
            [stage["source"] for stage in simclr["stages"]],
            ["corn1_raw", "region_manifest"],
        )
        collected = [collect_pretrain_images(stage) for stage in simclr["stages"]]
        self.assertEqual(len(collected[0][0]), 1516)
        self.assertGreaterEqual(collected[1][1]["num_paired_region_rows"], 2)
        self.assertFalse(collected[0][1]["uses_segmentation_masks"])

if __name__ == "__main__":
    unittest.main()
