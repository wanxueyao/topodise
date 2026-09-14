import tempfile
import unittest
from pathlib import Path

from experiments.topodise_experiments.batch_experiments import (
    BatchRunner,
    _normalize_reproducibility_mode,
    resolve_checkpoint_bundle,
)


class ReproducibilityModeTests(unittest.TestCase):
    def test_mode_aliases(self):
        expected = {
            "strict": "strict",
            "portable": "portable",
            "checkpoint": "checkpoint",
            True: "strict",
            False: "portable",
            "1": "strict",
            "2": "portable",
            "3": "checkpoint",
        }
        for value, mode in expected.items():
            with self.subTest(value=value):
                self.assertEqual(_normalize_reproducibility_mode(value), mode)

    def test_minimal_stage1_bundle(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "stage1" / "ema_best.pt"
            checkpoint.parent.mkdir()
            checkpoint.write_bytes(b"stage1-checkpoint")

            bundle = resolve_checkpoint_bundle(str(root))

            self.assertEqual(bundle["stage1_checkpoint"], str(checkpoint))
            self.assertIsNone(bundle["stage0_checkpoint"])
            self.assertEqual(
                bundle["stage1_sha256"],
                "a0913cc124a697ae704627d4929e51967f1537d3490ec0a4f65912c9d7516235",
            )

    def test_full_bundle_layout(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stage0 = root / "run_stage0" / "ckpt" / "ema_best.pt"
            stage1 = root / "run_stage1" / "ckpt" / "ema_best.pt"
            stage0.parent.mkdir(parents=True)
            stage1.parent.mkdir(parents=True)
            stage0.write_bytes(b"stage0")
            stage1.write_bytes(b"stage1")
            (root / "_experiment_def.json").write_text("{}")
            (root / "summary.json").write_text("{}")

            bundle = resolve_checkpoint_bundle(str(root))

            self.assertEqual(bundle["stage0_checkpoint"], str(stage0))
            self.assertEqual(bundle["stage1_checkpoint"], str(stage1))
            self.assertEqual(len(bundle["metadata_files"]), 2)

    def test_checkpoint_mode_rewrites_nested_stage2_jobs(self):
        runner = object.__new__(BatchRunner)
        node = {
            "stages": [2],
            "stage2_overrides": {},
            "oracle_experiments": [
                {"stages": [2], "stage2_overrides": {}},
            ],
        }

        runner._apply_reproducibility_mode(node, "checkpoint", "/tmp/stage1.pt")

        self.assertEqual(node["parent_ckpt_override"], "/tmp/stage1.pt")
        self.assertTrue(node["stage2_overrides"]["deterministic"])
        child = node["oracle_experiments"][0]
        self.assertEqual(child["parent_ckpt_override"], "/tmp/stage1.pt")
        self.assertEqual(child["stages"], [2])

    def test_strict_mode_reaches_explicit_empty_stage_overrides(self):
        runner = object.__new__(BatchRunner)
        node = {
            "stages": [0, 1, 2],
            "stage0_overrides": {},
            "stage1_overrides": {},
            "stage2_overrides": {},
        }

        runner._apply_reproducibility_mode(node, "strict", None)

        self.assertEqual(node["stage0_overrides"]["reproducibility"], "strict")
        self.assertEqual(node["stage1_overrides"]["reproducibility"], "strict")
        self.assertTrue(node["stage2_overrides"]["deterministic"])

    def test_strict_stage2_only_job_builds_from_zero_recipe(self):
        runner = BatchRunner({
            "PROJECT_ROOT": "/tmp/project",
            "STAGE_FILTER": None,
            "EXTERNAL_PARENT_CKPT": None,
            "DEFAULT_STAGE0_OVERRIDES": {"reproducibility": "strict"},
            "DEFAULT_STAGE1_OVERRIDES": {"reproducibility": "strict"},
        })
        node = {
            "stages": [2],
            "stage0_overrides": {},
            "stage1_overrides": {},
            "stage2_overrides": {},
        }
        runner._apply_reproducibility_mode(node, "strict", None)

        recipe = runner._fallback_recipe(node)

        self.assertIsNotNone(recipe)
        self.assertEqual(recipe[0]["reproducibility"], "strict")
        self.assertEqual(recipe[1]["reproducibility"], "strict")


if __name__ == "__main__":
    unittest.main()
