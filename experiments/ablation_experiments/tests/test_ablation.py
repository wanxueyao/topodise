"""CPU correctness checks and all-six-route miniature training integration."""
from copy import deepcopy
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
from PIL import Image
import pytest
import torch
from torchvision import transforms as T

from experiments.ablation_experiments import batch_experiments as batch
from experiments.ablation_experiments.aligned_corn import AlignedCORN1Dataset
from experiments.ablation_experiments.protocol import ROUTES, ENCODER, checkpoint, lock, read, write
from experiments.ablation_experiments.topo_targets import compute_topo_target
from experiments.ablation_experiments.worker import archive_incomplete, run_pretrain, run_stage2, training_overrides
from topodise.data import CORN1Dataset, build_pretrain_transform
from topodise.topo_targets import compute_topo_target as original_target


@pytest.fixture(autouse=True, scope="module")
def cpu_threads():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def test_hgrad_dual_thresholds_and_batch_independence():
    torch.manual_seed(4)
    x = torch.randn(2, 1, 64, 64)
    target = compute_topo_target(x, "hgrad_dual", 16)
    assert target.shape == (2, 2, 16, 16)
    assert torch.all(target[:, 1] <= target[:, 0])
    assert torch.any(target[:, 0] != target[:, 1])
    torch.testing.assert_close(target[:1], compute_topo_target(x[:1], "hgrad_dual", 16))
    altered = x.clone()
    altered[1] *= 1000
    torch.testing.assert_close(target[:1], compute_topo_target(altered, "hgrad_dual", 16)[:1])
    pooled = torch.nn.functional.adaptive_avg_pool2d(
        torch.nn.functional.pad((x[..., 1:] - x[..., :-1]).abs(), (1, 0, 0, 0)), 16)
    mu = pooled.mean((2, 3), keepdim=True)
    std = pooled.std((2, 3), keepdim=True)
    torch.testing.assert_close(target[:, :1], (pooled > mu).float())
    torch.testing.assert_close(target[:, 1:], (pooled > mu + std).float())
    assert compute_topo_target(torch.zeros_like(x), "hgrad_dual", 16).sum() == 0


@pytest.mark.parametrize("mode", ["gt", "frangi", "hgrad"])
def test_historical_targets_unchanged(mode):
    x = torch.randn(2, 1, 32, 32)
    mask = (torch.rand_like(x) > .7).float()
    torch.testing.assert_close(compute_topo_target(x, mode, 16, mask=mask),
                               original_target(x, mode, 16, mask=mask), rtol=0, atol=0)
    if mode == "hgrad":
        t = compute_topo_target(x, mode)
        assert torch.equal(t[:, 0], t[:, 1])


def make_corn(root, count=4):
    images = root / "images/train_orig"
    labels = root / "labels/train_label"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    for n in range(count):
        arr = np.zeros((40, 40), dtype=np.uint8)
        arr[5:30, 4 + n:10 + n] = 255
        arr[20:25, 9:35] = 255
        Image.fromarray(arr).save(images / f"{n}.png")
        Image.fromarray(arr).save(labels / f"{n}.png")
    return root


def test_mask_geometry_and_image_rng_are_shared(tmp_path):
    root = make_corn(tmp_path / "CORN-1")
    transform = build_pretrain_transform(32)
    # Nearest on the test image makes exact image-mask equality measurable.
    transform.transforms[0].interpolation = T.InterpolationMode.NEAREST
    plain = CORN1Dataset(str(root), transform=transform)
    paired = AlignedCORN1Dataset(str(root), transform=transform, load_masks=True)
    for seed in range(10):
        torch.manual_seed(seed)
        a = plain[0]
        rng = torch.get_rng_state()
        torch.manual_seed(seed)
        b = paired[0]
        for key in a:
            torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
        assert torch.equal(rng, torch.get_rng_state())
        assert torch.equal(b["x_mask"], (b["x_full"] > 0).float())
    (root / "labels/train_label/0.png").unlink()
    with pytest.raises(FileNotFoundError):
        AlignedCORN1Dataset(str(root), transform=transform, load_masks=True)


def test_plan_shares_three_parents_and_locks_one_stage2(tmp_path):
    jobs = batch.build_plan(tmp_path, tmp_path, tmp_path, tmp_path / "pd.json", tmp_path / "manifest.json", [42], backend="unified")
    assert [len(jobs[s]) for s in jobs] == [3, 6, 6]
    assert len({j["config"]["parent_ckpt"] for j in jobs["stage1"]}) == 3
    assert {j["config"]["topo_target"] for j in jobs["stage1"]} == {"hgrad_dual", "frangi"}
    assert sum(j["config"]["topo_target"] == "hgrad_dual" for j in jobs["stage1"]) == 3
    assert {j["encoder"] for j in jobs["stage2"]} == {ENCODER}
    cores = []
    for job in jobs["stage2"]:
        cfg = deepcopy(job["config"])
        cfg.pop("output_root")
        cfg["encoders"]["topodise_stage1"].pop("checkpoint")
        assert cfg["encoders"]["topodise_stage1"]["auto_pretrain_if_missing"] is False
        cores.append(cfg)
    assert all(c == cores[0] for c in cores)
    subset = batch.selected_jobs(jobs, "all", ["gt_frangi", "gt_hgrad_dual"])
    assert [len(subset[s]) for s in subset] == [1, 2, 2]
    lock(tmp_path / "lock.json", jobs)
    lock(tmp_path / "lock.json", jobs)
    with pytest.raises(RuntimeError):
        lock(tmp_path / "lock.json", {})


def test_incomplete_stage2_archived_without_touching_complete(tmp_path):
    cfg = {"tasks": ["pd_binary"], "splits": {"seeds": [42], "outer_folds": 5}}
    base = tmp_path / "runs" / ENCODER / "pd_binary/seed42"
    write(base / "fold0/DONE.json", {"status": "complete"})
    write(base / "fold1/summary.json", {"interrupted": True})
    archived = archive_incomplete(tmp_path, ENCODER, cfg)
    assert len(archived) == 1
    assert read(Path(archived[0]) / "summary.json")["interrupted"]
    assert (base / "fold0/DONE.json").exists()
    assert not (base / "fold1").exists()


def test_all_six_train_and_feed_frozen_unified_stage2(tmp_path, monkeypatch):
    from experiments.unified_stage2_encoder_comparison.encoders.factory import build_frozen_encoder
    from experiments.unified_stage2_encoder_comparison.config import DEFAULTS
    from experiments.unified_stage2_encoder_comparison.trainer import run_outer_fold
    from experiments.topodise_experiments.stage0_corn1_pretrain import main as stage0
    from experiments.topodise_experiments.stage1_rscp_finetune import main as stage1
    corn = make_corn(tmp_path / "CORN-1")
    regions = tmp_path / "unlabeled_dataset_en/test"
    regions.mkdir(parents=True)
    pairs = []
    rng = np.random.default_rng(5)
    for n in range(4):
        paths = []
        for region in ("central", "whorl"):
            path = regions / f"{n}_{region}.png"
            Image.fromarray(rng.integers(0, 255, (32, 32), dtype=np.uint8)).save(path)
            paths.append(str(path))
        pairs.append({"subject_id": str(n), "central": paths[0], "whorl": paths[1], "is_paired": True})
    manifest = tmp_path / "manifest.json"
    write(manifest, pairs)
    jobs = batch.build_plan(tmp_path / "results", tmp_path, corn, tmp_path / "pd.json", manifest, [42], backend="unified")
    for job in jobs["stage0"]:
        job["config"].update(epochs=1, batch_size=2, num_workers=0, expected_samples=4,
                             image_size=32, encoder_base_channels=8,
                             encoder_channel_mult=(1, 1), encoder_blocks_per_stage=(1, 1),
                             z_dim=16, unet_model_channels=8, unet_channel_mult=(1, 1), T=8)
        run_pretrain(job, "cpu")
    for job in jobs["stage1"]:
        job["config"].update(epochs=1, batch_size=2, num_workers=0, image_size=32)
        run_pretrain(job, "cpu")
        # Completion markers skip a second invocation, not reinitialize training.
        run_pretrain(job, "cpu")
    assert stage0.compute_topo_target is original_target
    assert stage1.compute_topo_target is original_target
    assert stage0.CORN1Dataset is CORN1Dataset
    for job in jobs["stage2"]:
        frozen = build_frozen_encoder(ENCODER, job["config"]["encoders"]["topodise_stage1"], "cpu")
        frozen.assert_frozen()
        before = {k: v.clone() for k, v in frozen.state_dict().items()}
        with torch.inference_mode():
            features = frozen.encode_pair(torch.randn(8, 1, 32, 32), torch.randn(8, 1, 32, 32))
        bags = {str(n): {"pair_features": features[n:n+1].clone(),
                         "morphology": torch.zeros(21), "label": n % 2} for n in range(8)}
        cfg = deepcopy(DEFAULTS)
        cfg["optimization"].update(epochs=1, warmup_epochs=0)
        summary = run_outer_fold(
            raw_dim=16, num_classes=2, model_cfg={**cfg["model"], "input_mode": "native_pair"},
            optim_cfg=cfg["optimization"], bags=bags,
            fold_spec={"fold": 0, "inner_train": ["0", "1", "2", "3"], "inner_val": ["4", "5"],
                       "outer_train": [str(n) for n in range(6)], "outer_test": ["6", "7"]},
            task="pd_binary", encoder_name=ENCODER, split_seed=42, device=torch.device("cpu"),
            amp=False, run_dir=Path(job["config"]["output_root"]) / "synthetic_fold", provenance={})
        assert summary["num_test_patients"] == 2
        assert summary["stage2_input_mode"] == "native_pair"
        assert all(torch.equal(before[k], v) for k, v in frozen.state_dict().items())
        assert all(p.grad is None and not p.requires_grad for p in frozen.parameters())
    # Check that the production Stage2 wrapper dispatches the original runner
    # with full defaults and the right checkpoint (training above was real).
    seen = []
    def inspect_dispatch():
        args = sys.argv
        config = read(args[args.index("--config") + 1])
        assert args[args.index("--encoder") + 1] == ENCODER
        assert config["optimization"] == DEFAULTS["optimization"]
        seen.append(config)
    monkeypatch.setattr("experiments.unified_stage2_encoder_comparison.run_encoder.main", inspect_dispatch)
    for job in jobs["stage2"]:
        run_stage2(job, "cpu")
    assert len(seen) == 6


def test_overrides_restore_after_failure():
    from experiments.topodise_experiments.stage0_corn1_pretrain import main
    with pytest.raises(RuntimeError):
        with training_overrides(main, "stage0"):
            raise RuntimeError("test")
    assert main.compute_topo_target is original_target
    assert main.CORN1Dataset is CORN1Dataset


@pytest.mark.parametrize("mode", ["hgrad_dual", "frangi"])
def test_stage1_targets_have_two_per_image_thresholds(mode):
    from experiments.topodise_experiments.stage1_rscp_finetune import main
    torch.manual_seed(17)
    x = torch.randn(2, 1, 64, 64)
    with training_overrides(main, "stage1"):
        target = main.compute_topo_target(x, mode=mode, out_size=32)
        altered = x.clone()
        altered[1] *= 1000
        changed_batch = main.compute_topo_target(altered, mode=mode, out_size=32)
    assert target.shape == (2, 2, 32, 32)
    assert torch.any(target[:, 0] != target[:, 1])
    assert torch.all(target[:, 1] <= target[:, 0])
    torch.testing.assert_close(target[0], changed_batch[0], rtol=0, atol=0)
    assert main.compute_topo_target is original_target


def test_summary_marks_partial_and_rejects_mixed_splits(tmp_path):
    from experiments.ablation_experiments.analyze import summarize
    jobs = batch.build_plan(tmp_path, tmp_path, tmp_path, tmp_path / "pd.json", tmp_path / "manifest.json", [42], backend="unified")
    write(tmp_path / "protocol.json", {"jobs": jobs})
    result = {"cohort_json_sha256": "cohort", "pair_manifest_sha256": "pairs",
              "split_manifest_sha256": "splits", "stage2_input_mode": "native_pair",
              "selected_epoch": 1, "inner_history": [{"epoch": 1, "inner_val_macro_f1": .5}],
              "outer_metrics": {"accuracy": .5, "balanced_accuracy": .5, "macro_f1": .5, "macro_auc_ovr": None}}
    path = Path(jobs["stage2"][0]["config"]["output_root"]) / "runs" / ENCODER / "pd_binary/seed42/fold0"
    write(path / "DONE.json", {"status": "complete"})
    write(path / "summary.json", result)
    summarize(tmp_path)
    summary = read(tmp_path / "analysis/summary.json")
    assert summary["complete"] is False
    assert summary["completed_folds"] == 1 and summary["missing_folds"] == 269
    path2 = Path(jobs["stage2"][1]["config"]["output_root"]) / "runs" / ENCODER / "pd_binary/seed42/fold0"
    write(path2 / "DONE.json", {"status": "complete"})
    write(path2 / "summary.json", {**result, "split_manifest_sha256": "different"})
    with pytest.raises(RuntimeError, match="different cohorts/pairs/splits"):
        summarize(tmp_path)
