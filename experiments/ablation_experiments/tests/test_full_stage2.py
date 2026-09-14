from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import pytest
import torch
from PIL import Image

from experiments.ablation_experiments.batch_experiments import build_plan
from experiments.ablation_experiments.full_stage2 import (
    paper_recipes, aggregate_task, patient_metrics, run_full_stage2,
)
from experiments.ablation_experiments.protocol import file_hash, read, write
from experiments.ablation_experiments.worker import run_stage2


def plan(tmp_path):
    return build_plan(tmp_path / "results", tmp_path, tmp_path / "corn",
                      tmp_path / "pd.json", tmp_path / "manifest.json", [42])


def mark_parent(job):
    path = Path(job["config"]["parent_ckpt"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"synthetic test parent")
    write(path.parent.parent / "ABLATION_DONE.json", {"checkpoint_sha256": file_hash(path)})
    return path


def fake_result(cfg):
    folds = []
    for fold in range(1, cfg.num_folds * cfg.kfold_repeats + 1):
        classes = 3 if cfg.task == "pd_stage" else 2
        predictions = [{"patient_id": f"{fold}-{y}", "y_true": y, "y_pred": y,
                        "patient_probs": [0.9 if i == y else .1 / (classes - 1) for i in range(classes)]}
                       for y in range(classes)]
        folds.append({"fold": fold, "n_val": classes, "best_patient_acc": 1.0,
                      "best_patient_predictions": predictions})
    return {"fold_results": folds}


def test_default_plan_matches_all_original_paper_candidates(tmp_path):
    jobs = plan(tmp_path)
    assert [len(jobs[s]) for s in jobs] == [3, 6, 6]
    assert all(j["backend"] == "full" for j in jobs["stage2"])
    original = paper_recipes()
    directories = []
    for job in jobs["stage2"]:
        assert [len(r["candidates"]) for r in job["config"]["recipes"]] == [1, 3, 5]
        for recipe, source in zip(job["config"]["recipes"], original):
            for c, old in zip(recipe["candidates"], source.get("oracle_experiments") or [source]):
                for key, value in old["stage2_overrides"].items():
                    if key not in ("pd_json", "base_path", "num_workers"):
                        assert c["config"][key] == value, key
                assert c["config"]["parent_ckpt"] == job["config"]["parent_ckpt"]
                assert c["config"]["kfold_repeats"] == 3
                assert c["config"]["eval_folds"] == 5
                directories.append(c["config"]["output_dir"])
    assert len(set(directories)) == 54


def test_six_routes_dispatch_full_stage2_and_resume(tmp_path, monkeypatch):
    seen = []
    def train(cfg):
        seen.append(cfg)
        return fake_result(cfg)
    monkeypatch.setattr("experiments.topodise_experiments.stage2_racf_downstream.main.run_single_experiment", train)
    jobs = plan(tmp_path)
    for job in jobs["stage2"]:
        mark_parent(job)
        run_stage2(job, "cpu")
        run_stage2(job, "cpu")
        summary = read(Path(job["config"]["output_root"]) / "full_stage2_summary.json")
        assert [r["report_method"] for r in summary["tasks"]] == ["single_model", "min_entropy_routing", "majority_vote_tie_avg_prob"]
    assert len(seen) == 54  # Each candidate runs once; second invocation skips.
    assert len({cfg.parent_ckpt for cfg in seen}) == 6
    assert all(cfg.device == "cpu" for cfg in seen)


def test_incomplete_candidate_cannot_finish_ensemble(tmp_path, monkeypatch):
    job = plan(tmp_path)["stage2"][0]
    mark_parent(job)
    def incomplete(cfg):
        result = fake_result(cfg)
        result["fold_results"].pop()
        return result
    monkeypatch.setattr("experiments.topodise_experiments.stage2_racf_downstream.main.run_single_experiment", incomplete)
    with pytest.raises(RuntimeError, match="Incomplete Stage2"):
        run_stage2(job, "cpu")
    assert not (Path(job["config"]["output_root"]) / "DONE.json").exists()


def test_fixed_vote_and_entropy_rules(tmp_path):
    from experiments._stage_configs import Stage2Config
    job = plan(tmp_path)["stage2"][0]
    vote = job["config"]["recipes"][2]
    for candidate in vote["candidates"]:
        candidate["config"].update(num_folds=2, eval_folds=2, kfold_repeats=1)
    results = [fake_result(Stage2Config(**c["config"])) for c in vote["candidates"]]
    # Three low-confidence TD votes beat two highly confident Rest votes.
    for index, result in enumerate(results):
        for f in result["fold_results"]:
            for p in f["best_patient_predictions"]:
                p["y_pred"] = 0 if index < 3 else 1
                p["patient_probs"] = [.51, .49] if index < 3 else [.01, .99]
    summary = aggregate_task(vote, results)
    assert all(p["y_pred"] == 0 and p["ensemble_probs"][1] > .5 for p in summary["predictions"])
    assert summary["pooled_metrics"]["confusion_matrix"] == [[2, 0], [2, 0]]
    broken = deepcopy(results)
    broken[0]["fold_results"][0]["best_patient_predictions"][0]["patient_id"] = "different"
    with pytest.raises(RuntimeError, match="not aligned"):
        aggregate_task(vote, broken)
    stage = job["config"]["recipes"][1]
    for c in stage["candidates"]:
        c["config"].update(num_folds=2, eval_folds=2, kfold_repeats=1)
    stage_results = [fake_result(Stage2Config(**c["config"])) for c in stage["candidates"]]
    for i, result in enumerate(stage_results):
        for f in result["fold_results"]:
            for p in f["best_patient_predictions"]:
                p["patient_probs"] = [.01, .98, .01] if i == 1 else [.4, .3, .3]
                p["y_pred"] = 1 if i == 1 else 0
    routed = aggregate_task(stage, stage_results)
    assert all(p["y_pred"] == 1 for p in routed["predictions"])


def test_full_summary_and_reuse_validation(tmp_path, monkeypatch):
    from experiments.ablation_experiments.analyze import summarize
    from experiments.ablation_experiments.reuse_pretrain import reuse_parents
    from experiments.ablation_experiments.protocol import checkpoint
    monkeypatch.setattr("experiments.topodise_experiments.stage2_racf_downstream.main.run_single_experiment", fake_result)
    jobs = plan(tmp_path)
    root = tmp_path / "results"
    write(root / "protocol.json", {"jobs": jobs, "stage2_backend": "full"})
    for job in jobs["stage2"]:
        mark_parent(job)
        run_full_stage2(job, "cpu")
    summarize(root)
    assert read(root / "analysis/summary.json")["complete"] is True
    first = Path(jobs["stage2"][0]["config"]["output_root"])
    (first / "pd_stage/task_summary.json").unlink()
    summarize(root)
    assert len(read(root / "analysis/summary.json")["pending"]) == 1
    # Create completed old Stage0/1 records, then link their checkpoint paths
    # into a new full-Stage2 protocol without copying or retraining weights.
    for job in jobs["stage0"] + jobs["stage1"]:
        cfg = job["config"]
        directory = Path(cfg["output_dir"]) / cfg["run_name"]
        path = checkpoint(directory)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(job["id"].encode())
        done = {"checkpoint_sha256": file_hash(path)}
        if job["stage"] == "stage1":
            done["parent_checkpoint_sha256"] = file_hash(cfg["parent_ckpt"])
        write(directory / "ABLATION_DONE.json", done)
        write(directory / "ablation_job.json", {"job": job})
    fresh = build_plan(tmp_path / "fresh", tmp_path, tmp_path / "corn", tmp_path / "pd.json",
                       tmp_path / "manifest.json", [42])
    reuse_parents(fresh, root, [j["variant"] for j in fresh["stage2"]])
    for job in fresh["stage2"]:
        assert str(root) in job["config"]["parent_ckpt"]
        assert all(c["config"]["parent_ckpt"] == job["config"]["parent_ckpt"]
                   for r in job["config"]["recipes"] for c in r["candidates"])
    bad = Path(jobs["stage1"][0]["config"]["output_dir"]) / jobs["stage1"][0]["config"]["run_name"] / "ablation_job.json"
    changed = read(bad)
    changed["job"]["config"]["topo_target"] = "hgrad"
    write(bad, changed)
    with pytest.raises(RuntimeError, match="Stage1 pretraining recipe differs"):
        reuse_parents(fresh, root, [j["variant"] for j in fresh["stage2"]])


def test_real_full_stage2_all_three_tasks(tmp_path):
    from experiments._stage_configs import Stage0Config
    from topodise.encoder import EncoderConfig, TopoDiSEEncoder
    from topodise.unet import UNetConfig, ConditionalUNet
    from topodise.topo_head import TopoHead
    from experiments.topodise_experiments.stage2_racf_downstream.main import _build_from_ckpt
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        job = plan(tmp_path)["stage2"][0]
        parent = mark_parent(job)
        cfg = Stage0Config(image_size=32, encoder_base_channels=8, encoder_channel_mult=(1, 1),
                           encoder_blocks_per_stage=(1, 1), z_dim=16,
                           unet_model_channels=8, unet_channel_mult=(1, 1), T=8)
        encoder = TopoDiSEEncoder(EncoderConfig(base_channels=8, channel_mult=(1, 1), blocks_per_stage=(1, 1), z_dim=16))
        unet = ConditionalUNet(UNetConfig(model_channels=8, channel_mult=(1, 1), z_dim=16, image_size=32))
        raw = encoder.state_dict()
        torch.save({"parent_config": asdict(cfg), "config": {}, "encoder": raw,
                    "ema_encoder": {k: torch.zeros_like(v) for k, v in raw.items()},
                    "unet": unet.state_dict(), "topo_head": TopoHead(z_dim=16).state_dict()}, parent)
        write(parent.parent.parent / "ABLATION_DONE.json", {"checkpoint_sha256": file_hash(parent)})
        loaded, *_ = _build_from_ckpt(str(parent), torch.device("cpu"))
        assert all(torch.equal(v, loaded.state_dict()[k]) for k, v in raw.items())
        rng = np.random.default_rng(13)
        patients = []
        for i in range(24):
            paths = {}
            for region in ("central", "whorl"):
                path = tmp_path / f"image_{i}_{region}.png"
                Image.fromarray(rng.integers(0, 256, (36, 36), dtype=np.uint8)).save(path)
                paths[region] = [str(path)]
            patients.append({"patient_id": i, "group": "control" if i < 6 else "pd",
                             "stage": (i % 3) + 1, "subtype": ["TD", "PIGD", "mixed"][i % 3],
                             "image_paths": paths})
        write(tmp_path / "pd.json", patients)
        for recipe in job["config"]["recipes"]:
            for candidate in recipe["candidates"]:
                candidate["config"].update(epochs=4, image_size=32, batch_size=2, num_workers=0,
                                           num_folds=2, eval_folds=2, kfold_repeats=1,
                                           adapter_rank=2, proto_dim=8, proto_min_per_class=1)
        run_full_stage2(job, "cpu")
        summary = read(Path(job["config"]["output_root"]) / "full_stage2_summary.json")
        assert len(summary["tasks"]) == 3
        assert all(len(t["fold_metrics"]) == 2 for t in summary["tasks"])
        assert all(0 <= t["pooled_metrics"]["macro_f1"] <= 1 for t in summary["tasks"])
        assert (Path(job["config"]["output_root"]) / "DONE.json").exists()
    finally:
        torch.set_num_threads(old_threads)
