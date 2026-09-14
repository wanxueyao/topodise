"""Run the original three paper Stage2 recipes on each ablation's own parent."""
from __future__ import annotations

import ast
from dataclasses import asdict
from pathlib import Path

from experiments._stage_configs import Stage2Config
from .protocol import CODES, file_hash, lock, read, require_complete, write

TASKS = ("pd_binary", "pd_stage", "pd_subtype_tdrest")
METHODS = {"pd_binary": "single_model", "pd_stage": "min_entropy_routing",
           "pd_subtype_tdrest": "majority_vote_tie_avg_prob"}


def paper_recipes():
    """Read the actual retained recipes without importing the batch scheduler."""
    path = CODES / "experiments/topodise_experiments/batch_experiments.py"
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "MAIN_TASKS_EXPERIMENTS" for t in node.targets
        ):
            recipes = ast.literal_eval(node.value)
            if [r["stage2_tasks"] for r in recipes] != [[t] for t in TASKS]:
                raise ValueError("Original paper tasks changed; review the full Stage2 adapter")
            return recipes
    raise RuntimeError("Cannot find original MAIN_TASKS_EXPERIMENTS")


def build_full_config(parent, output, pd_json, dataset_root, repeats=3, num_workers=0):
    recipes = []
    for original in paper_recipes():
        task = original["stage2_tasks"][0]
        candidates = []
        for leaf in original.get("oracle_experiments") or [original]:
            cfg = Stage2Config(**leaf["stage2_overrides"])
            cfg.parent_ckpt = str(parent)
            cfg.pd_json, cfg.base_path = str(pd_json), str(dataset_root)
            cfg.task = task
            cfg.output_dir = str(output / task / leaf["_group"])
            cfg.run_name = "full_stage2"
            cfg.num_workers = num_workers
            cfg.kfold_repeats = repeats
            cfg.deterministic = True
            candidates.append({"name": leaf["_group"], "config": asdict(cfg)})
        recipes.append({"task": task, "report_method": METHODS[task], "candidates": candidates})
    return {"parent_ckpt": str(parent), "output_root": str(output),
            "dataset_json": str(pd_json), "dataset_root": str(dataset_root),
            "tasks": list(TASKS), "recipes": recipes, "parent_weights": "encoder (raw)"}


def expected_folds(cfg):
    n = cfg["num_folds"]
    return 1 if n < 2 else min(n, cfg["eval_folds"] if cfg["eval_folds"] > 0 else n) * cfg["kfold_repeats"]


def prediction_index(result, cfg):
    """Fail on missing folds/patients; never aggregate a partial ensemble."""
    folds = result.get("fold_results", [])
    count = expected_folds(cfg)
    if result.get("error") or {int(f["fold"]) for f in folds} != set(range(1, count + 1)) or len(folds) != count:
        raise RuntimeError(f"Incomplete Stage2 folds for {cfg['task']}: expected {count}")
    rows = {}
    for fold in folds:
        predictions = fold.get("best_patient_predictions") or []
        if not predictions or len(predictions) != int(fold["n_val"]):
            raise RuntimeError(f"Missing patient predictions in {cfg['task']} fold {fold['fold']}")
        for p in predictions:
            key = (int(fold["fold"]), str(p["patient_id"]))
            if key in rows:
                raise RuntimeError(f"Duplicate patient prediction: {key}")
            rows[key] = p
    return rows


def patient_metrics(rows, classes):
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix
    y = [int(r["y_true"]) for r in rows]
    pred = [int(r["y_pred"]) for r in rows]
    # Majority vote can disagree with argmax(mean probabilities), so score the
    # actual reported class prediction rather than re-deriving it from probs.
    return {"accuracy": float(accuracy_score(y, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, labels=list(range(classes)), average="macro", zero_division=0)),
            "confusion_matrix": confusion_matrix(y, pred, labels=list(range(classes))).tolist(),
            "n": len(rows)}


def aggregate_task(recipe, results):
    from experiments.topodise_experiments.main import TopoDiSEExperiment

    task = recipe["task"]
    if len(results) != len(recipe["candidates"]):
        raise RuntimeError("Candidate count differs from the locked recipe")
    indices = [prediction_index(r, c["config"]) for c, r in zip(recipe["candidates"], results)]
    if any(set(index) != set(indices[0]) for index in indices):
        raise RuntimeError("Candidate folds/patients are not aligned")
    if any(int(index[k]["y_true"]) != int(indices[0][k]["y_true"])
           for index in indices for k in index):
        raise RuntimeError("Candidate patient labels are not aligned")
    method = recipe["report_method"]
    if method != METHODS[task]:
        raise ValueError("Do not select the report method by observed scores")
    if method == "single_model":
        rows = [{"fold": f, "patient_id": pid, "y_true": p["y_true"], "y_pred": p["y_pred"],
                 "ensemble_probs": p.get("patient_probs"), "method": method}
                for (f, pid), p in sorted(indices[0].items())]
    else:
        children = [{"_group": c["name"], "stage2_results": {task: r}}
                    for c, r in zip(recipe["candidates"], results)]
        ensemble = TopoDiSEExperiment({})._aggregate_patient_logit_ensembles(children, task)
        # Keep exactly the predefined paper rule; never use best_practical or
        # the original controller's label-selected oracle result.
        rows = ensemble.get("predictions_by_method", {}).get(method, [])
    if {(int(r["fold"]), str(r["patient_id"])) for r in rows} != set(indices[0]):
        raise RuntimeError("Incomplete predictions for the fixed report method")
    classes = 3 if task == "pd_stage" else 2
    folds = [{"fold": fold, **patient_metrics([r for r in rows if r["fold"] == fold], classes)}
             for fold in sorted({r["fold"] for r in rows})]
    return {"task": task, "report_method": method, "predictions": rows,
            "fold_metrics": folds, "pooled_metrics": patient_metrics(rows, classes),
            "candidate_names": [c["name"] for c in recipe["candidates"]],
            "evaluation": "original Stage2 repeated patient CV; each validation fold also selects its best epoch"}


def run_full_stage2(job, device):
    from experiments.topodise_experiments.stage2_racf_downstream.main import run_single_experiment

    values = job["config"]
    parent = require_complete(Path(values["parent_ckpt"]).parent.parent)
    output = Path(values["output_root"])
    contract = {"job": job, "checkpoint_sha256": file_hash(parent)}
    lock(output / "ablation_job.json", contract)
    all_tasks = []
    for recipe in values["recipes"]:
        results = []
        for candidate in recipe["candidates"]:
            cfg = Stage2Config(**candidate["config"])
            cfg.device = device
            directory = Path(cfg.output_dir)
            lock(directory / "ablation_config.json", candidate["config"])
            done_path, result_path = directory / "DONE.json", directory / "result.json"
            if done_path.exists():
                if read(done_path)["result_sha256"] != file_hash(result_path):
                    raise RuntimeError(f"Completed candidate result changed: {result_path}")
                result = read(result_path)
                print(f"[skip full Stage2] {candidate['name']}", flush=True)
            else:
                print(f"[full Stage2] {recipe['task']} / {candidate['name']}", flush=True)
                # The original runner resumes completed folds itself. An
                # interrupted epoch restarts only its unfinished fold.
                result = run_single_experiment(cfg)
                prediction_index(result, candidate["config"])
                write(result_path, result)
                write(done_path, {"status": "complete", "result_sha256": file_hash(result_path)})
            results.append(result)
        summary = aggregate_task(recipe, results)
        write(output / recipe["task"] / "task_summary.json", summary)
        all_tasks.append(summary)
    write(output / "full_stage2_summary.json", {"tasks": all_tasks, "parent_checkpoint_sha256": file_hash(parent)})
    write(output / "DONE.json", {"status": "complete", "tasks": values["tasks"]})


def summarize_full(root, protocol):
    import csv
    from collections import defaultdict
    from statistics import mean, stdev
    rows, pending, aligned = [], [], {}
    for job in protocol["jobs"]["stage2"]:
        cfg = job["config"]
        output = Path(cfg["output_root"])
        for recipe in cfg["recipes"]:
            path = output / recipe["task"] / "task_summary.json"
            if not path.exists():
                pending.append(str(path))
                continue
            # Ignore a stale aggregate unless every underlying candidate is complete.
            results = []
            for candidate in recipe["candidates"]:
                directory = Path(candidate["config"]["output_dir"])
                if not (directory / "DONE.json").exists():
                    pending.append(str(path))
                    break
                if read(directory / "DONE.json")["result_sha256"] != file_hash(directory / "result.json"):
                    raise RuntimeError(f"Candidate result changed: {directory}")
                results.append(read(directory / "result.json"))
            else:
                summary = aggregate_task(recipe, results)
                for f in summary["fold_metrics"]:
                    key = (recipe["task"], f["fold"])
                    cohort = sorted((str(r["patient_id"]), int(r["y_true"])) for r in summary["predictions"] if r["fold"] == f["fold"])
                    if aligned.setdefault(key, cohort) != cohort:
                        raise RuntimeError(f"Patient folds differ across ablations: {key}")
                    rows.append({"variant": job["variant"], "pretrain_seed": job["pretrain_seed"],
                                 "task": recipe["task"], "method": summary["report_method"],
                                 **{k: f[k] for k in ("fold", "accuracy", "balanced_accuracy", "macro_f1", "n")}})
    groups = defaultdict(list)
    for row in rows:
        groups[(row["variant"], row["task"])].append(row)
    aggregates = []
    for (variant, task), items in groups.items():
        a = {"variant": variant, "task": task, "completed_folds": len(items)}
        for metric in ("accuracy", "balanced_accuracy", "macro_f1"):
            v = [r[metric] for r in items]
            a[metric + "_mean"] = mean(v)
            a[metric + "_std"] = stdev(v) if len(v) > 1 else None
        aggregates.append(a)
    write(root / "analysis/summary.json", {"backend": "full", "complete": not pending,
        "completed_task_groups": len(protocol["jobs"]["stage2"]) * 3 - len(pending),
        "pending": pending, "aggregates": aggregates,
        "evaluation": "original Stage2 validation-selected repeated CV; not Unified nested outer-test estimates"})
    with (root / "analysis/fold_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["variant", "task", "fold", "accuracy", "macro_f1"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"[full Stage2 analysis] completed={len(rows)} reported folds, pending={len(pending)} task groups", flush=True)
