"""Run the four fixed full Stage2 tasks and export patient-level 95% CIs."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from experiments._stage_configs import Stage2Config
from .protocol import file_hash, lock, read, require_complete, write

TASKS = ("pd_binary", "pd_stage", "pd_subtype_tdrest", "dm_binary")
METHODS = {"pd_binary": "single_model", "pd_stage": "min_entropy_routing",
           "pd_subtype_tdrest": "majority_vote_tie_avg_prob", "dm_binary": "single_model"}


def paper_recipes():
    from experiments.topodise_experiments.recipes import MAIN_TASKS_EXPERIMENTS, DM_TRANSFER_EXPERIMENTS
    return MAIN_TASKS_EXPERIMENTS + DM_TRANSFER_EXPERIMENTS


def build_full_config(parent, output, pd_json, dataset_root, repeats=3, num_workers=0,
                      tasks=TASKS, dm_json=None, bootstrap_samples=2000, bootstrap_seed=20260915,
                      encoder_backend="topodise", encoder_checkpoint="", encoder_model_name="vit_small_patch14_dinov2"):
    if not tasks or set(tasks) - set(TASKS):
        raise ValueError("Unknown or empty task selection")
    recipes = []
    for original in paper_recipes():
        task = original["stage2_tasks"][0]
        if task not in tasks:
            continue
        candidates = []
        for leaf in original.get("oracle_experiments") or [original]:
            cfg = Stage2Config(**leaf["stage2_overrides"])
            cfg.parent_ckpt = str(parent)
            cfg.pd_json = str((dm_json or dataset_root / "dm_dataset_en/dm_patients_dataset_en.json") if task == "dm_binary" else pd_json)
            cfg.base_path, cfg.task = str(dataset_root), task
            cfg.output_dir = str(output / task / leaf["_group"])
            cfg.run_name, cfg.num_workers = "full_stage2", num_workers
            cfg.kfold_repeats, cfg.deterministic = repeats, True
            cfg.encoder_backend, cfg.encoder_checkpoint = encoder_backend, str(encoder_checkpoint)
            cfg.encoder_model_name = encoder_model_name
            candidates.append({"name": leaf["_group"], "config": asdict(cfg)})
        recipes.append({"task": task, "report_method": METHODS[task], "candidates": candidates,
                        "bootstrap_samples": bootstrap_samples, "bootstrap_seed": bootstrap_seed})
    return {"parent_ckpt": str(parent), "output_root": str(output),
            "dataset_json": str(pd_json), "dataset_root": str(dataset_root),
            "tasks": list(tasks), "recipes": recipes, "encoder_backend": encoder_backend,
            "encoder_checkpoint": str(encoder_checkpoint), "parent_weights": "encoder (raw)"}


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
    from utils.confidence_intervals import scores
    from sklearn.metrics import confusion_matrix
    import math
    y, pred = [int(r["y_true"]) for r in rows], [int(r["y_pred"]) for r in rows]
    metrics = scores(y, pred, [r.get("ensemble_probs", r.get("patient_probs")) for r in rows], classes)
    metrics = {k: v if math.isfinite(v) else None for k, v in metrics.items()}
    return {**metrics, "balanced_accuracy": metrics["bacc"],
            "confusion_matrix": confusion_matrix(y, pred, labels=list(range(classes))).tolist(), "n": len(rows)}


def aggregate_task(recipe, results):
    from experiments.topodise_experiments.main import fixed_patient_ensemble

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
        rows = fixed_patient_ensemble(indices, [c["name"] for c in recipe["candidates"]], method)
    if {(int(r["fold"]), str(r["patient_id"])) for r in rows} != set(indices[0]):
        raise RuntimeError("Incomplete predictions for the fixed report method")
    classes = 3 if task == "pd_stage" else 2
    folds = [{"fold": fold, **patient_metrics([r for r in rows if r["fold"] == fold], classes)}
             for fold in sorted({r["fold"] for r in rows})]
    from utils.confidence_intervals import report
    statistics = report(rows, classes, recipe["candidates"][0]["config"]["num_folds"],
                        recipe.get("bootstrap_samples", 2000), recipe.get("bootstrap_seed", 20260915))
    return {"task": task, "report_method": method, "predictions": rows, "statistics": statistics,
            "fold_metrics": folds, "pooled_metrics": patient_metrics(rows, classes),
            "candidate_names": [c["name"] for c in recipe["candidates"]],
            "evaluation": "original Stage2 repeated patient CV; each validation fold also selects its best epoch"}


def run_full_stage2(job, device):
    from topodise.training.stage2 import run_single_experiment

    values = job["config"]
    parent = (require_complete(Path(values["parent_ckpt"]).parent.parent)
              if values.get("encoder_backend", "topodise") == "topodise"
              else Path(values["encoder_checkpoint"]))
    output = Path(values["output_root"])
    contract = {"job": job, "checkpoint_sha256": file_hash(parent)}
    lock(output / "training_job.json", contract)
    all_tasks = []
    for recipe in values["recipes"]:
        results = []
        for candidate in recipe["candidates"]:
            cfg = Stage2Config(**candidate["config"])
            cfg.device = device
            directory = Path(cfg.output_dir)
            lock(directory / "training_config.json", candidate["config"])
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
        write_task_reports(output / recipe["task"], summary)
        all_tasks.append(summary)
    write(output / "full_stage2_summary.json", {"tasks": all_tasks, "parent_checkpoint_sha256": file_hash(parent)})
    write(output / "DONE.json", {"status": "complete", "tasks": values["tasks"]})


def summarize_full(root, protocol):
    import csv
    from collections import defaultdict
    from statistics import mean, stdev
    rows, pending, aligned, task_reports = [], [], {}, []
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
                write(path, summary)
                write_task_reports(path.parent, summary)
                task_reports.append({"variant": job["variant"], "pretrain_seed": job["pretrain_seed"],
                                     "task": recipe["task"], **summary["statistics"]})
                for f in summary["fold_metrics"]:
                    key = (recipe["task"], job["pretrain_seed"], f["fold"])
                    cohort = sorted((str(r["patient_id"]), int(r["y_true"])) for r in summary["predictions"] if r["fold"] == f["fold"])
                    if aligned.setdefault(key, cohort) != cohort:
                        raise RuntimeError(f"Patient folds differ across ablations: {key}")
                    rows.append({"variant": job["variant"], "pretrain_seed": job["pretrain_seed"],
                                 "task": recipe["task"], "method": summary["report_method"],
                                 **{k: f[k] for k in ("fold", "accuracy", "bacc", "balanced_accuracy", "macro_f1", "auc", "n")}})
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
        "completed_task_groups": sum(len(j["config"]["recipes"]) for j in protocol["jobs"]["stage2"]) - len(pending),
        "pending": pending, "aggregates": aggregates, "task_reports": task_reports,
        "evaluation": "original Stage2 validation-selected repeated CV; not Unified nested outer-test estimates"})
    with (root / "analysis/fold_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["variant", "task", "fold", "accuracy", "macro_f1"])
        writer.writeheader()
        writer.writerows(rows)
    write_summary_table(root / "analysis", task_reports)
    print(f"[full Stage2 analysis] completed={len(rows)} reported folds, pending={len(pending)} task groups", flush=True)


def write_summary_table(directory, reports):
    import csv
    from utils.confidence_intervals import METRICS
    directory.mkdir(parents=True, exist_ok=True)
    columns = ["variant", "pretrain_seed", "task", "n_patients", "n_repeats"]
    columns += [f"{m}_{k}" for m in METRICS for k in ("estimate", "ci95_low", "ci95_high")]
    with (directory / "metrics_ci95.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns); writer.writeheader()
        for report in reports:
            row = {k: report.get(k) for k in columns[:5]}
            for m in METRICS:
                row.update({f"{m}_{k}": report["metrics"][m][k] for k in ("estimate", "ci95_low", "ci95_high")})
            writer.writerow(row)


def write_task_reports(directory, summary):
    stats = summary["statistics"]
    write_summary_table(directory, [{"task": summary["task"], **stats}])
    lines = ["| Metric | Estimate | 95% CI |", "|---|---:|---:|"]
    for metric, value in stats["metrics"].items():
        estimate, lo, hi = (value[k] for k in ("estimate", "ci95_low", "ci95_high"))
        lines.append(f"| {metric} | {estimate:.4f} | [{lo:.4f}, {hi:.4f}] |" if lo is not None else f"| {metric} | undefined | undefined |")
    (directory / "metrics_ci95.md").write_text("\n".join(lines) + "\n\n" + stats["ci_method"] + "\n" + stats["limitation"] + "\n", encoding="utf-8")
    print(f"[metrics] {summary['task']} " + "; ".join(lines[2:]), flush=True)
