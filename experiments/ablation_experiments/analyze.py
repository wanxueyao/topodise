"""Cross-route fold summary; never choose a method using outer-test scores."""
import csv
from collections import defaultdict
from statistics import mean, stdev
from pathlib import Path

from .protocol import ENCODER, ROUTES, read, write


def summarize(root):
    protocol = read(root / "protocol.json")
    rows, missing = [], []
    common = {}
    for job in protocol["jobs"]["stage2"]:
        cfg = job["config"]
        output = Path(cfg["output_root"])
        variant = output.name
        pretrain_seed = int(output.parent.parent.name.removeprefix("pretrain_seed"))
        for task in cfg["tasks"]:
            for seed in cfg["splits"]["seeds"]:
                for fold in range(cfg["splits"]["outer_folds"]):
                    path = output / "runs" / ENCODER / task / f"seed{seed}" / f"fold{fold}"
                    if not (path / "DONE.json").is_file():
                        missing.append(str(path))
                        continue
                    summary = read(path / "summary.json")
                    key = (task, seed, fold)
                    signature = tuple(summary[k] for k in (
                        "cohort_json_sha256", "pair_manifest_sha256", "split_manifest_sha256"))
                    if common.setdefault(key, signature) != signature:
                        raise RuntimeError(f"Cannot compare routes with different cohorts/pairs/splits: {path}")
                    if summary["stage2_input_mode"] != "native_pair":
                        raise RuntimeError(f"Unexpected Stage2 input mode: {path}")
                    # Inner selection score is reported separately. An overall
                    # method-selection protocol must be fixed before test use.
                    selected = int(summary["selected_epoch"])
                    history = summary["inner_history"]
                    inner = next(r["inner_val_macro_f1"] for r in history if int(r["epoch"]) == selected)
                    metrics = summary["outer_metrics"]
                    rows.append({"variant": variant, "stage0": ROUTES[variant][0],
                                 "stage1": ROUTES[variant][1], "pretrain_seed": pretrain_seed,
                                 "task": task, "split_seed": seed, "fold": fold,
                                 "selected_epoch": selected, "inner_val_macro_f1": inner,
                                 **{k: metrics[k] for k in ("accuracy", "balanced_accuracy", "macro_f1", "macro_auc_ovr")}})
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["variant"], row["task"])].append(row)
    aggregates = []
    for (variant, task), group in grouped.items():
        aggregate = {"variant": variant, "task": task, "completed_folds": len(group)}
        for metric in ("inner_val_macro_f1", "accuracy", "balanced_accuracy", "macro_f1", "macro_auc_ovr"):
            values = [r[metric] for r in group if r[metric] is not None]
            aggregate[metric + "_mean"] = mean(values) if values else None
            aggregate[metric + "_std"] = stdev(values) if len(values) > 1 else None
        aggregates.append(aggregate)
    destination = root / "analysis"
    write(destination / "summary.json", {
        "complete": not missing, "completed_folds": len(rows), "missing_folds": len(missing),
        "aggregation": "unweighted fold mean and sample std; overlapping CV folds are not independent repeats",
        "method_selection": "No automatic outer-test ranking or winner selection",
        "aggregates": aggregates, "pending": missing,
    })
    # Write header even before any fold has run; overwrite stale summaries.
    fields = list(rows[0]) if rows else ["variant", "task", "pretrain_seed", "split_seed", "fold"]
    with (destination / "fold_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[analysis] complete={not missing}, completed={len(rows)}, pending={len(missing)}; {destination}")
