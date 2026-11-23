# Reproducing the TopoDiSE ablations

[中文说明](README.zh-CN.md) · [Data and installation](../README.md)

| Result | Comparison | Evaluation |
|---|---|---|
| Table 3, supplement E | Stage0 × Stage1 targets | Complete PD system, pretraining seed 42, 15-fold means |
| Table 4, supplement F | A06, Stage0 mask loss on/off | Frozen linear probes, paired per-image AP at seeds 42, 43, 44 |
| Supplement C | Individual candidates and fixed inference | Same candidate predictions as Table 1 |
| Supplement E | M0, A01–A05, C03/C04, selected D/E components | US0, U1 and TDREST5 separately, three repeats per pretraining seed |
| Supplement E | D11, D12, E06 | Replay of the selected U1 checkpoints |
| Supplement F | H05 | Stage1 versus Stage0 probe, seed 42 |

## Target comparison

Run at the repository root. The two variants below hold Stage0 GT fixed. Omit `--variants` to run the full six-cell matrix. All routes use the main system's original downstream recipes. The archived comparator uses `frangi`; `hgrad_dual` is the paper's HGrad.

```bash
python -m ablation_experiments.paper_targets --dry-run
python -m ablation_experiments.paper_targets --dataset-root /path/to/datasets --variants gt_frangi gt_hgrad_dual --devices cuda:0
```

Outputs are under `ablation_experiments/results/paper_pd_targets/`. Read `accuracy_mean` and `macro_f1_mean` in `analysis/summary.json`. To inspect a candidate or apply the average-probability and fixed rules, use the F01/F02 analysis on all three H&Y or all five subtype candidates. Its JSON bundle contains `task` and `candidates`; each candidate has `name`, `config` from the Stage2 recipe, and `result` (a loaded result object or its JSON path). For the published values, average the metrics of the individual folds.

```bash
python -m ablation_experiments.analyze --experiment F01 --bundle /path/to/candidates.json --output inference.json
```

## Candidate and probe training

```bash
python -m ablation_experiments.run --experiments paper --tasks pd_binary pd_stage pd_subtype_tdrest \
  --pretrain-seeds 42 43 44 --repeats 3 --unit CANDIDATE --dataset-root /path/to/datasets \
  --output /path/to/paper-components --dry-run
```

Replace `--dry-run` with `--execute --devices cuda:0` to train. Use `--only A06` with the same plan to run just the Stage0 mask comparison and its dependencies. C03/C04 restrict all three regional inputs to the same paired cohort (M0.paired). Each other component uses its matched M0 candidate. The reported metric pools the five out-of-fold sets within each repeat, then averages over repeats and seeds. The full exploratory registry is available through `--experiments all`; the paper subset is sufficient for the training rows above.

For D11, D12 and E06, pass the 15 selected U1 fold checkpoints **from one pretraining seed per invocation**, including the saved queries and support bank. Repeat at seeds 42, 43 and 44. Example with the checkpoint paths in a Bash array named `fold_checkpoints`:

```bash
python -m ablation_experiments.analyze --experiment D11 --checkpoints "${fold_checkpoints[@]}" --output classifier_seed42.json
```

## Frozen CORN-1 probe

Generate the split manifest locally from CORN-1. Its canonical folders define the probe partitions. Identical image files are retained once, with training, validation and test precedence. For each seed, extract and fit the probe twice, once per A06 checkpoint listed in the plan. Keep separate output directories for GT off and GT on.

```bash
python -m ablation_experiments.H01_H06_structure_diagnostics.build_manifest --root /path/to/datasets/CORN-1 --output results/probe_split.json
python -m ablation_experiments.H01_H06_structure_diagnostics.extract \
  --manifest results/probe_split.json \
  --checkpoint /path/to/gt_off_seed42.pt --output off/features.npz --protocol STR-A --device cuda:0
python -m ablation_experiments.analyze --experiment H02 --features off/features.npz --output off/probe
# Repeat the two commands for the GT-on checkpoint, writing to on/.
python -m ablation_experiments.analyze --experiment H05 --reference off/probe/predictions.npz \
  --variants on/probe/predictions.npz --output gt_on_minus_off.json
```

For the Stage0-to-Stage1 comparison, use the GT-trained Stage0 reference and the full Stage1 encoder at seed 42 with the same split. H03 fits the skeleton-descriptor probe on the same features (use the H02 command with `--experiment H03`). The separate common Core Stage2 encoder comparison in supplement D uses the [historical shared-readout implementation](https://github.com/wanxueyao/topodise/tree/e7dc1791dfb7457e1faf209e18fdfeeb6c7f7b67/experiments/unified_stage2_encoder_comparison).
