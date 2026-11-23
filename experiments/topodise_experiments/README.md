# TopoDiSE training and patient inference

[中文说明](README.zh-CN.md) · [Installation and data layout](../../README.md)

Stage0 uses diffusion noise prediction and mask-derived presence and dense-core grids on CORN-1. Stage1 pools horizontal absolute intensity differences to a 32×32 grid and thresholds each image at its mean and at its mean plus one standard deviation. Gradients pass through the frozen topology head to the trainable encoder parameters. Stage2 loads the raw Stage1 encoder, trains residual adapters and patient readouts, and also fine-tunes fusion and projection for diagnosis and H&Y.

| Task | Candidates | Patient prediction |
|---|---|---|
| PD diagnosis | US0 | Single classifier |
| H&Y grouping | U0, U1, U2 | Select the lowest-entropy probability vector |
| TD/non-TD | TDREST0, 1, 2, 5, 7 | Majority vote, with mean probability as tie-break |
| DM transfer | DMMIL3F1 | Single classifier |

The candidates train independently. See [recipes.py](recipes.py) for all settings. The archived DM experiment in Table 2 used an earlier Frangi Stage1 checkpoint. Running the current DM entry evaluates transfer from GT+HGrad.

```bash
python -m experiments.topodise_experiments.batch_experiments --section main_tasks --dry-run
python -m experiments.topodise_experiments.batch_experiments --section main_tasks --dataset-root /path/to/datasets --devices cuda:0
```

Use `--phase prepare` to audit inputs and folds, `--phase stage2` to continue after pretraining, and `--phase analyze` to refresh summaries for a completed batch. Use the same batch name and options to resume. Choose a new `--name` when changing the protocol. Outputs default to `experiments/topodise_experiments/results/gt_hgrad_dual_full_v1/`.

The PD paper tables average the 15 fold metrics with equal weight. The descriptive half-width is 1.96 × sample standard deviation / √15. In `analysis/summary.json`, these means are under `aggregates`. The separate `analysis/metrics_ci95.csv` report computes metrics on complete out-of-fold predictions within each repeat and resamples patients across repeats.
