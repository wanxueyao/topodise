# Unified Stage2 frozen-encoder comparison

This protocol changes only the frozen regional encoder while keeping cohort,
patient split, pairing, morphology input, Stage2 model and optimization fixed.
The configured encoders are TopoDiSE Stage1, data-matched SimCLR ResNet-18 and
official DINOv2-S/14.

## Inputs

The default config is `configs/protocol_v1.yaml`. Paths are relative to
`codes/`. TopoDiSE and SimCLR checkpoints are trained and cached automatically
when missing. TopoDiSE uses the paper's unified `c8bf6b548656` Stage0/1 recipe.

DINOv2 is externally pretrained and must be placed at:

```text
experiments/unified_stage2_encoder_comparison/pretrained/dinov2_vits14_pretrain.pth
```

The repository does not include this large file. Use the official DINOv2-S/14
weight compatible with the configured timm model and retain its SHA256.

Runtime values in the YAML are conservative defaults. Adjust worker counts,
feature batch size, pinning and training batch sizes for the target machine.
`device: auto` chooses CUDA, then MPS, then CPU. No host name, SSH, rsync, node
index or dual-machine directory convention is required.

## Audit and smoke test

```bash
cd codes
python -m experiments.unified_stage2_encoder_comparison.audit \
  --config experiments/unified_stage2_encoder_comparison/configs/protocol_v1.yaml

python -m experiments.unified_stage2_encoder_comparison.smoke_test \
  --config experiments/unified_stage2_encoder_comparison/configs/protocol_v1.yaml
```

The audit validates cohort counts, patient-only splits, pair manifests and
pretraining sources before GPU work starts.

## Run on one machine

Preview pending work:

```bash
python -m experiments.unified_stage2_encoder_comparison.run_matrix \
  --config experiments/unified_stage2_encoder_comparison/configs/protocol_v1.yaml \
  --dry-run
```

Run the full matrix:

```bash
python -m experiments.unified_stage2_encoder_comparison.run_matrix \
  --config experiments/unified_stage2_encoder_comparison/configs/protocol_v1.yaml
```

For a short resource check, add `--max-folds-per-encoder 1`. Select work with
`--encoders`, `--tasks`, `--split-seeds` or `--folds`. Each encoder is loaded
once and processes requested folds sequentially. Completed folds contain
`DONE.json` and are skipped on the next invocation.

A single fold can also be run directly:

```bash
python -m experiments.unified_stage2_encoder_comparison.run \
  --config experiments/unified_stage2_encoder_comparison/configs/protocol_v1.yaml \
  --encoder topodise_stage1 --task pd_binary --split-seed 42 --fold 0
```

## Monitor and analyze

```bash
python -m experiments.unified_stage2_encoder_comparison.status \
  --config experiments/unified_stage2_encoder_comparison/configs/protocol_v1.yaml

python -m experiments.unified_stage2_encoder_comparison.analyze \
  --config experiments/unified_stage2_encoder_comparison/configs/protocol_v1.yaml \
  --bootstrap 10000
```

The analyzer rejects duplicate patient predictions and mixed protocol hashes.
Outputs under `results/unified_stage2_encoder_comparison/` include the locked
config, cohort/split manifests, encoder metadata, checkpoint hashes, cached
features, fold predictions and aggregate statistics.
