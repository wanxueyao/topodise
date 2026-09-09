# Reproducing the c8bf6b548656 Stage0/Stage1 checkpoint

Edit the `EXPERIMENTS` configuration in each stage's `batch_experiments.py`.
The defaults retain the recorded c8bf6b548656 recipe: 200 epochs for Stage0,
80 epochs for Stage1, batch size 16 and 4 data workers. `main.py` implements
training and receives the resolved configuration from the batch controller.

From `codes`:

```bash
python -m experiments.topodise_experiments.stage0_corn1_pretrain.batch_experiments
# Set config.parent_ckpt in Stage1's batch file to Stage0's actual ema_best.pt.
python -m experiments.topodise_experiments.stage1_rscp_finetune.batch_experiments
```

The batch file controls GPUs, experiment selection and output paths. Each run
saves its configuration in the output directory. To resume a previous standalone
stage, set `config.resume_ckpt` to its checkpoint. The combined TopoDiSE batch
controller also supports automatic matching and resuming of earlier batches.

Set `config.reproducibility=False` to disable the initialization call to
`torch.manual_seed(42)`. This switch does not enable deterministic kernels or
set Python/NumPy seeds. Checkpoints save and restore PyTorch CPU/CUDA RNG states.
The stage entry points validate their input data and sample counts; they do not
enforce the historical dataset SHA256 fingerprints.

Reference facts recovered from the saved run:

- Source code snapshot: Git commit `6e688f2` (the run began about two minutes
  after that commit).
- Stage0: 1,516 CORN-1 images, 224 px, 200 epochs, batch 16, 4 workers, seed 42;
  best loss `0.11062798957201711` at epoch 100.
- Stage1: 182 paired control/control-cataract samples, 224 px, 80 epochs,
  batch 16, 4 workers, seed 42; best loss `0.02590857886455276` at epoch 77.
- Stage1 initializes from Stage0's raw `encoder` state. The Stage0 file is
  called `ema_best.pt`, but the historical recipe did **not** initialize
  Stage1 from the EMA shadow.
- Recorded runtime: Python 3.10.19, PyTorch 2.9.1+cu128, cuDNN 91002, RTX A6000.
- Saved reference Stage1 SHA-256:
  `45146fce10112b0abbc6c782d55099cb18593ec413019f4f52cd8d469725bf03`.

Use the repository's single `requirements.txt`. For byte-level comparison,
install the recorded PyTorch 2.9.1+cu128 stack explicitly; the portable
requirements file intentionally leaves the PyTorch wheel to the target
machine. Exact byte-for-byte equality is only a realistic expectation on the
recorded CUDA stack. Other CPU/GPU/MPS machines should reproduce the recipe and
comparable loss trajectory, but floating-point kernel differences can change
the final hash and the epoch selected as best.

For intentional ablations, the ordinary Stage configs remain configurable.
Set `parent_encoder_weights="ema"` explicitly to test EMA initialization;
enable AMP, channels-last, or fused AdamW only as separate experiments. The stage batch configurations enable `reproducibility` by default; set it
to `False` for training without that initialization seed.
