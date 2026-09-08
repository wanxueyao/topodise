# Reproducing the c8bf6b548656 Stage0/Stage1 checkpoint

The dedicated entry point freezes the Stage0/Stage1 values recorded in
`reproduction/_experiment_def.json`. It does not depend on the current batch
experiment defaults and resolves all paths independently of the launch working
directory.

From the `codes` directory:

```bash
python -m experiments.topodise_experiments.reproduce_c8bf6b548656 \
  --data-root datasets \
  --output-dir results/reproduce_c8bf6b548656 \
  --device cuda
```

The fresh-run command refuses to reuse or overwrite an existing checkpoint.
Pass `--resume` to continue a run; new checkpoints contain PyTorch CPU/CUDA RNG
states so resumed epochs keep the same random streams.

Stage0 and Stage1 expose one `reproducibility` switch. It controls only whether
each stage calls `torch.manual_seed(42)` during initialization. When disabled
(CLI: `--no-reproducibility`), no seed is set and training uses the environment
RNG state. Neither mode sets Python/NumPy seeds, cuDNN options, TF32/matmul
precision, deterministic algorithms, or `CUBLAS_WORKSPACE_CONFIG`. Checkpoints
always save PyTorch CPU/CUDA RNG states, which are restored when resuming
regardless of this switch.

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
enable AMP, channels-last, or fused AdamW only as separate experiments. The
dedicated reproduction entry enables `reproducibility` by default; use
`--no-reproducibility` for ordinary training without the fixed seed.
