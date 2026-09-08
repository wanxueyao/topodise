from __future__ import annotations

import argparse
import platform
import time

import torch

from .config import load_config, resolved_paths
from .model import CommonStage2
from .utils import atomic_json_dump, pick_device


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description="Microbenchmark Stage-2 patient batching")
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--patients", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = pick_device(args.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    raw_dim = 512
    patients, pairs = args.patients, 8
    torch.manual_seed(20260831)
    central = torch.randn(patients, pairs, raw_dim, device=device)
    whorl = torch.randn(patients, pairs, raw_dim, device=device)
    morphology = torch.randn(patients, 21, device=device)
    mask = torch.ones(patients, pairs, dtype=torch.bool, device=device)
    model = CommonStage2(
        raw_dim=raw_dim,
        num_classes=3,
        common_dim=int(cfg["model"]["common_dim"]),
        adapter_rank=int(cfg["model"]["adapter_rank"]),
        attention_hidden=int(cfg["model"]["attention_hidden"]),
        attention_dropout=0.0,
        morphology_dim=int(cfg["model"]["morphology_dim"]),
    ).to(device).eval()

    sequential = torch.cat([
        model(central[i], whorl[i], morphology[i])[0] for i in range(patients)
    ])
    batched, _ = model(central, whorl, morphology, mask)
    max_difference = float((sequential - batched).abs().max().cpu())
    if max_difference > 1e-5:
        raise AssertionError(f"batched logits differ from sequential logits: {max_difference}")

    for _ in range(3):
        model(central, whorl, morphology, mask)
    _sync(device)
    start = time.perf_counter()
    for _ in range(args.iterations):
        for i in range(patients):
            model(central[i], whorl[i], morphology[i])
    _sync(device)
    sequential_seconds = time.perf_counter() - start

    eval_batch = int(cfg["optimization"].get("eval_patient_batch_size", 64))
    start = time.perf_counter()
    for _ in range(args.iterations):
        for offset in range(0, patients, eval_batch):
            end = offset + eval_batch
            model(central[offset:end], whorl[offset:end], morphology[offset:end], mask[offset:end])
    _sync(device)
    batched_seconds = time.perf_counter() - start
    report = {
        "device": str(device),
        "platform": platform.platform(),
        "patients": patients,
        "pairs_per_patient": pairs,
        "iterations": args.iterations,
        "sequential_seconds": sequential_seconds,
        "batched_seconds": batched_seconds,
        "speedup": sequential_seconds / batched_seconds,
        "max_logit_difference": max_difference,
    }
    output = resolved_paths(cfg)["output_root"] / "benchmarks" / f"stage2_{device.type}.json"
    atomic_json_dump(report, output)
    print(report)


if __name__ == "__main__":
    main()
