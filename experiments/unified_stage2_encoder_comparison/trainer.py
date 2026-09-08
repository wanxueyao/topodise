from __future__ import annotations

import math
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F
from core.progress import ETAEstimator

from .bag_store import PackedBagStore
from .metrics import classification_metrics
from .model import CommonStage2
from .utils import atomic_json_dump, append_jsonl, seed_everything


def _autocast(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return torch.autocast(device_type="cpu", enabled=False)


def _class_weights(
    ids: list[str], store: PackedBagStore, num_classes: int
) -> torch.Tensor:
    counts = torch.bincount(
        torch.tensor([store.label(pid) for pid in ids]), minlength=num_classes
    ).float()
    if (counts == 0).any():
        raise RuntimeError(f"training split misses a class: counts={counts.tolist()}")
    weights = 1.0 / counts
    return (weights / weights.mean()).to(store.device)


def _epoch_order(
    ids: list[str], store: PackedBagStore, task: str, seed: int, epoch: int
) -> list[str]:
    rng = random.Random(seed + epoch * 1009)
    if task == "pd_binary":
        order = list(ids)
        rng.shuffle(order)
        return order
    by_class: dict[int, list[str]] = defaultdict(list)
    for pid in ids:
        by_class[store.label(pid)].append(pid)
    target = max(len(items) for items in by_class.values())
    expanded = {}
    for label, items in by_class.items():
        rng.shuffle(items)
        expanded[label] = [items[index % len(items)] for index in range(target)]
    labels = sorted(expanded)
    order = [expanded[label][index] for index in range(target) for label in labels]
    rng.shuffle(order)
    return order


def _chunks(items: list[str], size: int):
    for start in range(0, len(items), size):
        yield items[start:start + size]


def _make_model(
    raw_dim: int, num_classes: int, model_cfg: dict, device: torch.device
) -> CommonStage2:
    return CommonStage2(
        raw_dim=raw_dim,
        num_classes=num_classes,
        common_dim=int(model_cfg["common_dim"]),
        adapter_rank=int(model_cfg["adapter_rank"]),
        attention_hidden=int(model_cfg["attention_hidden"]),
        attention_dropout=float(model_cfg["attention_dropout"]),
        morphology_dim=int(model_cfg["morphology_dim"]),
        input_mode=str(model_cfg.get("input_mode", "region")),
    ).to(device)


def _make_optimizer(model: CommonStage2, optim_cfg: dict, device: torch.device):
    kwargs = {
        "lr": float(optim_cfg["lr"]),
        "weight_decay": float(optim_cfg["weight_decay"]),
    }
    if device.type == "cuda" and bool(optim_cfg.get("fused_adamw", True)):
        try:
            return torch.optim.AdamW(model.parameters(), fused=True, **kwargs)
        except (TypeError, RuntimeError):
            pass
    return torch.optim.AdamW(model.parameters(), **kwargs)


def _scheduler(optimizer, optim_cfg: dict, total_epochs: int):
    warmup = int(optim_cfg["warmup_epochs"])
    min_ratio = float(optim_cfg["min_lr"]) / float(optim_cfg["lr"])

    def factor(epoch: int) -> float:
        if epoch < warmup:
            return float(epoch + 1) / max(warmup, 1)
        progress = (epoch - warmup) / max(total_epochs - warmup - 1, 1)
        return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=factor)


def _patient_batch_size(optim_cfg: dict) -> int:
    effective = int(optim_cfg["accumulate_patients"])
    batch_size = int(optim_cfg.get("patient_batch_size", effective))
    if batch_size != effective:
        raise ValueError(
            "patient_batch_size must equal accumulate_patients in Core-v1 so the effective "
            "optimization batch remains locked"
        )
    return batch_size


def _train_one_epoch(
    model: CommonStage2,
    store: PackedBagStore,
    train_ids: list[str],
    task: str,
    epoch: int,
    seed: int,
    optimizer,
    class_weights: torch.Tensor | None,
    optim_cfg: dict,
    amp: bool,
) -> float:
    model.train()
    order = _epoch_order(train_ids, store, task, seed, epoch)
    batch_size = _patient_batch_size(optim_cfg)
    weighted_loss_sum = 0.0
    for batch_ids in _chunks(order, batch_size):
        central, whorl, morphology, targets, pair_mask = store.batch(batch_ids)
        optimizer.zero_grad(set_to_none=True)
        with _autocast(store.device, amp):
            logits, _ = model(central, whorl, morphology, pair_mask)
            per_patient = F.cross_entropy(
                logits.float(), targets, weight=class_weights, reduction="none"
            )
                # 保持锁定的有效批大小和历史最后不完整批次的缩放方式。
            loss = per_patient.sum() / batch_size
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(optim_cfg["grad_clip"]))
        optimizer.step()
        weighted_loss_sum += float(per_patient.detach().sum())
    return weighted_loss_sum / max(len(order), 1)


def _train_epochs(
    model: CommonStage2,
    store: PackedBagStore,
    train_ids: list[str],
    task: str,
    num_classes: int,
    epochs: int,
    optim_cfg: dict,
    amp: bool,
    seed: int,
    event_path: Path | None = None,
    progress_callback: Callable[[dict], None] | None = None,
) -> None:
    optimizer = _make_optimizer(model, optim_cfg, store.device)
    scheduler = _scheduler(optimizer, optim_cfg, epochs)
    weights = None if task == "pd_binary" else _class_weights(train_ids, store, num_classes)
    progress = ETAEstimator(max(epochs, 1), window=5)
    for epoch in range(epochs):
        epoch_started = time.perf_counter()
        loss = _train_one_epoch(
            model, store, train_ids, task, epoch, seed, optimizer, weights, optim_cfg, amp
        )
        scheduler.step()
        epoch_seconds = time.perf_counter() - epoch_started
        state = progress.update(epoch + 1, interval_seconds=epoch_seconds)
        event = {
            "kind": "stage2_epoch_progress",
            "phase": "outer_retrain",
            "epoch": epoch + 1,
            "epochs": epochs,
            "epoch_seconds": round(epoch_seconds, 3),
            "loss": loss,
            "lr": optimizer.param_groups[0]["lr"],
            **state,
        }
        if event_path is not None:
            append_jsonl(event, event_path)
        if progress_callback is not None:
            progress_callback(event)


@torch.inference_mode()
def evaluate(
    model: CommonStage2,
    store: PackedBagStore,
    patient_ids: list[str],
    amp: bool,
    batch_size: int = 64,
) -> tuple[dict, list[dict]]:
    model.eval()
    predictions = []
    for batch_ids in _chunks(patient_ids, batch_size):
        central, whorl, morphology, targets, pair_mask = store.batch(batch_ids)
        with _autocast(store.device, amp):
            logits, attention = model(central, whorl, morphology, pair_mask)
        probabilities = torch.softmax(logits.float(), dim=-1).cpu().tolist()
        labels = targets.cpu().tolist()
        attention = attention.float().cpu()
        valid_counts = pair_mask.sum(dim=1).cpu().tolist()
        for row_index, pid in enumerate(batch_ids):
            probs = probabilities[row_index]
            predictions.append({
                "patient_id": pid,
                "label": int(labels[row_index]),
                "prediction": int(max(range(len(probs)), key=probs.__getitem__)),
                "probabilities": probs,
                "attention": attention[row_index, :valid_counts[row_index]].tolist(),
            })
    metrics = classification_metrics(
        [row["label"] for row in predictions], [row["probabilities"] for row in predictions]
    )
    return metrics, predictions


def select_epoch(
    raw_dim: int,
    num_classes: int,
    model_cfg: dict,
    optim_cfg: dict,
    store: PackedBagStore,
    inner_train_ids: list[str],
    inner_val_ids: list[str],
    task: str,
    amp: bool,
    seed: int,
    event_path: Path,
    progress_callback: Callable[[dict], None] | None = None,
) -> tuple[int, list[dict]]:
    seed_everything(seed)
    model = _make_model(raw_dim, num_classes, model_cfg, store.device)
    optimizer = _make_optimizer(model, optim_cfg, store.device)
    epochs = int(optim_cfg["epochs"])
    scheduler = _scheduler(optimizer, optim_cfg, epochs)
    weights = None if task == "pd_binary" else _class_weights(inner_train_ids, store, num_classes)
    eval_batch_size = int(optim_cfg.get("eval_patient_batch_size", 64))
    history, best_epoch, best_score = [], 1, -1.0
    progress = ETAEstimator(epochs, window=5)
    for epoch in range(epochs):
        epoch_started = time.perf_counter()
        loss = _train_one_epoch(
            model, store, inner_train_ids, task, epoch, seed,
            optimizer, weights, optim_cfg, amp,
        )
        val_metrics, _ = evaluate(model, store, inner_val_ids, amp, eval_batch_size)
        row = {
            "epoch": epoch + 1,
            "train_loss": loss,
            "inner_val_macro_f1": val_metrics["macro_f1"],
            "inner_val_balanced_accuracy": val_metrics["balanced_accuracy"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        epoch_seconds = time.perf_counter() - epoch_started
        state = progress.update(epoch + 1, interval_seconds=epoch_seconds)
        event = {
            "kind": "stage2_epoch_progress",
            "phase": "inner_selection",
            "epoch_seconds": round(epoch_seconds, 3),
            **row,
            **state,
        }
        append_jsonl(event, event_path)
        if progress_callback is not None:
            progress_callback(event)
        if row["inner_val_macro_f1"] > best_score:
            best_score = row["inner_val_macro_f1"]
            best_epoch = epoch + 1
        scheduler.step()
    return best_epoch, history


def run_outer_fold(
    *,
    raw_dim: int,
    num_classes: int,
    model_cfg: dict,
    optim_cfg: dict,
    bags: dict[str, dict] | PackedBagStore,
    fold_spec: dict,
    task: str,
    encoder_name: str,
    split_seed: int,
    device: torch.device,
    amp: bool,
    run_dir: Path,
    provenance: dict,
    progress_callback: Callable[[dict], None] | None = None,
) -> dict:
    started = time.perf_counter()
    store = bags if isinstance(bags, PackedBagStore) else PackedBagStore.from_bags(bags, device)
    input_mode = str(model_cfg.get("input_mode", "region"))
    if store.feature_mode != input_mode:
        raise ValueError(
            "packed feature mode and CommonStage2 input mode differ: "
            f"{store.feature_mode} != {input_mode}"
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    event_path = run_dir / "events.jsonl"
    for stale in (event_path, run_dir / "predictions.jsonl", run_dir / "DONE.json"):
        if stale.exists():
            stale.unlink()
    training_seed = split_seed * 1000 + int(fold_spec["fold"])
    best_epoch, history = select_epoch(
        raw_dim, num_classes, model_cfg, optim_cfg, store,
        fold_spec["inner_train"], fold_spec["inner_val"], task,
        amp, training_seed, event_path, progress_callback,
    )
    seed_everything(training_seed)
    model = _make_model(raw_dim, num_classes, model_cfg, store.device)
    _train_epochs(
        model, store, fold_spec["outer_train"], task, num_classes, best_epoch,
        optim_cfg, amp, training_seed, event_path, progress_callback,
    )
            # 这是唯一一次评估外层测试标签的调用。
    outer_metrics, predictions = evaluate(
        model, store, fold_spec["outer_test"], amp,
        int(optim_cfg.get("eval_patient_batch_size", 64)),
    )
    for row in predictions:
        row.update({
            "encoder": encoder_name,
            "task": task,
            "split_seed": split_seed,
            "fold": int(fold_spec["fold"]),
            "protocol_version": provenance.get("protocol_version", "smoke"),
            "protocol_config_sha256": provenance.get("protocol_config_sha256", "smoke"),
        })
        append_jsonl(row, run_dir / "predictions.jsonl")
    checkpoint_path = run_dir / "common_stage2.pt"
    torch.save({
        "model_state": model.state_dict(),
        "raw_dim": raw_dim,
        "num_classes": num_classes,
        "selected_epoch": best_epoch,
        "training_seed": training_seed,
        "provenance": provenance,
    }, checkpoint_path)
    elapsed = time.perf_counter() - started
    summary = {
        **provenance,
        "encoder": encoder_name,
        "task": task,
        "split_seed": split_seed,
        "fold": int(fold_spec["fold"]),
        "selected_epoch": best_epoch,
        "inner_history": history,
        "outer_metrics": outer_metrics,
        "num_train_patients": len(fold_spec["outer_train"]),
        "num_test_patients": len(fold_spec["outer_test"]),
        "patient_batch_size": _patient_batch_size(optim_cfg),
        "eval_patient_batch_size": int(optim_cfg.get("eval_patient_batch_size", 64)),
        "trainable_stage2_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "stage2_input_mode": str(model_cfg.get("input_mode", "region")),
        "elapsed_seconds": elapsed,
        "checkpoint": str(checkpoint_path),
    }
    atomic_json_dump(summary, run_dir / "summary.json")
    atomic_json_dump({"status": "complete", "elapsed_seconds": elapsed}, run_dir / "DONE.json")
    return summary
