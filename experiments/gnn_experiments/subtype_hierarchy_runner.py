"""
subtype_hierarchy_runner.py - GNN subtype 层级三分类公平重算 runner

方法与 basic / MTL 版本一致：
- TD-vs-rest gate: TD=0, rest=1
- PIGD-vs-Mixed: PIGD=0, mixed=1，训练排除 TD
- 两个二分类 GNN subtype head，patient-level 概率均值聚合后层级融合
- 输出 summary.json 与 fold_predictions.csv
"""
from __future__ import annotations

import csv
import json
import os
import sys
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

_THIS_FILE = Path(__file__).resolve()
CODES_ROOT = _THIS_FILE.parents[2]
if str(CODES_ROOT) not in sys.path:
    sys.path.insert(0, str(CODES_ROOT))

from experiments.gnn_experiments.gnn_main import GNNExperimentConfig
from gnn.graph_dataset import NerveGraphDataset, collate_fn
from gnn.models import GNNMultiTaskModel
from gnn.trainer import batch_to_device
from utils.misc import set_seed
from utils.subtype_hierarchy import (
    SUBTYPE_TO_3CLS,
    build_patient_5x3_subtype_folds,
    compute_subtype_hierarchy_metrics,
    fuse_subtype_predictions,
    subtype_pigd_mixed_label,
    subtype_tdrest_label,
    summarize_15fold_results,
)


def _subtype_binary_label(subtype: str, task: str) -> int:
    y3 = SUBTYPE_TO_3CLS[subtype]
    if task == 'pd_subtype_tdrest':
        return subtype_tdrest_label(y3)
    label = subtype_pigd_mixed_label(y3)
    return -1 if label is None else label


def _build_dataset_with_patient_ids(
    patients: List[Dict],
    base_path: str,
    cache_dir: str,
    task: str | None = None,
) -> Tuple[NerveGraphDataset, List[str]]:
    """构建 NerveGraphDataset，并生成与 samples 顺序严格对齐的 patient_id 列表。"""
    ds = NerveGraphDataset(patients, base_path, cache_dir=cache_dir)

    sample_pids: List[str] = []
    for patient in patients:
        central = patient['image_paths'].get('central', [])
        whorl = patient['image_paths'].get('whorl', [])
        for c_rel, w_rel in zip(central, whorl):
            c_path = os.path.join(base_path, c_rel)
            w_path = os.path.join(base_path, w_rel)
            if os.path.exists(c_path) and os.path.exists(w_path):
                sample_pids.append(patient['patient_id'])

    if len(sample_pids) != len(ds.samples):
        raise RuntimeError(
            f"sample/patient_id alignment mismatch: {len(ds.samples)} samples vs "
            f"{len(sample_pids)} ids"
        )

    if task is not None:
        subtype_by_pid = {p['patient_id']: p['subtype'] for p in patients}
        relabeled = []
        for sample, pid in zip(ds.samples, sample_pids):
            c_path, w_path, pd_l, stage_l, _sub_l = sample
            relabeled.append((
                c_path,
                w_path,
                pd_l,
                stage_l,
                _subtype_binary_label(subtype_by_pid[pid], task),
            ))
        ds.samples = relabeled

    return ds, sample_pids


def _build_model(config: GNNExperimentConfig, device: torch.device) -> GNNMultiTaskModel:
    model = GNNMultiTaskModel(
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        gnn_type=config.gnn_type,
        num_heads=config.num_heads,
        dropout=config.dropout,
        classifier_dropout=config.classifier_dropout,
        shared_encoder=config.shared_encoder,
        num_subtype_classes=2,
    )
    return model.to(device)


def _build_optimizer(config: GNNExperimentConfig, model: torch.nn.Module):
    if config.optimizer == 'adamw':
        return torch.optim.AdamW(model.parameters(), lr=config.lr,
                                 weight_decay=config.weight_decay)
    return torch.optim.Adam(model.parameters(), lr=config.lr,
                            weight_decay=config.weight_decay)


def _build_scheduler(config: GNNExperimentConfig, optimizer):
    if config.lr_scheduler == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(config.epochs - config.warmup_epochs, 1)
        )
    if config.lr_scheduler == 'step':
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=max(config.epochs // 3, 1), gamma=0.5
        )
    return None


def _train_one_subtask(
    config: GNNExperimentConfig,
    train_patients: List[Dict],
    val_patients: List[Dict],
    base_path: str,
    cache_dir: str,
    device: torch.device,
    task: str,
) -> GNNMultiTaskModel:
    train_ds, _ = _build_dataset_with_patient_ids(train_patients, base_path, cache_dir, task)
    val_ds, _ = _build_dataset_with_patient_ids(val_patients, base_path, cache_dir, task)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config.num_workers,
        drop_last=(len(train_ds) > config.batch_size),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config.num_workers,
    )

    model = _build_model(config, device)
    optimizer = _build_optimizer(config, model)
    scheduler = _build_scheduler(config, optimizer)

    best_acc = -1.0
    best_state = None
    patience_counter = 0

    for epoch in range(1, config.epochs + 1):
        if 0 < config.warmup_epochs and epoch <= config.warmup_epochs:
            for group in optimizer.param_groups:
                group['lr'] = config.lr * epoch / config.warmup_epochs

        model.train()
        total_loss, total_items = 0.0, 0
        for batch in train_loader:
            central_b, whorl_b, _pd_ls, _stage_ls, sub_ls = batch[:5]
            central_b = batch_to_device(central_b, device)
            whorl_b = batch_to_device(whorl_b, device)
            sub_ls = sub_ls.to(device)
            mask = sub_ls >= 0
            if not mask.any():
                continue

            optimizer.zero_grad(set_to_none=True)
            _pd_out, _stage_out, sub_out = model(central_b, whorl_b)
            loss = F.cross_entropy(
                sub_out[mask],
                sub_ls[mask],
                label_smoothing=config.label_smoothing,
            ) * config.subtype_loss_weight
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += float(loss.item()) * int(mask.sum().item())
            total_items += int(mask.sum().item())

        if scheduler is not None and epoch > config.warmup_epochs:
            scheduler.step()

        val_acc = _eval_binary_acc(model, val_loader, device)
        improved = val_acc > best_acc
        if improved:
            best_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        train_loss = total_loss / max(1, total_items)
        if epoch == 1 or epoch == config.epochs or epoch % 10 == 0:
            print(
                f"    [{task}] E{epoch:03d}/{config.epochs} "
                f"train_loss={train_loss:.4f} val_bin_acc={val_acc:.4f}"
                f"{' *' if improved else ''}",
                flush=True,
            )
        if patience_counter >= config.patience:
            print(f"    [{task}] early stop at epoch {epoch}", flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def _eval_binary_acc(model: GNNMultiTaskModel, loader: DataLoader,
                     device: torch.device) -> float:
    model.eval()
    correct, total = 0, 0
    for batch in loader:
        central_b, whorl_b, _pd_ls, _stage_ls, sub_ls = batch[:5]
        central_b = batch_to_device(central_b, device)
        whorl_b = batch_to_device(whorl_b, device)
        sub_ls = sub_ls.to(device)
        mask = sub_ls >= 0
        if not mask.any():
            continue
        _pd_out, _stage_out, sub_out = model(central_b, whorl_b)
        pred = sub_out.argmax(dim=1)
        correct += int((pred[mask] == sub_ls[mask]).sum().item())
        total += int(mask.sum().item())
    return correct / total if total else 0.0


@torch.no_grad()
def _infer_patient_level(
    model: GNNMultiTaskModel,
    patients: List[Dict],
    config: GNNExperimentConfig,
    base_path: str,
    cache_dir: str,
    device: torch.device,
) -> Tuple[Dict[str, int], Dict[str, float]]:
    ds, sample_pids = _build_dataset_with_patient_ids(patients, base_path, cache_dir)
    loader = DataLoader(
        ds,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config.num_workers,
    )

    model.eval()
    sums: Dict[str, np.ndarray] = defaultdict(lambda: np.zeros(2, dtype=np.float64))
    counts: Dict[str, int] = defaultdict(int)
    offset = 0
    for batch in loader:
        central_b, whorl_b, _pd_ls, _stage_ls, sub_ls = batch[:5]
        batch_size = int(sub_ls.size(0))
        central_b = batch_to_device(central_b, device)
        whorl_b = batch_to_device(whorl_b, device)
        _pd_out, _stage_out, sub_out = model(central_b, whorl_b)
        probs = torch.softmax(sub_out, dim=1).cpu().numpy()
        for i, pid in enumerate(sample_pids[offset:offset + batch_size]):
            sums[pid] += probs[i]
            counts[pid] += 1
        offset += batch_size

    pred_by_pid: Dict[str, int] = {}
    prob_class1_by_pid: Dict[str, float] = {}
    for pid, total_prob in sums.items():
        mean_prob = total_prob / max(1, counts[pid])
        pred_by_pid[pid] = int(np.argmax(mean_prob))
        prob_class1_by_pid[pid] = float(mean_prob[1])
    return pred_by_pid, prob_class1_by_pid


def run_gnn_subtype_hierarchy_experiment(
    config: GNNExperimentConfig,
    output_dir: str,
    seed: int = 42,
    n_splits: int = 5,
    repeats: int = 3,
) -> Dict:
    """运行 GNN subtype 层级三分类 5-fold x 3 重算实验。"""
    os.makedirs(output_dir, exist_ok=True)
    set_seed(seed)

    cfg = deepcopy(config)
    cfg.primary_task = 'subtype'
    cfg.pd_loss_weight = 0.0
    cfg.stage_loss_weight = 0.0
    if cfg.subtype_loss_weight <= 0:
        cfg.subtype_loss_weight = 1.0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_path = str(CODES_ROOT / 'datasets')
    cache_dir = cfg.graph_cache_dir or str(CODES_ROOT / 'results' / 'graph_cache')
    json_path = str(CODES_ROOT / cfg.dataset_json)

    with open(json_path, 'r', encoding='utf-8') as f:
        all_patients = json.load(f)
    subtype_patients = [
        p for p in all_patients
        if p.get('group') == 'pd' and p.get('subtype') in SUBTYPE_TO_3CLS
    ]
    pid_to_patient = {p['patient_id']: p for p in subtype_patients}
    records = [{'patient_id': p['patient_id'], 'subtype': p['subtype']}
               for p in subtype_patients]
    folds = build_patient_5x3_subtype_folds(
        records, seed=seed, n_splits=n_splits, repeats=repeats
    )

    print('=' * 60)
    print('GNN Subtype 层级三分类公平重算 (5-fold x 3, 仅 subtype 任务)')
    print(f"subtype 患者数: {len(subtype_patients)} | gnn={cfg.gnn_type} | "
          f"hidden={cfg.hidden_dim} | folds={len(folds)} | device={device}")
    print('=' * 60)

    fold_results: List[Dict] = []
    pred_rows: List[Dict] = []

    for global_idx, fold in enumerate(folds, start=1):
        repeat, fold_no = fold['repeat'], fold['fold']
        train_patients = [pid_to_patient[pid] for pid in fold['train_patients_full']]
        val_patients = [pid_to_patient[pid] for pid in fold['val_patients_full']]
        print(
            f"\n--- repeat {repeat} fold {fold_no} ({global_idx}/{len(folds)}) | "
            f"train={len(train_patients)} val={len(val_patients)} ---",
            flush=True,
        )

        tdrest_model = _train_one_subtask(
            cfg, train_patients, val_patients, base_path, cache_dir, device,
            'pd_subtype_tdrest'
        )
        tdrest_pred, tdrest_prob = _infer_patient_level(
            tdrest_model, val_patients, cfg, base_path, cache_dir, device
        )
        del tdrest_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        pm_train_patients = [p for p in train_patients if p['subtype'] in ('PIGD', 'mixed')]
        pm_val_patients = [p for p in val_patients if p['subtype'] in ('PIGD', 'mixed')]
        pm_model = _train_one_subtask(
            cfg, pm_train_patients, pm_val_patients, base_path, cache_dir, device,
            'pd_subtype_pigd_mixed'
        )
        pm_pred, pm_prob = _infer_patient_level(
            pm_model, val_patients, cfg, base_path, cache_dir, device
        )
        del pm_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        rows: List[Dict] = []
        for pid in fold['val_patients_full']:
            true_subtype = SUBTYPE_TO_3CLS[pid_to_patient[pid]['subtype']]
            gate_pred = tdrest_pred.get(pid, 1)
            pm_stage_pred = pm_pred.get(pid)
            final_pred = fuse_subtype_predictions(gate_pred, pm_stage_pred)
            rows.append({
                'true_subtype': true_subtype,
                'tdrest_pred': gate_pred,
                'pigd_mixed_pred': pm_stage_pred,
                'final_pred': final_pred,
            })
            pred_rows.append({
                'repeat': repeat,
                'fold': fold_no,
                'patient_id': pid,
                'true_subtype': true_subtype,
                'tdrest_pred': gate_pred,
                'tdrest_prob_rest': round(tdrest_prob.get(pid, float('nan')), 4),
                'pigd_mixed_pred': '' if pm_stage_pred is None else pm_stage_pred,
                'pigd_mixed_prob_mixed': round(pm_prob.get(pid, float('nan')), 4),
                'final_pred': final_pred,
            })

        metrics = compute_subtype_hierarchy_metrics(rows)
        metrics['fold'] = fold_no
        metrics['repeat'] = repeat
        fold_results.append(metrics)
        print(
            f"  tdrest_acc={metrics['tdrest_acc']:.4f} | "
            f"pigd_mixed_acc={metrics['pigd_mixed_acc']:.4f} | "
            f"hierarchy_acc={metrics['subtype_hierarchy_acc']:.4f}",
            flush=True,
        )

    summary = summarize_15fold_results(fold_results, n_splits=n_splits, repeats=repeats)
    summary['gnn_type'] = cfg.gnn_type
    summary['hidden_dim'] = cfg.hidden_dim
    summary['seed'] = seed

    summary_path = os.path.join(output_dir, 'summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    csv_path = os.path.join(output_dir, 'fold_predictions.csv')
    fieldnames = [
        'repeat', 'fold', 'patient_id', 'true_subtype', 'tdrest_pred',
        'tdrest_prob_rest', 'pigd_mixed_pred', 'pigd_mixed_prob_mixed',
        'final_pred',
    ]
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(pred_rows)

    print('\n' + '=' * 60)
    print('GNN Subtype 层级三分类汇总:')
    print(f"  tdrest_acc         = {summary['tdrest_acc_mean']:.4f} "
          f"± {summary['tdrest_acc_std']:.4f}")
    print(f"  pigd_mixed_acc     = {summary['pigd_mixed_acc_mean']:.4f} "
          f"± {summary['pigd_mixed_acc_std']:.4f}")
    print(f"  hierarchy_3cls_acc = {summary['subtype_hierarchy_acc_mean']:.4f} "
          f"± {summary['subtype_hierarchy_acc_std']:.4f}")
    print(f"  summary -> {summary_path}")
    print(f"  predictions -> {csv_path}")
    print('=' * 60)

    return summary
