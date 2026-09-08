"""
dual_stream_subtype_hierarchy_runner.py - 双视角 CNN subtype 层级三分类重算

参考 files/subtype_hierarchy_revalidation_guide.md：
- TD-vs-rest gate: TD=0, rest=1
- PIGD-vs-Mixed: PIGD=0, mixed=1，训练排除 TD
- full validation fold 上做 patient-level 概率均值聚合，再层级融合
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
from sklearn.metrics import accuracy_score

_THIS_FILE = Path(__file__).resolve()
CODES_ROOT = _THIS_FILE.parents[2]
if str(CODES_ROOT) not in sys.path:
    sys.path.insert(0, str(CODES_ROOT))

from experiments.gnn_experiments.dual_stream_main import DualStreamExperimentConfig
from gnn.dual_view_classifier import _build_transforms
from gnn.dual_view_dataset import DualViewDataset
from gnn.dual_view_model import DualViewModel
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


def _load_json(path: str) -> List[Dict]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _align_subtype_patients(config: DualStreamExperimentConfig) -> Tuple[List[Dict], List[Dict]]:
    en_json = CODES_ROOT / config.en_dataset_json
    unet_json = CODES_ROOT / config.unet_dataset_json
    en_all = sorted(_load_json(str(en_json)), key=lambda p: p['patient_id'])
    unet_map = {p['patient_id']: p for p in _load_json(str(unet_json))}

    en_patients = [
        p for p in en_all
        if p.get('group') == 'pd'
        and p.get('subtype') in SUBTYPE_TO_3CLS
        and p['patient_id'] in unet_map
    ]
    unet_patients = [unet_map[p['patient_id']] for p in en_patients]
    return en_patients, unet_patients


def _binary_label(subtype: str, task: str) -> int:
    y3 = SUBTYPE_TO_3CLS[subtype]
    if task == 'pd_subtype_tdrest':
        return subtype_tdrest_label(y3)
    label = subtype_pigd_mixed_label(y3)
    return -1 if label is None else label


def _sample_patient_ids(
    en_patients: List[Dict],
    unet_patients: List[Dict],
    base_path: str,
    region: str,
) -> List[str]:
    sample_pids: List[str] = []
    for en_p, unet_p in zip(en_patients, unet_patients):
        en_paths = en_p['image_paths']
        unet_paths = unet_p['image_paths']
        if region in ('both', 'central'):
            for ep, up in zip(en_paths.get('central', []), unet_paths.get('central', [])):
                if os.path.exists(os.path.join(base_path, ep)) and os.path.exists(os.path.join(base_path, up)):
                    sample_pids.append(en_p['patient_id'])
        if region in ('both', 'whorl'):
            for ep, up in zip(en_paths.get('whorl', []), unet_paths.get('whorl', [])):
                if os.path.exists(os.path.join(base_path, ep)) and os.path.exists(os.path.join(base_path, up)):
                    sample_pids.append(en_p['patient_id'])
    return sample_pids


def _build_dataset_with_patient_ids(
    en_patients: List[Dict],
    unet_patients: List[Dict],
    base_path: str,
    en_transform,
    unet_transform,
    region: str,
    task: str | None = None,
) -> Tuple[DualViewDataset, List[str]]:
    ds = DualViewDataset(
        en_patients,
        unet_patients,
        base_path,
        en_transform=en_transform,
        unet_transform=unet_transform,
        region=region,
    )
    sample_pids = _sample_patient_ids(en_patients, unet_patients, base_path, region)
    if len(sample_pids) != len(ds.samples):
        raise RuntimeError(
            f"sample/patient alignment mismatch: {len(ds.samples)} samples vs {len(sample_pids)} ids"
        )

    if task is not None:
        subtype_by_pid = {p['patient_id']: p['subtype'] for p in en_patients}
        relabeled = []
        for sample, pid in zip(ds.samples, sample_pids):
            en_path, unet_path, pd_l, stage_l, _sub_l = sample[:5]
            relabeled.append((
                en_path,
                unet_path,
                pd_l,
                stage_l,
                _binary_label(subtype_by_pid[pid], task),
                pid,
            ))
        ds.samples = relabeled

    return ds, sample_pids


def _build_model(config: DualStreamExperimentConfig, task: str, device: torch.device) -> DualViewModel:
    model = DualViewModel(
        task=task,
        backbone=config.backbone,
        pretrained=config.pretrained,
        shared_backbone=config.shared_backbone,
        fusion_dim=config.fusion_dim,
        dropout=config.dropout,
        classifier_dropout=config.classifier_dropout,
    )
    return model.to(device)


def _build_optimizer(config: DualStreamExperimentConfig, model: torch.nn.Module):
    backbone_params, other_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'backbone_' in name:
            backbone_params.append(param)
        else:
            other_params.append(param)

    param_groups = [
        {'params': backbone_params, 'lr': config.learning_rate * config.backbone_lr_ratio},
        {'params': other_params, 'lr': config.learning_rate},
    ]
    if config.optimizer == 'adam':
        return torch.optim.Adam(param_groups, weight_decay=config.weight_decay)
    return torch.optim.AdamW(param_groups, weight_decay=config.weight_decay)


def _build_scheduler(config: DualStreamExperimentConfig, optimizer):
    def _lr_lambda(epoch: int) -> float:
        if epoch < config.warmup_epochs:
            return (epoch + 1) / max(config.warmup_epochs, 1)
        if config.lr_scheduler == 'cosine':
            import math
            progress = (epoch - config.warmup_epochs) / max(config.epochs - config.warmup_epochs, 1)
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)


def _train_one_subtask(
    config: DualStreamExperimentConfig,
    train_en: List[Dict],
    val_en: List[Dict],
    train_unet: List[Dict],
    val_unet: List[Dict],
    base_path: str,
    device: torch.device,
    task: str,
    fold_idx: int,
) -> DualViewModel:
    train_tf, val_tf = _build_transforms(_to_dual_view_config(config))
    train_ds, _ = _build_dataset_with_patient_ids(
        train_en, train_unet, base_path, train_tf, train_tf, config.region, task
    )
    val_ds, _ = _build_dataset_with_patient_ids(
        val_en, val_unet, base_path, val_tf, val_tf, config.region, task
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    model = _build_model(config, task, device)
    optimizer = _build_optimizer(config, model)
    scheduler = _build_scheduler(config, optimizer)

    best_metric = -1.0
    best_state = None
    patience_counter = 0

    for epoch in range(config.epochs):
        model.train()
        total_loss = 0.0
        batches = 0
        train_preds, train_labels = [], []
        for batch in train_loader:
            en_img, unet_img, _pd_l, _stage_l, sub_l = batch[:5]
            mask = sub_l >= 0
            if not mask.any():
                continue
            en_m = en_img[mask].to(device)
            unet_m = unet_img[mask].to(device)
            labels = sub_l[mask].to(device)

            logits = model(en_m, unet_m)
            loss = F.cross_entropy(logits, labels, label_smoothing=config.label_smoothing)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += float(loss.item())
            batches += 1
            train_preds.extend(logits.argmax(1).detach().cpu().numpy())
            train_labels.extend(labels.detach().cpu().numpy())

        scheduler.step()
        val_acc = _eval_binary_acc(model, val_loader, device)
        improved = val_acc > best_metric
        if improved:
            best_metric = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch == 0 or epoch + 1 == config.epochs or (epoch + 1) % 10 == 0:
            train_acc = accuracy_score(train_labels, train_preds) if train_labels else 0.0
            train_loss = total_loss / max(1, batches)
            print(
                f"    [{task}] fold={fold_idx} E{epoch + 1:03d}/{config.epochs} "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"val_bin_acc={val_acc:.4f}{' *' if improved else ''}",
                flush=True,
            )
        if patience_counter >= config.patience:
            print(f"    [{task}] early stop at epoch {epoch + 1}", flush=True)
            break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return model


def _to_dual_view_config(config: DualStreamExperimentConfig):
    from gnn.dual_view_classifier import DualViewConfig

    return DualViewConfig(
        task='subtype',
        backbone=config.backbone,
        pretrained=config.pretrained,
        shared_backbone=config.shared_backbone,
        fusion_dim=config.fusion_dim,
        dropout=config.dropout,
        classifier_dropout=config.classifier_dropout,
        lr=config.learning_rate,
        backbone_lr_ratio=config.backbone_lr_ratio,
        weight_decay=config.weight_decay,
        optimizer=config.optimizer,
        lr_scheduler=config.lr_scheduler,
        warmup_epochs=config.warmup_epochs,
        epochs=config.epochs,
        patience=config.patience,
        batch_size=config.batch_size,
        region=config.region,
        data_augmentation=config.data_augmentation,
        aug_horizontal_flip=config.aug_horizontal_flip,
        aug_rotation=config.aug_rotation,
        aug_brightness=config.aug_brightness,
        aug_contrast=config.aug_contrast,
        label_smoothing=config.label_smoothing,
        num_workers=config.num_workers,
        seed=config.seed,
    )


@torch.no_grad()
def _eval_binary_acc(model: DualViewModel, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct, total = 0, 0
    for batch in loader:
        en_img, unet_img, _pd_l, _stage_l, sub_l = batch[:5]
        mask = sub_l >= 0
        if not mask.any():
            continue
        logits = model(en_img[mask].to(device), unet_img[mask].to(device))
        preds = logits.argmax(1).cpu()
        labels = sub_l[mask]
        correct += int((preds == labels).sum().item())
        total += int(labels.numel())
    return correct / total if total else 0.0


@torch.no_grad()
def _infer_patient_level(
    model: DualViewModel,
    en_patients: List[Dict],
    unet_patients: List[Dict],
    config: DualStreamExperimentConfig,
    base_path: str,
    device: torch.device,
) -> Tuple[Dict[str, int], Dict[str, float]]:
    _train_tf, val_tf = _build_transforms(_to_dual_view_config(config))
    ds, sample_pids = _build_dataset_with_patient_ids(
        en_patients, unet_patients, base_path, val_tf, val_tf, config.region
    )
    loader = DataLoader(
        ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    model.eval()
    sums: Dict[str, np.ndarray] = defaultdict(lambda: np.zeros(2, dtype=np.float64))
    counts: Dict[str, int] = defaultdict(int)
    offset = 0
    for batch in loader:
        en_img, unet_img, _pd_l, _stage_l, sub_l = batch[:5]
        batch_size = int(sub_l.size(0))
        logits = model(en_img.to(device), unet_img.to(device))
        probs = torch.softmax(logits, dim=1).cpu().numpy()
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


def run_dual_stream_subtype_hierarchy_experiment(
    config: DualStreamExperimentConfig,
    output_dir: str,
    seed: int = 42,
    n_splits: int = 5,
    repeats: int = 3,
) -> Dict:
    os.makedirs(output_dir, exist_ok=True)
    cfg = deepcopy(config)
    cfg.seed = seed
    set_seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_path = str(CODES_ROOT / 'datasets')

    en_patients, unet_patients = _align_subtype_patients(cfg)
    en_by_pid = {p['patient_id']: p for p in en_patients}
    unet_by_pid = {p['patient_id']: p for p in unet_patients}
    records = [{'patient_id': p['patient_id'], 'subtype': p['subtype']} for p in en_patients]
    folds = build_patient_5x3_subtype_folds(records, seed=seed, n_splits=n_splits, repeats=repeats)

    print('=' * 60)
    print('Dual-stream subtype hierarchy revalidation (patient 5-fold x3)')
    print(f"subtype patients: {len(en_patients)} | backbone={cfg.backbone} | "
          f"shared={cfg.shared_backbone} | folds={len(folds)} | device={device}")
    print('=' * 60)

    fold_results: List[Dict] = []
    pred_rows: List[Dict] = []

    for global_idx, fold in enumerate(folds, start=1):
        repeat, fold_no = fold['repeat'], fold['fold']
        train_pids = fold['train_patients_full']
        val_pids = fold['val_patients_full']
        train_en = [en_by_pid[pid] for pid in train_pids]
        val_en = [en_by_pid[pid] for pid in val_pids]
        train_unet = [unet_by_pid[pid] for pid in train_pids]
        val_unet = [unet_by_pid[pid] for pid in val_pids]

        print(
            f"\n--- repeat {repeat} fold {fold_no} ({global_idx}/{len(folds)}) | "
            f"train={len(train_en)} val={len(val_en)} ---",
            flush=True,
        )

        tdrest_model = _train_one_subtask(
            cfg, train_en, val_en, train_unet, val_unet, base_path, device,
            'pd_subtype_tdrest', fold_idx=global_idx,
        )
        tdrest_pred, tdrest_prob = _infer_patient_level(
            tdrest_model, val_en, val_unet, cfg, base_path, device
        )
        del tdrest_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        pm_train_pids = [pid for pid in train_pids if en_by_pid[pid]['subtype'] in ('PIGD', 'mixed')]
        pm_val_pids = [pid for pid in val_pids if en_by_pid[pid]['subtype'] in ('PIGD', 'mixed')]
        pm_train_en = [en_by_pid[pid] for pid in pm_train_pids]
        pm_val_en = [en_by_pid[pid] for pid in pm_val_pids]
        pm_train_unet = [unet_by_pid[pid] for pid in pm_train_pids]
        pm_val_unet = [unet_by_pid[pid] for pid in pm_val_pids]

        pm_model = _train_one_subtask(
            cfg, pm_train_en, pm_val_en, pm_train_unet, pm_val_unet, base_path, device,
            'pd_subtype_pigd_mixed', fold_idx=global_idx,
        )
        pm_pred, pm_prob = _infer_patient_level(
            pm_model, val_en, val_unet, cfg, base_path, device
        )
        del pm_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        rows: List[Dict] = []
        for pid in val_pids:
            true_subtype = SUBTYPE_TO_3CLS[en_by_pid[pid]['subtype']]
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
    summary['backbone'] = cfg.backbone
    summary['shared_backbone'] = cfg.shared_backbone
    summary['fusion_dim'] = cfg.fusion_dim
    summary['seed'] = seed

    summary_path = os.path.join(output_dir, 'summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    csv_path = os.path.join(output_dir, 'fold_predictions.csv')
    fieldnames = [
        'repeat', 'fold', 'patient_id', 'true_subtype', 'tdrest_pred',
        'tdrest_prob_rest', 'pigd_mixed_pred', 'pigd_mixed_prob_mixed', 'final_pred',
    ]
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(pred_rows)

    print('\n' + '=' * 60)
    print('Dual-stream subtype hierarchy summary:')
    print(f"  tdrest_acc         = {summary['tdrest_acc_mean']:.4f} +/- {summary['tdrest_acc_std']:.4f}")
    print(f"  pigd_mixed_acc     = {summary['pigd_mixed_acc_mean']:.4f} +/- {summary['pigd_mixed_acc_std']:.4f}")
    print(f"  hierarchy_3cls_acc = {summary['subtype_hierarchy_acc_mean']:.4f} +/- {summary['subtype_hierarchy_acc_std']:.4f}")
    print(f"  summary -> {summary_path}")
    print(f"  predictions -> {csv_path}")
    print('=' * 60)
    return summary
