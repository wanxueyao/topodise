"""
unet_subtype_hierarchy_runner.py - CNN-on-UNet subtype 层级三分类公平重算

参考 files/subtype_hierarchy_revalidation_guide.md：
- TD-vs-rest gate: TD=0, rest=1
- PIGD-vs-Mixed: PIGD=0, mixed=1，训练排除 TD
- 两个二分类 CNN head，patient-level 概率均值聚合后层级融合
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
from torch.utils.data import DataLoader

_THIS_FILE = Path(__file__).resolve()
CODES_ROOT = _THIS_FILE.parents[2]
if str(CODES_ROOT) not in sys.path:
    sys.path.insert(0, str(CODES_ROOT))

from core.data_loader import PDDataset, load_and_filter_dataset
from experiments.gnn_experiments.main import (
    GNNExperimentConfig,
    LIGHT_BACKBONES,
    _build_basic_classifier,
    _build_light_classifier,
)
from utils.misc import set_seed
from utils.subtype_hierarchy import (
    SUBTYPE_TO_3CLS,
    build_patient_5x3_subtype_folds,
    compute_subtype_hierarchy_metrics,
    fuse_subtype_predictions,
    summarize_15fold_results,
)


def _resolve_dataset_json(path: str) -> str:
    if os.path.isabs(path):
        return path
    return str(CODES_ROOT / path)


def _build_classifier(config: GNNExperimentConfig):
    if config.BACKBONE in LIGHT_BACKBONES:
        return _build_light_classifier(config)
    return _build_basic_classifier(config)


def _patient_records(patients: List[Dict]) -> List[Dict]:
    return [
        {'patient_id': p['patient_id'], 'subtype': p['subtype']}
        for p in patients
        if p.get('group') == 'pd' and p.get('subtype') in SUBTYPE_TO_3CLS
    ]


def _subset_patients(pids: List[str], pid_to_patient: Dict[str, Dict]) -> List[Dict]:
    return [pid_to_patient[pid] for pid in pids]


def _make_dataset(
    classifier,
    patients: List[Dict],
    config: GNNExperimentConfig,
    training: bool = False,
) -> PDDataset:
    return PDDataset(
        patients,
        config.ARCHITECTURE,
        classifier.train_transform if training else classifier.val_transform,
        base_path=str(CODES_ROOT / 'datasets'),
    )


def _train_one_subtask(
    config: GNNExperimentConfig,
    train_patients: List[Dict],
    val_patients: List[Dict],
    task: str,
    fold_idx: int,
):
    classifier = _build_classifier(config)
    train_ds = _make_dataset(classifier, train_patients, config, training=True)
    val_ds = _make_dataset(classifier, val_patients, config, training=False)
    result = classifier.train(
        train_ds,
        val_ds,
        task,
        fold_idx=fold_idx,
        suffix='_subtype_hierarchy',
    )
    return classifier, result['model']


@torch.no_grad()
def _infer_patient_level(
    model: torch.nn.Module,
    classifier,
    patients: List[Dict],
    config: GNNExperimentConfig,
) -> Tuple[Dict[str, int], Dict[str, float]]:
    dataset = _make_dataset(classifier, patients, config, training=False)
    local_pid_to_real = {
        local_idx: patient['patient_id']
        for local_idx, patient in enumerate(patients)
    }
    sample_pids = [local_pid_to_real[int(pid)] for pid in dataset.patient_ids]
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=(config.DEVICE != 'cpu'),
    )

    model.eval()
    sums: Dict[str, np.ndarray] = defaultdict(lambda: np.zeros(2, dtype=np.float64))
    counts: Dict[str, int] = defaultdict(int)
    offset = 0

    for batch in loader:
        is_dual = isinstance(batch[0], (tuple, list))
        if is_dual:
            (central, whorl), _pd, _stage, subtype = batch
            outputs = model(central.to(config.DEVICE), whorl.to(config.DEVICE))
            batch_size = int(subtype.size(0))
        else:
            images, _pd, _stage, subtype = batch
            outputs = model(images.to(config.DEVICE))
            batch_size = int(subtype.size(0))

        probs = torch.softmax(outputs, dim=1).detach().cpu().numpy()
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


def run_unet_subtype_hierarchy_experiment(
    config: GNNExperimentConfig,
    output_dir: str,
    seed: int = 42,
    n_splits: int = 5,
    repeats: int = 3,
) -> Dict:
    """运行 CNN-on-UNet subtype 层级三分类 5-fold x 3 重算实验。"""
    os.makedirs(output_dir, exist_ok=True)
    set_seed(seed)

    cfg = deepcopy(config)
    cfg.DATASET_MODE = 'pd_all'
    cfg.KFOLD_MODE = 'patient'
    cfg.NUM_FOLDS = n_splits
    cfg.SEED = seed

    dataset_json = _resolve_dataset_json(cfg.DATASET_JSON)
    all_patients = load_and_filter_dataset(dataset_json, cfg.DATASET_MODE)
    records = _patient_records(all_patients)
    pid_to_patient = {
        p['patient_id']: p for p in all_patients
        if p.get('group') == 'pd' and p.get('subtype') in SUBTYPE_TO_3CLS
    }
    folds = build_patient_5x3_subtype_folds(
        records, seed=seed, n_splits=n_splits, repeats=repeats
    )

    print('=' * 60)
    print(f'CNN-on-UNet Subtype 层级三分类公平重算 ({n_splits}-fold x {repeats}, 仅 subtype 任务)')
    print(f"subtype 患者数: {len(records)} | backbone={cfg.BACKBONE} | "
          f"folds={len(folds)} | device={cfg.DEVICE}")
    print('=' * 60)

    fold_results: List[Dict] = []
    pred_rows: List[Dict] = []

    for global_idx, fold in enumerate(folds, start=1):
        repeat, fold_no = fold['repeat'], fold['fold']
        train_patients = _subset_patients(fold['train_patients_full'], pid_to_patient)
        val_patients = _subset_patients(fold['val_patients_full'], pid_to_patient)
        print(
            f"\n--- repeat {repeat} fold {fold_no} ({global_idx}/{len(folds)}) | "
            f"train={len(train_patients)} val={len(val_patients)} ---",
            flush=True,
        )

        tdrest_classifier, tdrest_model = _train_one_subtask(
            cfg, train_patients, val_patients,
            'pd_subtype_tdrest', fold_idx=global_idx,
        )
        tdrest_pred, tdrest_prob = _infer_patient_level(
            tdrest_model, tdrest_classifier, val_patients, cfg
        )
        del tdrest_model, tdrest_classifier
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        pm_train_patients = [p for p in train_patients if p['subtype'] in ('PIGD', 'mixed')]
        pm_val_patients = [p for p in val_patients if p['subtype'] in ('PIGD', 'mixed')]
        pm_classifier, pm_model = _train_one_subtask(
            cfg, pm_train_patients, pm_val_patients,
            'pd_subtype_pigd_mixed', fold_idx=global_idx,
        )
        pm_pred, pm_prob = _infer_patient_level(
            pm_model, pm_classifier, val_patients, cfg
        )
        del pm_model, pm_classifier
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
    summary['backbone'] = cfg.BACKBONE
    summary['architecture'] = cfg.ARCHITECTURE
    summary['dataset_json'] = cfg.DATASET_JSON
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
    print('CNN-on-UNet Subtype 层级三分类汇总:')
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