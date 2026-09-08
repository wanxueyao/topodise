"""
subtype_hierarchy_runner.py - 基础模型 subtype 层级三分类公平重算 runner

参考 files/subtype_hierarchy_revalidation_guide.md (§3/§4/§6.2)。

流程
----
1. 取 63 个 PD subtype 患者，构建 canonical patient-level 5-fold x 3 repeats fold 表。
2. 每个 full fold 训练两个二分类子任务:
   - pd_subtype_tdrest      (TD vs rest，全 subtype 患者)
   - pd_subtype_pigd_mixed  (PIGD vs mixed，训练排除 TD)
3. 对 full val fold 做 patient-level 推理（图像概率均值聚合）。
4. gate-rest 患者进入第二层，按层级规则合成最终三分类。
5. 输出 summary.json 与 fold_predictions.csv。

与现有 basic_classifier 复用方式
--------------------------------
通过覆盖 PDDataset.subtype_labels，并使用 TASK_CONFIG 中新增的两个二分类任务
(label_key='subtype', num_classes=2) 直接复用 PDClassifier.train。
"""
import csv
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.basic_classifier import PDClassifier
from core.data_loader import PDDataset, load_and_filter_dataset, SUBTYPE_MAP
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

BASE_PATH = 'datasets'


def _relabel_subtype(dataset: PDDataset, patients: List[Dict], task: str) -> None:
    """根据子任务覆盖 dataset.subtype_labels（pigd_mixed 的 TD 置 -1 以被 mask）。"""
    new_labels: List[int] = []
    for i in range(len(dataset.samples)):
        y3 = SUBTYPE_TO_3CLS[patients[dataset.patient_ids[i]]['subtype']]
        if task == 'pd_subtype_tdrest':
            new_labels.append(subtype_tdrest_label(y3))
        else:  # pd_subtype_pigd_mixed
            lab = subtype_pigd_mixed_label(y3)
            new_labels.append(-1 if lab is None else lab)
    dataset.subtype_labels = new_labels


def _train_binary_subtask(classifier: PDClassifier, train_patients: List[Dict],
                          val_patients: List[Dict], task: str, fold_idx: int):
    """训练单个二分类子任务，返回训练好的模型。"""
    cfg = classifier.config
    train_ds = PDDataset(train_patients, cfg.ARCHITECTURE,
                         classifier.train_transform, base_path=BASE_PATH)
    val_ds = PDDataset(val_patients, cfg.ARCHITECTURE,
                       classifier.val_transform, base_path=BASE_PATH)
    _relabel_subtype(train_ds, train_patients, task)
    _relabel_subtype(val_ds, val_patients, task)
    result = classifier.train(train_ds, val_ds, task=task,
                              fold_idx=fold_idx, suffix=f"_{task}")
    return result['model']


def _infer_patient_level(classifier: PDClassifier, model, patients: List[Dict]
                         ) -> Tuple[Dict[str, int], Dict[str, float]]:
    """
    对一组患者做 patient-level 推理。

    对每个患者的所有图像概率取均值，argmax 作为患者预测。

    Returns:
        (pred_by_pid, prob_class1_by_pid)
    """
    cfg = classifier.config
    dataset = PDDataset(patients, cfg.ARCHITECTURE,
                        classifier.val_transform, base_path=BASE_PATH)
    loader = DataLoader(dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=0)

    device = cfg.DEVICE
    model.eval()
    all_probs: List[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            if isinstance(batch[0], (tuple, list)):
                (central, whorl), _, _, _ = batch
                outputs = model(central.to(device), whorl.to(device))
            else:
                img, _, _, _ = batch
                outputs = model(img.to(device))
            probs = torch.softmax(outputs, dim=1)
            all_probs.append(probs.cpu().numpy())

    if not all_probs:
        return {}, {}

    probs = np.concatenate(all_probs, axis=0)  # [N, 2]，与 dataset 顺序对齐

    # 按患者聚合（patient_ids 为 patients 列表内的下标）
    sum_probs: Dict[int, np.ndarray] = defaultdict(lambda: np.zeros(probs.shape[1]))
    counts: Dict[int, int] = defaultdict(int)
    for i, pidx in enumerate(dataset.patient_ids):
        sum_probs[pidx] += probs[i]
        counts[pidx] += 1

    pred_by_pid: Dict[str, int] = {}
    prob_class1_by_pid: Dict[str, float] = {}
    for pidx, total in sum_probs.items():
        mean_prob = total / max(1, counts[pidx])
        patient_id = patients[pidx]['patient_id']
        pred_by_pid[patient_id] = int(np.argmax(mean_prob))
        prob_class1_by_pid[patient_id] = float(mean_prob[1])
    return pred_by_pid, prob_class1_by_pid


def run_subtype_hierarchy_experiment(config, output_dir: str,
                                     seed: int = 42, n_splits: int = 5,
                                     repeats: int = 3) -> Dict:
    """
    运行 subtype 层级三分类 5-fold x 3 重算实验。

    Args:
        config: ExperimentConfig（提供 backbone / 训练超参）。
        output_dir: 输出目录。
        seed: 基础随机种子。
        n_splits: fold 数。
        repeats: 重复次数。

    Returns:
        summary 字典（同时写入 summary.json）。
    """
    os.makedirs(output_dir, exist_ok=True)
    set_seed(seed)

    patients_all = load_and_filter_dataset(config.DATASET_JSON, 'pd_all')
    subtype_patients = [p for p in patients_all
                        if p['group'] == 'pd' and p.get('subtype') in SUBTYPE_MAP]
    pid_to_patient = {p['patient_id']: p for p in subtype_patients}
    records = [{'patient_id': p['patient_id'], 'subtype': p['subtype']}
               for p in subtype_patients]

    folds = build_patient_5x3_subtype_folds(records, seed=seed,
                                            n_splits=n_splits, repeats=repeats)

    print("=" * 60)
    print("Subtype 层级三分类公平重算 (5-fold x 3)")
    print(f"subtype 患者数: {len(subtype_patients)} | "
          f"backbone: {config.BACKBONE} | fold 总数: {len(folds)}")
    print("=" * 60)

    classifier = PDClassifier(config)
    fold_results: List[Dict] = []
    pred_rows: List[Dict] = []

    for global_idx, fold in enumerate(folds, start=1):
        repeat, fold_no = fold['repeat'], fold['fold']
        train_patients = [pid_to_patient[pid] for pid in fold['train_patients_full']]
        val_patients = [pid_to_patient[pid] for pid in fold['val_patients_full']]

        print(f"\n--- repeat {repeat} fold {fold_no} "
              f"({global_idx}/{len(folds)}) | "
              f"train={len(train_patients)} val={len(val_patients)} ---")

        # 第一层: TD vs rest（全部 subtype 患者）
        tdrest_model = _train_binary_subtask(
            classifier, train_patients, val_patients,
            'pd_subtype_tdrest', global_idx)
        tdrest_pred, tdrest_prob = _infer_patient_level(
            classifier, tdrest_model, val_patients)

        # 第二层: PIGD vs mixed（训练排除 TD；推理覆盖 full val fold）
        pm_train_patients = [p for p in train_patients
                             if p['subtype'] in ('PIGD', 'mixed')]
        pm_val_patients = [p for p in val_patients
                           if p['subtype'] in ('PIGD', 'mixed')]
        pm_model = _train_binary_subtask(
            classifier, pm_train_patients, pm_val_patients,
            'pd_subtype_pigd_mixed', global_idx)
        # 推理在 full val fold 上进行，让 gate 误判为 rest 的 TD 患者也进入第二层
        pm_pred, pm_prob = _infer_patient_level(
            classifier, pm_model, val_patients)

        rows: List[Dict] = []
        for pid in fold['val_patients_full']:
            y3 = SUBTYPE_TO_3CLS[pid_to_patient[pid]['subtype']]
            g = tdrest_pred.get(pid, 1)
            pm = pm_pred.get(pid)
            final = fuse_subtype_predictions(g, pm)
            rows.append({
                'true_subtype': y3,
                'tdrest_pred': g,
                'pigd_mixed_pred': pm,
                'final_pred': final,
            })
            pred_rows.append({
                'repeat': repeat,
                'fold': fold_no,
                'patient_id': pid,
                'true_subtype': y3,
                'tdrest_pred': g,
                'tdrest_prob_rest': round(tdrest_prob.get(pid, float('nan')), 4),
                'pigd_mixed_pred': '' if pm is None else pm,
                'pigd_mixed_prob_mixed': round(pm_prob.get(pid, float('nan')), 4),
                'final_pred': final,
            })

        metrics = compute_subtype_hierarchy_metrics(rows)
        metrics['fold'] = fold_no
        metrics['repeat'] = repeat
        fold_results.append(metrics)
        print(f"  tdrest_acc={metrics['tdrest_acc']:.4f} | "
              f"pigd_mixed_acc={metrics['pigd_mixed_acc']:.4f} | "
              f"hierarchy_acc={metrics['subtype_hierarchy_acc']:.4f}")

    summary = summarize_15fold_results(fold_results, n_splits=n_splits, repeats=repeats)
    summary['backbone'] = config.BACKBONE
    summary['seed'] = seed

    summary_path = os.path.join(output_dir, 'summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    csv_path = os.path.join(output_dir, 'fold_predictions.csv')
    fieldnames = ['repeat', 'fold', 'patient_id', 'true_subtype', 'tdrest_pred',
                  'tdrest_prob_rest', 'pigd_mixed_pred', 'pigd_mixed_prob_mixed',
                  'final_pred']
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(pred_rows)

    print("\n" + "=" * 60)
    print("Subtype 层级三分类汇总:")
    print(f"  tdrest_acc           = {summary['tdrest_acc_mean']:.4f} "
          f"± {summary['tdrest_acc_std']:.4f}")
    print(f"  pigd_mixed_acc       = {summary['pigd_mixed_acc_mean']:.4f} "
          f"± {summary['pigd_mixed_acc_std']:.4f}")
    print(f"  hierarchy_3cls_acc   = {summary['subtype_hierarchy_acc_mean']:.4f} "
          f"± {summary['subtype_hierarchy_acc_std']:.4f}")
    print(f"  hierarchy_balanced   = {summary['subtype_hierarchy_balanced_acc_mean']:.4f} "
          f"± {summary['subtype_hierarchy_balanced_acc_std']:.4f}")
    print(f"  summary -> {summary_path}")
    print(f"  predictions -> {csv_path}")
    print("=" * 60)

    return summary
