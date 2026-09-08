"""
subtype_hierarchy.py - PD subtype 层级三分类公平重算公共工具

参考 files/subtype_hierarchy_revalidation_guide.md，提供所有 baseline 共用的
标签定义 / patient-level fold 构造 / 层级合成 / 指标计算 / 15-fold 汇总逻辑，
避免各 baseline 各自实现导致标签方向不一致。

核心约定
--------
原始三分类标签:  TD=0, PIGD=1, mixed=2
子任务 1 (tdrest):      TD=0, rest(PIGD+mixed)=1
子任务 2 (pigd_mixed):  PIGD=0, mixed=1, TD=exclude
层级合成: gate 判 TD -> TD; gate 判 rest -> 进入 PIGD-vs-Mixed 输出 PIGD/mixed
"""
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

# 原始 subtype 字符串 -> 三分类标签
SUBTYPE_TO_3CLS = {'TD': 0, 'PIGD': 1, 'mixed': 2}
CLS3_NAMES = ['TD', 'PIGD', 'mixed']


# ==================== 标签定义 ====================

def subtype_tdrest_label(y3: int) -> int:
    """TD-vs-rest 二分类标签: TD=0, rest=1。"""
    return 0 if y3 == 0 else 1


def subtype_pigd_mixed_label(y3: int) -> Optional[int]:
    """PIGD-vs-Mixed 二分类标签: PIGD=0, mixed=1, TD=None(exclude)。"""
    if y3 == 1:
        return 0
    if y3 == 2:
        return 1
    return None


def make_subtype_binary_labels(records: List[Dict], task: str) -> List[Optional[int]]:
    """
    为一组患者记录生成指定子任务的二分类标签。

    Args:
        records: 患者记录列表，每个元素需含 'subtype3' (0/1/2) 或 'subtype' 字符串。
        task: 'pd_subtype_tdrest' 或 'pd_subtype_pigd_mixed'。

    Returns:
        与 records 等长的标签列表；pigd_mixed 任务中 TD 患者对应 None。
    """
    labels: List[Optional[int]] = []
    for rec in records:
        y3 = _record_subtype3(rec)
        if task == 'pd_subtype_tdrest':
            labels.append(subtype_tdrest_label(y3))
        elif task == 'pd_subtype_pigd_mixed':
            labels.append(subtype_pigd_mixed_label(y3))
        else:
            raise ValueError(f"未知子任务: {task}")
    return labels


def _record_subtype3(rec: Dict) -> int:
    if 'subtype3' in rec and rec['subtype3'] is not None:
        return int(rec['subtype3'])
    return SUBTYPE_TO_3CLS[rec['subtype']]


# ==================== Fold 构造 ====================

def build_patient_5x3_subtype_folds(records: List[Dict], seed: int = 42,
                                    n_splits: int = 5, repeats: int = 3
                                    ) -> List[Dict]:
    """
    构建 canonical 的 patient-level `n_splits-fold x repeats` subtype fold 表。

    每个 repeat 使用不同 seed (seed, seed+1000, seed+2000)，对全部 subtype 患者
    做 patient-level stratified k-fold。所有子任务复用同一份 full fold。

    Args:
        records: 患者记录列表，每个需含 'patient_id' 与 subtype 信息。
        seed: 基础随机种子。
        n_splits: fold 数 (默认 5)。
        repeats: 重复次数 (默认 3)。

    Returns:
        长度为 n_splits*repeats 的 fold 列表，每个元素:
        {
            'repeat': int, 'fold': int,
            'train_patients_full': [patient_id, ...],
            'val_patients_full': [patient_id, ...],
        }
    """
    pid_to_y3 = {rec['patient_id']: _record_subtype3(rec) for rec in records}
    all_pids = list(pid_to_y3.keys())

    folds: List[Dict] = []
    for repeat in range(repeats):
        rep_seed = seed + repeat * 1000
        assignment = _stratified_patient_fold_assignment(
            all_pids, [pid_to_y3[p] for p in all_pids], n_splits, rep_seed
        )
        for fold_idx in range(n_splits):
            val_pids = [p for p, f in zip(all_pids, assignment) if f == fold_idx]
            train_pids = [p for p, f in zip(all_pids, assignment) if f != fold_idx]
            folds.append({
                'repeat': repeat,
                'fold': fold_idx + 1,
                'train_patients_full': train_pids,
                'val_patients_full': val_pids,
            })
    return folds


def _stratified_patient_fold_assignment(pids: List[str], y3: List[int],
                                        n_splits: int, seed: int) -> List[int]:
    """按 subtype 分层，将每个 patient 分配到 0..n_splits-1 的 fold。"""
    rng = np.random.RandomState(seed)
    label_groups: Dict[int, List[int]] = defaultdict(list)
    for i, y in enumerate(y3):
        label_groups[y].append(i)

    assignment = [0] * len(pids)
    for label in sorted(label_groups.keys()):
        idxs = label_groups[label]
        rng.shuffle(idxs)
        for j, idx in enumerate(idxs):
            assignment[idx] = j % n_splits
    return assignment


# ==================== 层级合成 ====================

def fuse_subtype_predictions(tdrest_pred: int,
                             pigd_mixed_pred: Optional[int]) -> int:
    """
    层级合成最终三分类。

    Returns: 0=TD, 1=PIGD, 2=mixed。
    """
    if tdrest_pred == 0:
        return 0
    if pigd_mixed_pred is None:
        # 理论上 gate 判 rest 必须能拿到 second-stage 预测；缺失时退回 PIGD。
        return 1
    return 1 if pigd_mixed_pred == 0 else 2


# ==================== 指标计算 ====================

def _per_class_recall(true_labels: List[int], pred_labels: List[int],
                      classes: List[int]) -> Dict[str, float]:
    recall: Dict[str, float] = {}
    for c in classes:
        idx = [i for i, t in enumerate(true_labels) if t == c]
        if not idx:
            recall[str(c)] = float('nan')
            continue
        correct = sum(1 for i in idx if pred_labels[i] == c)
        recall[str(c)] = correct / len(idx)
    return recall


def compute_subtype_hierarchy_metrics(rows: List[Dict]) -> Dict:
    """
    基于一个 fold 的 patient-level 配对预测计算层级三分类指标。

    Args:
        rows: 该 fold 的 full val 患者列表，每个元素含:
            'true_subtype'    : 0/1/2
            'tdrest_pred'     : 0/1
            'pigd_mixed_pred' : 0/1 或 None
            'final_pred'      : 0/1/2 (可不传，自动合成)

    Returns:
        指标字典 (字段与 guide §5 对齐)。
    """
    n_full = len(rows)
    true3 = [r['true_subtype'] for r in rows]
    final = [r.get('final_pred',
                   fuse_subtype_predictions(r['tdrest_pred'], r.get('pigd_mixed_pred')))
             for r in rows]

    # tdrest: 全 full val fold
    tdrest_true = [subtype_tdrest_label(y) for y in true3]
    tdrest_pred = [r['tdrest_pred'] for r in rows]
    tdrest_acc = _safe_mean([int(p == t) for p, t in zip(tdrest_pred, tdrest_true)])
    tdrest_per_class = _per_class_recall(tdrest_true, tdrest_pred, [0, 1])

    # pigd_mixed: 仅 non-TD 子集
    pm_idx = [i for i, y in enumerate(true3) if y in (1, 2)]
    pm_true = [subtype_pigd_mixed_label(true3[i]) for i in pm_idx]
    pm_pred = [rows[i].get('pigd_mixed_pred') for i in pm_idx]
    pm_valid = [(p, t) for p, t in zip(pm_pred, pm_true) if p is not None]
    pigd_mixed_acc = _safe_mean([int(p == t) for p, t in pm_valid])
    pigd_mixed_per_class = _per_class_recall(
        [t for _, t in pm_valid], [p for p, _ in pm_valid], [0, 1]
    )

    # 层级三分类
    hierarchy_acc = _safe_mean([int(f == t) for f, t in zip(final, true3)])
    hierarchy_per_class = _per_class_recall(true3, final, [0, 1, 2])
    present = [v for v in hierarchy_per_class.values() if not _is_nan(v)]
    balanced_acc = float(np.mean(present)) if present else float('nan')

    cm = [[0, 0, 0] for _ in range(3)]
    for t, f in zip(true3, final):
        cm[t][f] += 1

    n_td = sum(1 for y in true3 if y == 0)
    return {
        'n_val_full': n_full,
        'n_val_td': n_td,
        'n_val_rest': n_full - n_td,
        'n_val_pigd_mixed': len(pm_idx),
        'tdrest_acc': tdrest_acc,
        'tdrest_per_class_acc': tdrest_per_class,
        'pigd_mixed_acc': pigd_mixed_acc,
        'pigd_mixed_per_class_acc': pigd_mixed_per_class,
        'subtype_hierarchy_acc': hierarchy_acc,
        'subtype_hierarchy_balanced_acc': balanced_acc,
        'subtype_hierarchy_per_class_acc': hierarchy_per_class,
        'confusion_matrix_3cls': cm,
    }


def summarize_15fold_results(fold_results: List[Dict],
                             n_splits: int = 5, repeats: int = 3) -> Dict:
    """对所有 fold 指标做 mean/std 汇总 (guide §5 summary 字段)。"""
    def _ms(key: str) -> Tuple[float, float]:
        vals = [fr[key] for fr in fold_results if not _is_nan(fr.get(key))]
        if not vals:
            return float('nan'), float('nan')
        return float(np.mean(vals)), float(np.std(vals))

    tdrest_mean, tdrest_std = _ms('tdrest_acc')
    pm_mean, pm_std = _ms('pigd_mixed_acc')
    hier_mean, hier_std = _ms('subtype_hierarchy_acc')
    bal_mean, bal_std = _ms('subtype_hierarchy_balanced_acc')

    return {
        'protocol': 'patient_5fold_x3_subtype_hierarchy',
        'num_folds': n_splits,
        'kfold_repeats': repeats,
        'evaluated_folds': len(fold_results),
        'tdrest_acc_mean': tdrest_mean,
        'tdrest_acc_std': tdrest_std,
        'pigd_mixed_acc_mean': pm_mean,
        'pigd_mixed_acc_std': pm_std,
        'subtype_hierarchy_acc_mean': hier_mean,
        'subtype_hierarchy_acc_std': hier_std,
        'subtype_hierarchy_balanced_acc_mean': bal_mean,
        'subtype_hierarchy_balanced_acc_std': bal_std,
        'fold_results': fold_results,
    }


# ==================== 工具 ====================

def _safe_mean(values: List[float]) -> float:
    return float(np.mean(values)) if values else float('nan')


def _is_nan(v) -> bool:
    try:
        return isinstance(v, float) and np.isnan(v)
    except (TypeError, ValueError):
        return False
