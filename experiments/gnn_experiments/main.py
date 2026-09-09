"""gnn 实验实现：训练、评估及单组执行；参数与批量控制见 batch_experiments.py。"""
from __future__ import annotations

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from experiments._execution import single_main
    single_main("experiments.gnn_experiments.main")
    raise SystemExit(0)

from utils.td_rest import resolve_baseline_tasks, td_rest_rows, td_rest_fold, save_td_rest_results
import os
import sys
import json
import logging
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any
import numpy as np
import torch
from core.data_loader import load_and_filter_dataset, stratified_kfold, SUBTYPE_MAP
from utils.patient_outputs import save_fold_patient_outputs
from gnn.classifier import GNNClassifier, GNNClassifierConfig
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Tuple
import torch.nn.functional as F
from torch.utils.data import DataLoader
from gnn.graph_dataset import NerveGraphDataset, collate_fn
from gnn.models import GNNMultiTaskModel
from gnn.trainer import batch_to_device
from utils.misc import set_seed
from utils.subtype_hierarchy import SUBTYPE_TO_3CLS, build_patient_5x3_subtype_folds, subtype_pigd_mixed_label, subtype_tdrest_label




# ─── 路径配置 ─────────────────────────────────────────────────────────────────
_CODES_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _CODES_DIR not in sys.path:
    sys.path.insert(0, _CODES_DIR)



# ─── 实验配置 ─────────────────────────────────────────────────────────────────

@dataclass
class GNNExperimentConfig:
    """GNN 单次实验的全部超参数。"""
    name: str = 'gnn_experiment'

    # ── 数据 ────────────────────────────────────────────────────────────────────
    dataset_json: str = 'datasets/pd_dataset_unet_en/patients_dataset_en.json'
    dataset_group: str = 'pd_control'   # 见 DATASET_GROUPS
    num_folds: int = 2
    kfold_mode: str = 'patient'
    graph_cache_dir: Optional[str] = None   # None = 自动使用 results/graph_cache

    # ── 模型 ────────────────────────────────────────────────────────────────────
    gnn_type: str = 'gat'           # 'gat'
    hidden_dim: int = 64
    num_layers: int = 2
    num_heads: int = 4
    shared_encoder: bool = False

    # ── 训练 ────────────────────────────────────────────────────────────────────
    lr: float = 1e-3
    weight_decay: float = 1e-4
    optimizer: str = 'adam'
    lr_scheduler: str = 'cosine'
    warmup_epochs: int = 5
    epochs: int = 50
    patience: int = 10
    batch_size: int = 16
    dropout: float = 0.2
    classifier_dropout: float = 0.3

    # ── 损失 ────────────────────────────────────────────────────────────────────
    pd_loss_weight: float = 1.0
    stage_loss_weight: float = 0.5
    subtype_loss_weight: float = 0.5
    label_smoothing: float = 0.0

    # ── 其他 ────────────────────────────────────────────────────────────────────
    primary_task: str = 'pd'        # 'pd' | 'stage' | 'subtype'
    num_workers: int = 0
    seed: int = 42
    reuse_checkpoints: bool = True


# ─── 日志设置 ─────────────────────────────────────────────────────────────────

def _setup_logging(output_dir: str, name: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, f'{name}.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[
            logging.FileHandler(log_path, mode='w', encoding='utf-8'),
            logging.StreamHandler(),
        ],
        force=True,
    )


# ─── 指标汇总 ─────────────────────────────────────────────────────────────────

def _summarize(
    fold_results: List[Dict],
    config: GNNExperimentConfig,
    tasks: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """计算 K-fold 均值 ± 标准差汇总。"""
    selected_tasks = tasks or ['pd', 'stage', 'subtype']
    metric_keys = []
    if 'pd' in selected_tasks:
        metric_keys.extend(['pd_acc', 'pd_patient_acc', 'pd_balanced_acc', 'pd_f1', 'pd_auc'])
    if 'stage' in selected_tasks:
        metric_keys.extend(['stage_acc', 'stage_patient_acc', 'stage_balanced_acc', 'stage_f1', 'stage_auc'])
    if 'subtype' in selected_tasks:
        metric_keys.extend(['subtype_acc', 'subtype_patient_acc', 'subtype_balanced_acc', 'subtype_f1', 'subtype_auc'])
    agg: Dict[str, float] = {}
    for k in metric_keys:
        vals = [r[k] for r in fold_results if k in r]
        if vals:
            agg[f'{k}_mean'] = float(np.mean(vals))
            agg[f'{k}_std']  = float(np.std(vals))

    return {
        'name': config.name,
        'config': asdict(config),
        'num_folds': len(fold_results),
        'fold_results': fold_results,
        **agg,
    }


def _log_summary(summary: Dict, logger: logging.Logger) -> None:
    def fmt(mk, sk):
        return f"{summary.get(mk, 0):.4f} ± {summary.get(sk, 0):.4f}"

    logger.info(
        f"PD     | acc {fmt('pd_acc_mean', 'pd_acc_std')} | "
        f"pat {fmt('pd_patient_acc_mean', 'pd_patient_acc_std')} | "
        f"bacc {fmt('pd_balanced_acc_mean', 'pd_balanced_acc_std')} | "
        f"f1 {fmt('pd_f1_mean', 'pd_f1_std')} | "
        f"auc {fmt('pd_auc_mean', 'pd_auc_std')}"
    )
    logger.info(
        f"Stage  | acc {fmt('stage_acc_mean', 'stage_acc_std')} | "
        f"pat {fmt('stage_patient_acc_mean', 'stage_patient_acc_std')} | "
        f"bacc {fmt('stage_balanced_acc_mean', 'stage_balanced_acc_std')} | "
        f"f1 {fmt('stage_f1_mean', 'stage_f1_std')}"
    )
    logger.info(
        f"Subtype| acc {fmt('subtype_acc_mean', 'subtype_acc_std')} | "
        f"pat {fmt('subtype_patient_acc_mean', 'subtype_patient_acc_std')} | "
        f"bacc {fmt('subtype_balanced_acc_mean', 'subtype_balanced_acc_std')} | "
        f"f1 {fmt('subtype_f1_mean', 'subtype_f1_std')}"
    )


# ─── 主入口 ───────────────────────────────────────────────────────────────────

def run_single_experiment(
    config: GNNExperimentConfig,
    output_dir: str = 'experiment_records',
    tasks: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    运行单次 GNN 实验 (K-fold)。

    Args:
        config:     GNNExperimentConfig
        output_dir: 实验结果保存根目录
        tasks:      要汇总和保存 patient logits 的任务列表；None 表示全部任务

    Returns:
        summary: 包含各 fold 指标及均值 / 标准差的字典
    """
    # ── 准备目录与日志 ───────────────────────────────────────────────────────────
    exp_dir = os.path.join(output_dir, config.name)
    os.makedirs(exp_dir, exist_ok=True)
    _setup_logging(exp_dir, config.name)
    logger = logging.getLogger(__name__)

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    selected_tasks = tasks or ['pd', 'stage', 'subtype']

    logger.info(f"=== GNN 实验: {config.name} ===")
    logger.info(f"配置: {asdict(config)}")

    # ── 加载患者数据 ─────────────────────────────────────────────────────────────
    json_path = os.path.join(_CODES_DIR, config.dataset_json)
    patients = load_and_filter_dataset(json_path, config.dataset_group)
    logger.info(f"患者总数: {len(patients)}  (group={config.dataset_group})")

    # ── 图缓存目录（所有实验共享，避免重复构建）────────────────────────────────────
    cache_dir = config.graph_cache_dir
    if cache_dir is None:
        # 存放在 results/graph_cache 下，与数据集目录隔离，实验间共享
        cache_dir = os.path.join(_CODES_DIR, 'results', 'graph_cache')
    logger.info(f"图缓存目录: {cache_dir}")

    # ── 提取患者级标签（用于 stratified kfold）───────────────────────────────────
    # 每个 patient 当作一个"样本"，patient_ids[i] = i 使得 patient kfold 直接返回患者索引
    pd_labels      = [1 if p['group'] == 'pd' else 0 for p in patients]
    stage_labels   = [int(p.get('stage', 0)) for p in patients]
    subtype_labels = [SUBTYPE_MAP.get(p.get('subtype') or '', -1) for p in patients]
    patient_ids    = list(range(len(patients)))   # 每人唯一索引

    # ── K-fold 划分 ──────────────────────────────────────────────────────────────
    folds = list(stratified_kfold(
        pd_labels=pd_labels,
        stage_labels=stage_labels,
        subtype_labels=subtype_labels,
        n_splits=config.num_folds,
        kfold_mode=config.kfold_mode,
        patient_ids=patient_ids,
    ))
    logger.info(f"K-fold: {config.num_folds} folds (mode={config.kfold_mode})")

    # ── 分类器配置 ───────────────────────────────────────────────────────────────
    clf_cfg = GNNClassifierConfig(
        gnn_type=config.gnn_type,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        shared_encoder=config.shared_encoder,
        lr=config.lr,
        weight_decay=config.weight_decay,
        optimizer=config.optimizer,
        lr_scheduler=config.lr_scheduler,
        warmup_epochs=config.warmup_epochs,
        epochs=config.epochs,
        patience=config.patience,
        batch_size=config.batch_size,
        dropout=config.dropout,
        classifier_dropout=config.classifier_dropout,
        pd_loss_weight=config.pd_loss_weight,
        stage_loss_weight=config.stage_loss_weight,
        subtype_loss_weight=config.subtype_loss_weight,
        label_smoothing=config.label_smoothing,
        primary_task=config.primary_task,
        graph_cache_dir=cache_dir,
        num_workers=config.num_workers,
    )

    base_path = os.path.join(_CODES_DIR, 'datasets')

    # ── 运行 K-fold ──────────────────────────────────────────────────────────────
    fold_results = []
    for fold_idx, (train_indices, val_indices) in enumerate(folds):
        logger.info(f"\n── Fold {fold_idx + 1}/{config.num_folds} "
                    f"(train={len(train_indices)}, val={len(val_indices)}) ──")

        train_patients = [patients[i] for i in train_indices]
        val_patients   = [patients[i] for i in val_indices]

        clf = GNNClassifier(config=clf_cfg)
        checkpoint_path = os.path.join(exp_dir, f'fold_{fold_idx + 1}_best_model.pt')
        result = clf.train_one_fold(
            train_patients, val_patients, base_path, fold_idx=fold_idx + 1,
            checkpoint_path=checkpoint_path,
            reuse_checkpoint=config.reuse_checkpoints,
        )
        predictions = result.pop('predictions', {})
        if predictions:
            for task in selected_tasks:
                task_pred = predictions.get(task, {})
                patient_result = save_fold_patient_outputs(
                    exp_dir, task, '', fold_idx + 1,
                    predictions.get('patient_ids', []),
                    task_pred.get('labels', []),
                    probs=task_pred.get('probs'),
                    logits=task_pred.get('logits'),
                )
                result[f'{task}_patient_acc'] = patient_result['patient_acc']
                result[f'{task}_patient_logits_json'] = patient_result['patient_logits_json']
                result[f'{task}_patient_logits_csv'] = patient_result['patient_logits_csv']
            keep_prefixes = tuple(f'{task}_' for task in selected_tasks)
            result = {k: v for k, v in result.items() if k == 'fold' or k.startswith(keep_prefixes)}
        fold_results.append(result)

        # 保存单 fold 结果
        fold_path = os.path.join(exp_dir, f'fold_{fold_idx + 1}_result.json')
        with open(fold_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    # ── 汇总 ─────────────────────────────────────────────────────────────────────
    summary = _summarize(fold_results, config, tasks=selected_tasks)
    summary_path = os.path.join(exp_dir, 'summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info(f"\n=== {config.name} 完成 ===")
    _log_summary(summary, logger)

    return summary


# ─── 命令行入口 ───────────────────────────────────────────────────────────────


def build_paper_config(exp, device="cpu", num_workers=0):
    """Translate a selected paper recipe without changing its hyperparameters."""
    cfg = GNNExperimentConfig()
    cfg.gnn_type = exp.get('gnn_type', cfg.gnn_type)
    cfg.hidden_dim = exp.get('hidden_dim', cfg.hidden_dim)
    cfg.num_layers = exp.get('num_layers', cfg.num_layers)
    cfg.num_heads = exp.get('num_heads', cfg.num_heads)
    cfg.shared_encoder = exp.get('shared_encoder', cfg.shared_encoder)
    cfg.lr = exp.get('lr', cfg.lr)
    cfg.weight_decay = exp.get('weight_decay', cfg.weight_decay)
    cfg.optimizer = exp.get('optimizer', cfg.optimizer)
    cfg.lr_scheduler = exp.get('lr_scheduler', cfg.lr_scheduler)
    cfg.warmup_epochs = exp.get('warmup_epochs', cfg.warmup_epochs)
    cfg.epochs = exp.get('epochs', cfg.epochs)
    cfg.patience = exp.get('patience', cfg.patience)
    cfg.batch_size = exp.get('batch_size', cfg.batch_size)
    cfg.dropout = exp.get('dropout', cfg.dropout)
    cfg.classifier_dropout = exp.get('classifier_dropout', cfg.classifier_dropout)
    cfg.pd_loss_weight = exp.get('pd_loss_weight', cfg.pd_loss_weight)
    cfg.stage_loss_weight = exp.get('stage_loss_weight', cfg.stage_loss_weight)
    cfg.subtype_loss_weight = exp.get('subtype_loss_weight', cfg.subtype_loss_weight)
    cfg.label_smoothing = exp.get('label_smoothing', cfg.label_smoothing)
    cfg.dataset_group = exp.get('dataset_group', cfg.dataset_group)
    cfg.num_folds = exp.get('num_folds', cfg.num_folds)
    cfg.kfold_mode = exp.get('kfold_mode', cfg.kfold_mode)
    cfg.primary_task = exp.get('primary_task', cfg.primary_task)
    cfg.seed = exp.get('seed', cfg.seed)
    cfg.name = exp['_group']
    cfg.num_workers = num_workers
    return cfg


def run_paper_experiment(exp, output_dir, device="cpu", num_workers=0):
    exp = resolve_baseline_tasks(exp)
    cfg = build_paper_config(exp, device, num_workers)
    if exp.get('_subtype_td_rest'):
        return run_gnn_subtype_td_rest_experiment(cfg, output_dir=output_dir, seed=cfg.seed,
            n_splits=exp.get('num_folds', 5), repeats=exp.get('kfold_repeats', 3))
    return run_single_experiment(cfg, tasks=exp['_run_tasks'], output_dir=output_dir)

# ==================== SUBTYPE workflow ====================



_subtype_THIS_FILE = Path(__file__).resolve()
CODES_ROOT = _subtype_THIS_FILE.parents[2]
if str(CODES_ROOT) not in sys.path:
    sys.path.insert(0, str(CODES_ROOT))



def _subtype_subtype_binary_label(subtype: str, task: str) -> int:
    y3 = SUBTYPE_TO_3CLS[subtype]
    if task == 'pd_subtype_tdrest':
        return subtype_tdrest_label(y3)
    label = subtype_pigd_mixed_label(y3)
    return -1 if label is None else label


def _subtype_build_dataset_with_patient_ids(
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
                _subtype_subtype_binary_label(subtype_by_pid[pid], task),
            ))
        ds.samples = relabeled

    return ds, sample_pids


def _subtype_build_model(config: GNNExperimentConfig, device: torch.device) -> GNNMultiTaskModel:
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


def _subtype_build_optimizer(config: GNNExperimentConfig, model: torch.nn.Module):
    if config.optimizer == 'adamw':
        return torch.optim.AdamW(model.parameters(), lr=config.lr,
                                 weight_decay=config.weight_decay)
    return torch.optim.Adam(model.parameters(), lr=config.lr,
                            weight_decay=config.weight_decay)


def _subtype_build_scheduler(config: GNNExperimentConfig, optimizer):
    if config.lr_scheduler == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(config.epochs - config.warmup_epochs, 1)
        )
    if config.lr_scheduler == 'step':
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=max(config.epochs // 3, 1), gamma=0.5
        )
    return None


def _subtype_train_one_subtask(
    config: GNNExperimentConfig,
    train_patients: List[Dict],
    val_patients: List[Dict],
    base_path: str,
    cache_dir: str,
    device: torch.device,
    task: str,
) -> GNNMultiTaskModel:
    train_ds, _ = _subtype_build_dataset_with_patient_ids(train_patients, base_path, cache_dir, task)
    val_ds, _ = _subtype_build_dataset_with_patient_ids(val_patients, base_path, cache_dir, task)

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

    model = _subtype_build_model(config, device)
    optimizer = _subtype_build_optimizer(config, model)
    scheduler = _subtype_build_scheduler(config, optimizer)

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

        val_acc = _subtype_eval_binary_acc(model, val_loader, device)
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
def _subtype_eval_binary_acc(model: GNNMultiTaskModel, loader: DataLoader,
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
def _subtype_infer_patient_level(
    model: GNNMultiTaskModel,
    patients: List[Dict],
    config: GNNExperimentConfig,
    base_path: str,
    cache_dir: str,
    device: torch.device,
) -> Tuple[Dict[str, int], Dict[str, float]]:
    ds, sample_pids = _subtype_build_dataset_with_patient_ids(patients, base_path, cache_dir)
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


def run_gnn_subtype_td_rest_experiment(
    config: GNNExperimentConfig,
    output_dir: str,
    seed: int = 42,
    n_splits: int = 5,
    repeats: int = 3,
) -> Dict:
    """运行 GNN subtype TD-rest 二分类 5-fold x 3 重算实验。"""
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
    print('GNN Subtype TD-rest 二分类公平重算 (5-fold x 3, 仅 subtype 任务)')
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

        tdrest_model = _subtype_train_one_subtask(
            cfg, train_patients, val_patients, base_path, cache_dir, device,
            'pd_subtype_tdrest'
        )
        tdrest_pred, tdrest_prob = _subtype_infer_patient_level(
            tdrest_model, val_patients, cfg, base_path, cache_dir, device
        )
        del tdrest_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        rows = td_rest_rows(pid_to_patient, fold['val_patients_full'],
                            tdrest_pred, tdrest_prob, repeat, fold_no)
        fold_results.append(td_rest_fold(rows))
        pred_rows.extend(rows)

    return save_td_rest_results(output_dir, fold_results, pred_rows,
                                n_splits, repeats, seed)
