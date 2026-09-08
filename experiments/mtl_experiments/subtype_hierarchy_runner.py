"""
subtype_hierarchy_runner.py - MTL subtype 层级三分类公平重算 runner

参考 files/subtype_hierarchy_revalidation_guide.md §6.4。

与 basic_experiments 同样的方法，但 Branch B 只跑 subtype 任务：
1. 取 63 个 PD subtype 患者，构建 canonical patient-level 5-fold x 3 fold 表。
2. 每个 full fold 训练两个二分类子任务（共享 MTL encoder + Branch A 辅助任务）:
   - pd_subtype_tdrest      (TD vs rest，全 subtype 患者)
   - pd_subtype_pigd_mixed  (PIGD vs mixed，训练排除 TD)
   每个子任务把 num_subtype_classes 设为 2，只对 subtype 头计算分类损失
   （PD/Stage 权重置 0），Branch A（segmentation / mae / pseudo）保持联合训练，
   即 guide §6.4 的 "MTL representation + hierarchy heads"。
3. 对 full val fold 做 patient-level 推理（图像概率均值聚合）。
4. gate-rest 患者进入第二层，按层级规则合成最终三分类。
5. 输出 summary.json 与 fold_predictions.csv。
"""
from __future__ import annotations

import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

_THIS_FILE = Path(__file__).resolve()


def _discover_project_root() -> Path:
    """自动发现仓库根目录，兼容服务器上多一层 PD 的挂载结构。"""
    candidates: List[Path] = []
    for p in [_THIS_FILE.parent, *_THIS_FILE.parents]:
        candidates.append(p)
        candidates.append(p / 'PD')

    seen = set()
    for p in candidates:
        if p in seen:
            continue
        seen.add(p)
        if (p / 'codes' / 'multitasklearning').is_dir():
            return p
        if p.name == 'codes' and (p / 'multitasklearning').is_dir():
            return p.parent
    return _THIS_FILE.parents[3]


PROJECT_ROOT = _discover_project_root()
CODES_ROOT = PROJECT_ROOT / 'codes'
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(CODES_ROOT))

from experiments.mtl_experiments.main import (
    MTLExperimentConfig,
    DiceBCELoss,
    FocalCELoss,
    _dice,
    _freeze_backbone_half,
    _make_optimizer,
)
from multitasklearning.unet_multitask import UNetMultiTask
from multitasklearning.multitask_data_loader import (
    PDImageDataset,
    MultiTaskDataLoader,
    build_corn1_loaders,
)
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


def _resolve_dir(rel: str, is_dir: bool = True) -> str:
    """兼容以仓库根、codes 或服务器 PD 子目录为基准的部署，解析数据路径。"""
    rel_path = Path(rel)
    if rel_path.is_absolute():
        candidates = [rel_path]
    else:
        candidates = [
            PROJECT_ROOT / rel_path,
            CODES_ROOT / rel_path,
        ]
        if PROJECT_ROOT.name != 'PD' and (PROJECT_ROOT / 'PD').is_dir():
            candidates.extend([
                PROJECT_ROOT / 'PD' / rel_path,
                PROJECT_ROOT / 'PD' / 'codes' / rel_path,
            ])
        rel_norm = rel.replace('\\', '/').lstrip('./')
        if rel_norm.startswith('codes/'):
            candidates.append(PROJECT_ROOT / rel_norm[len('codes/'):])

    unique_candidates: List[Path] = []
    seen = set()
    for cand in candidates:
        cand = cand.resolve() if cand.exists() else cand
        key = str(cand)
        if key in seen:
            continue
        seen.add(key)
        unique_candidates.append(cand)

    for cand in unique_candidates:
        if cand.is_dir() if is_dir else cand.is_file():
            return str(cand)

    expected = '目录' if is_dir else '文件'
    tried = '\n'.join(f'  - {cand}' for cand in unique_candidates)
    raise FileNotFoundError(f"MTL subtype hierarchy 路径未找到有效{expected}: {rel}\n已尝试路径:\n{tried}")


def _relabel_samples(dataset: PDImageDataset, patients: List[Dict], task: str) -> None:
    """覆盖 PDImageDataset.samples 中的 subtype 标签为二分类标签。"""
    new_samples = []
    for (path, pd_l, stage_l, _sub_l, p_idx) in dataset.samples:
        y3 = SUBTYPE_TO_3CLS[patients[p_idx]['subtype']]
        if task == 'pd_subtype_tdrest':
            bin_l = subtype_tdrest_label(y3)
        else:  # pd_subtype_pigd_mixed
            lab = subtype_pigd_mixed_label(y3)
            bin_l = -1 if lab is None else lab
        new_samples.append((path, pd_l, stage_l, bin_l, p_idx))
    dataset.samples = new_samples


def _build_model(config: MTLExperimentConfig) -> UNetMultiTask:
    pretrained_bool = (config.PRETRAINED is True or config.PRETRAINED == 'Half')
    model = UNetMultiTask(
        encoder_type=config.BACKBONE,
        encoder_pretrained=pretrained_bool,
        classification_mode='dual_stream',
        num_pd_classes=2,
        num_stage_classes=5,
        num_subtype_classes=2,   # 二分类子任务
        mae_enabled=(config.BRANCH_A_TASK == 'mae'),
        dropout_rate=config.DROPOUT,
    )
    if config.PRETRAINED == 'Half':
        _freeze_backbone_half(model)
    return model


def _jsonable_config(config: MTLExperimentConfig) -> Dict:
    """返回可序列化为 JSON 的 MTL 配置快照。"""
    out = {}
    for key, value in vars(config).items():
        if key.startswith('_'):
            continue
        try:
            json.dumps(value)
            out[key] = value
        except TypeError:
            out[key] = str(value)
    return out


def _branch_a_loss(config: MTLExperimentConfig, outputs: Dict,
                   corn1_batch: Dict, seg_criterion, device) -> torch.Tensor:
    task = config.BRANCH_A_TASK
    if task in ('segmentation', 'pseudo'):
        masks = corn1_batch['mask'].to(device)
        seg = outputs['segmentation']
        if seg.shape[2:] != masks.shape[2:]:
            seg = torch.nn.functional.interpolate(
                seg, size=masks.shape[2:], mode='bilinear', align_corners=False)
        weight = config.SEG_WEIGHT if task == 'segmentation' else config.PSEUDO_WEIGHT
        return seg_criterion(seg, masks) * weight
    if task == 'mae':
        mae_raw = outputs.get('losses', {}).get('seg_mae')
        if mae_raw is not None and torch.is_tensor(mae_raw):
            return mae_raw * config.MAE_WEIGHT
    return torch.tensor(0.0, device=device)


def _train_one_subtask(config: MTLExperimentConfig, corn1_root: str,
                       train_patients: List[Dict], val_patients: List[Dict],
                       task: str, datasets_base: str,
                       artifact_dir: Optional[str] = None,
                       reuse_checkpoint: bool = True) -> UNetMultiTask:
    """训练单个二分类子任务，返回 best 模型。"""
    device = torch.device(config.DEVICE)
    model = _build_model(config).to(device)
    checkpoint_path = os.path.join(artifact_dir, 'best_model.pth') if artifact_dir else None
    history_path = os.path.join(artifact_dir, 'history.json') if artifact_dir else None

    if checkpoint_path and reuse_checkpoint and os.path.exists(checkpoint_path):
        payload = torch.load(checkpoint_path, map_location=device)
        state = payload.get('model_state', payload) if isinstance(payload, dict) else payload
        model.load_state_dict(state)
        print(f"    [{task}] 复用已有模型，跳过训练: {checkpoint_path}", flush=True)
        return model

    if artifact_dir:
        os.makedirs(artifact_dir, exist_ok=True)

    # Branch A: CORN-1 加载器
    corn1_train, _ = build_corn1_loaders(
        corn1_root, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS,
        augment_train=config.DATA_AUGMENTATION, image_size=config.IMAGE_SIZE,
        generate_pseudo=(config.BRANCH_A_TASK == 'pseudo'),
    )

    # Branch B: subtype 二分类 PD 加载器
    train_ds = PDImageDataset(train_patients, datasets_base=datasets_base,
                              augment=config.DATA_AUGMENTATION, image_size=config.IMAGE_SIZE)
    val_ds = PDImageDataset(val_patients, datasets_base=datasets_base,
                            augment=False, image_size=config.IMAGE_SIZE)
    _relabel_samples(train_ds, train_patients, task)
    _relabel_samples(val_ds, val_patients, task)

    pd_train = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True,
                          num_workers=config.NUM_WORKERS, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False,
                            num_workers=config.NUM_WORKERS)
    mtl_loader = MultiTaskDataLoader(corn1_train, pd_train)

    optimizer = _make_optimizer(model, config)
    if config.LR_SCHEDULER == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.NUM_EPOCHS, eta_min=1e-7)
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

    use_amp = config.MIXED_PRECISION and 'cuda' in str(config.DEVICE)
    scaler = GradScaler() if use_amp else None
    seg_criterion = DiceBCELoss()
    sub_criterion = FocalCELoss(ignore_index=-1)

    best_acc, best_state = -1.0, None
    patience_counter = 0
    history = []
    for epoch in range(1, config.NUM_EPOCHS + 1):
        model.train()
        epoch_losses = []
        for corn1_batch, pd_batch in mtl_loader:
            optimizer.zero_grad(set_to_none=True)
            with (autocast(device_type='cuda') if scaler
                  else torch.autocast('cpu', enabled=False)):
                out_a = model(corn1_batch['image'].to(device),
                              use_mae=(config.BRANCH_A_TASK == 'mae'), training=True)
                loss_a = _branch_a_loss(config, out_a, corn1_batch, seg_criterion, device)

                sub_l = pd_batch['subtype_label'].to(device)
                out_b = model(pd_batch['image'].to(device),
                              subtype_labels=sub_l, use_mae=False, training=True)
                loss_b = sub_criterion(out_b['subtype_logits'], sub_l) * config.SUBTYPE_WEIGHT
                loss = loss_a + loss_b
                epoch_losses.append(float(loss.detach().cpu().item()))

            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
                optimizer.step()
        scheduler.step()

        # 验证：image-level subtype 二分类准确率（用于早停/选最佳）
        val_acc = _eval_subtype_acc(model, val_loader, device)
        improved = val_acc > best_acc
        if improved:
            best_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            if checkpoint_path:
                torch.save({
                    'model_state': best_state,
                    'task': task,
                    'best_val_acc': best_acc,
                    'epoch': epoch,
                    'config': _jsonable_config(config),
                }, checkpoint_path)
        else:
            patience_counter += 1
        history.append({
            'epoch': epoch,
            'train_loss': float(np.mean(epoch_losses)) if epoch_losses else 0.0,
            'val_bin_acc': float(val_acc),
            'best_val_acc': float(best_acc),
            'improved': bool(improved),
        })
        if history_path:
            with open(history_path, 'w', encoding='utf-8') as f:
                json.dump(history, f, indent=2, ensure_ascii=False)
        print(f"    [{task}] E{epoch:03d}/{config.NUM_EPOCHS} "
              f"val_bin_acc={val_acc:.4f}{' ↑' if improved else ''}", flush=True)
        if config.EARLY_STOPPING and patience_counter >= config.PATIENCE:
            print(f"    [{task}] 早停 (patience={config.PATIENCE})", flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def _eval_subtype_acc(model, loader, device) -> float:
    model.eval()
    correct, total = 0, 0
    for batch in loader:
        labels = batch['subtype_label'].to(device)
        mask = labels >= 0
        if mask.sum() == 0:
            continue
        out = model(batch['image'].to(device), use_mae=False, training=False)
        preds = out['subtype_logits'].argmax(dim=1)
        correct += int((preds[mask] == labels[mask]).sum().item())
        total += int(mask.sum().item())
    return correct / total if total else 0.0


@torch.no_grad()
def _infer_patient_level(model, patients: List[Dict], config: MTLExperimentConfig,
                         datasets_base: str) -> Tuple[Dict[str, int], Dict[str, float]]:
    """对一组患者做 patient-level 推理（图像 softmax 概率均值聚合）。"""
    device = torch.device(config.DEVICE)
    ds = PDImageDataset(patients, datasets_base=datasets_base,
                        augment=False, image_size=config.IMAGE_SIZE)
    loader = DataLoader(ds, batch_size=config.BATCH_SIZE, shuffle=False,
                        num_workers=config.NUM_WORKERS)

    model.eval()
    sum_probs: Dict[int, np.ndarray] = defaultdict(lambda: np.zeros(2))
    counts: Dict[int, int] = defaultdict(int)
    for batch in loader:
        out = model(batch['image'].to(device), use_mae=False, training=False)
        probs = torch.softmax(out['subtype_logits'], dim=1).cpu().numpy()
        pidxs = batch['patient_idx']
        pidxs = pidxs.tolist() if torch.is_tensor(pidxs) else list(pidxs)
        for i, pidx in enumerate(pidxs):
            sum_probs[int(pidx)] += probs[i]
            counts[int(pidx)] += 1

    pred_by_pid: Dict[str, int] = {}
    prob_class1_by_pid: Dict[str, float] = {}
    for pidx, total in sum_probs.items():
        mean_prob = total / max(1, counts[pidx])
        patient_id = patients[pidx]['patient_id']
        pred_by_pid[patient_id] = int(np.argmax(mean_prob))
        prob_class1_by_pid[patient_id] = float(mean_prob[1])
    return pred_by_pid, prob_class1_by_pid


def run_mtl_subtype_hierarchy_experiment(config: MTLExperimentConfig, output_dir: str,
                                         seed: int = 42, n_splits: int = 5,
                                         repeats: int = 3) -> Dict:
    """
    运行 MTL subtype 层级三分类 5-fold x 3 重算实验（只跑 subtype 任务）。

    Returns:
        summary 字典（同时写入 summary.json）。
    """
    os.makedirs(output_dir, exist_ok=True)
    set_seed(seed)

    config_path = os.path.join(output_dir, 'config.json')
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump({
            'config': _jsonable_config(config),
            'seed': seed,
            'n_splits': n_splits,
            'repeats': repeats,
        }, f, indent=2, ensure_ascii=False)

    # 只跑 subtype：关闭 PD / Stage 分类损失
    config.PD_WEIGHT = 0.0
    config.STAGE_WEIGHT = 0.0
    if config.SUBTYPE_WEIGHT <= 0:
        config.SUBTYPE_WEIGHT = 1.0

    corn1_root = _resolve_dir(config.CORN1_ROOT, is_dir=True)
    datasets_base = _resolve_dir(config.DATASETS_BASE, is_dir=True)
    pd_json = _resolve_dir(config.PD_DATASET_JSON, is_dir=False)

    with open(pd_json, 'r', encoding='utf-8') as f:
        all_patients = json.load(f)
    subtype_patients = [p for p in all_patients
                        if p['group'] == 'pd' and p.get('subtype') in SUBTYPE_TO_3CLS]
    pid_to_patient = {p['patient_id']: p for p in subtype_patients}
    records = [{'patient_id': p['patient_id'], 'subtype': p['subtype']}
               for p in subtype_patients]
    folds = build_patient_5x3_subtype_folds(records, seed=seed,
                                            n_splits=n_splits, repeats=repeats)

    print("=" * 60)
    print("MTL Subtype 层级三分类公平重算 (5-fold x 3, 仅 subtype 任务)")
    print(f"subtype 患者数: {len(subtype_patients)} | "
          f"backbone: {config.BACKBONE} | BranchA: {config.BRANCH_A_TASK} | "
          f"fold 总数: {len(folds)}")
    print("=" * 60)

    fold_results: List[Dict] = []
    pred_rows: List[Dict] = []

    for global_idx, fold in enumerate(folds, start=1):
        repeat, fold_no = fold['repeat'], fold['fold']
        train_patients = [pid_to_patient[pid] for pid in fold['train_patients_full']]
        val_patients = [pid_to_patient[pid] for pid in fold['val_patients_full']]

        print(f"\n--- repeat {repeat} fold {fold_no} ({global_idx}/{len(folds)}) | "
              f"train={len(train_patients)} val={len(val_patients)} ---", flush=True)
        fold_artifact_dir = os.path.join(output_dir, f'repeat_{repeat}_fold_{fold_no}')

        # 第一层: TD vs rest（全部 subtype 患者）
        tdrest_model = _train_one_subtask(
            config, corn1_root, train_patients, val_patients,
            'pd_subtype_tdrest', datasets_base,
            artifact_dir=os.path.join(fold_artifact_dir, 'pd_subtype_tdrest'))
        tdrest_pred, tdrest_prob = _infer_patient_level(
            tdrest_model, val_patients, config, datasets_base)
        del tdrest_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 第二层: PIGD vs mixed（训练排除 TD；推理覆盖 full val fold）
        pm_train_patients = [p for p in train_patients if p['subtype'] in ('PIGD', 'mixed')]
        pm_val_patients = [p for p in val_patients if p['subtype'] in ('PIGD', 'mixed')]
        pm_model = _train_one_subtask(
            config, corn1_root, pm_train_patients, pm_val_patients,
            'pd_subtype_pigd_mixed', datasets_base,
            artifact_dir=os.path.join(fold_artifact_dir, 'pd_subtype_pigd_mixed'))
        pm_pred, pm_prob = _infer_patient_level(
            pm_model, val_patients, config, datasets_base)
        del pm_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
              f"hierarchy_acc={metrics['subtype_hierarchy_acc']:.4f}", flush=True)

    summary = summarize_15fold_results(fold_results, n_splits=n_splits, repeats=repeats)
    summary['backbone'] = config.BACKBONE
    summary['branch_a_task'] = config.BRANCH_A_TASK
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
    print("MTL Subtype 层级三分类汇总:")
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
