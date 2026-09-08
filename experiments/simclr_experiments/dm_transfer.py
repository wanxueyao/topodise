"""SimCLR 到 DM 数据集的迁移运行器

目的
----
把在 ``codes/datasets/pd_dataset_en`` 上学到的 SimCLR 自监督表征迁移到糖尿病
视网膜数据集 ``codes/datasets/dm_dataset_en``，只重新训练下游分类头，不重新做
自监督预训练（"可以不重新训练，利用之前在 pd_dataset_en 中的训练结果"）。

与 TopoDiSE 的 DM transfer 配置思路对齐：
- topodise 复用 PD 最佳 Stage1 表征 ckpt，只重训 Stage2。
- 这里复用在 pd_dataset_en 上完成的 SimCLR backbone 表征，只重训有监督微调头。

DM 任务（均为二分类，复用二分类 'pd' 头）
-----------------------------------------
- ``dm_binary``：control (0) vs NDR/NPDR (1)，全部 80 患者。
- ``dm_subtype``：NDR (0) vs NPDR (1)，排除 control（60 患者）。

复用机制
--------
1. 首次需要某个 SimCLR 配置的 backbone 时，在 **完整 pd_dataset_en**（pd_all，
   无标签）上做一次自监督预训练，并把 backbone 权重原子化缓存到磁盘。
2. 之后所有 DM 实验（不同 fold / seed / 任务）直接加载该缓存 backbone，跳过预
   训练，仅做有监督微调。
3. 缓存签名只取 SimCLR 超参与 backbone（不含微调 seed），因此共享同一 SimCLR
   配置的实验复用同一份 pd 表征，最大化"不重新训练"。
4. 也支持用环境变量 ``RUN_SIMCLR_PRETRAIN_CKPT`` 指定一个已有 backbone ckpt，
   直接复用、永不预训练。

该模块由 ``batch_experiments.py`` 中保留的论文 DM 配置调用
（实验字典里带 ``_transfer='dm'``）。
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Subset

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.simclr_classifier import SimCLRClassifier  # noqa: E402
from core.simclr_trainer import SimCLRModel, simclr_pretrain  # noqa: E402
from core.data_loader import PDDataset, load_and_filter_dataset, stratified_kfold  # noqa: E402
from utils.misc import set_seed, Tee, save_config  # noqa: E402
from utils.patient_outputs import save_fold_patient_outputs, task_labels_for_indices  # noqa: E402

from experiments.simclr_experiments.main import SimCLRExperimentConfig  # noqa: E402


# ==================== 路径与任务常量 ====================

DM_JSON = 'datasets/dm_dataset_en/dm_patients_dataset_en.json'
PD_PRETRAIN_JSON = 'datasets/pd_dataset_en/patients_dataset_en.json'
PD_PRETRAIN_MODE = 'pd_all'
CORN1_PRETRAIN_CACHE_TAG = 'corn1'

DM_TASKS: List[str] = ['dm_binary', 'dm_subtype']

# DM 原始 group → DM 二分类标签
_DM_SUBTYPE_MAP = {'NDR': 0, 'NPDR': 1}

# 所有 DM 任务都映射到二分类 'pd' 头（num_classes=2, label_key='pd'）
_DM_HEAD_TASK = 'pd'

# pd 自监督预训练使用固定 seed，使所有 DM 实验共享同一份 pd 表征
PRETRAIN_SEED = 42

# backbone 缓存目录（相对 PROJECT_ROOT）
PRETRAIN_CACHE_DIR = 'results/simclr_dm_transfer/pretrain_cache'

# 进程内缓存：签名 -> SimCLRModel，避免同一进程多任务重复加载
_INPROC_PRETRAIN_CACHE: Dict[str, SimCLRModel] = {}


def _json_default(o: Any):
    """让 json.dump 能序列化 numpy 标量/数组（如 int64/float64/ndarray）。"""
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")


# ==================== 路径解析 ====================

def _resolve_project_path(path: str) -> str:
    """把相对路径解析为存在的绝对路径，兼容从 ARIS 根或 codes 启动。"""
    if os.path.isabs(path):
        return path
    candidates = [
        os.path.join(PROJECT_ROOT, path),
        os.path.join(os.path.dirname(PROJECT_ROOT), path),
    ]
    for cand in candidates:
        if os.path.exists(cand):
            return cand
    return candidates[0]


def _datasets_base_path(json_abs_path: str) -> str:
    """由数据集 JSON 绝对路径推断 datasets 根目录（image_paths 的拼接基准）。"""
    return os.path.dirname(os.path.dirname(json_abs_path))


# ==================== DM 数据构建 ====================

def build_dm_patients(task: str) -> List[Dict[str, Any]]:
    """构建符合 PD 数据 schema 的 DM 患者列表，使其可直接喂给 ``PDDataset``。

    DM 原始 JSON 的 ``image_paths`` 是扁平列表（无 central/whorl 区分），这里把
    同一张图同时作为 central 与 whorl（duplicate 配对），以适配 dual_stream
    编码器；并通过 ``group`` 字段编码二分类标签（control→0, pd→1），``stage`` 与
    ``subtype`` 置空使其编码为 -1（不参与）。

    参数：
        task: 'dm_binary' 或 'dm_subtype'。

    返回：
        PD-schema 患者列表（每个含 group/stage/subtype/image_paths）。
    """
    if task not in DM_TASKS:
        raise ValueError(f"未知 DM 任务: {task}（应为 {DM_TASKS}）")

    json_abs = _resolve_project_path(DM_JSON)
    with open(json_abs, 'r', encoding='utf-8') as f:
        raw_patients = json.load(f)

    out: List[Dict[str, Any]] = []
    for p in raw_patients:
        group = p.get('group', '')
        if task == 'dm_binary':
            label = 0 if group == 'control' else 1
        else:  # dm_subtype：排除 control，NDR vs NPDR
            if group == 'control':
                continue
            label = _DM_SUBTYPE_MAP.get(group)
            if label is None:
                continue

        imgs = list(p.get('image_paths', []))
        if not imgs:
            continue

        out.append({
            'patient_id': p.get('patient_id', ''),
            'name': p.get('name', ''),
            # group 仅用于 encode_labels 得到二分类 pd_label：1→'pd'，0→'control'
            'group': 'pd' if label == 1 else 'control',
            'stage': 0,        # → stage_label = -1（不参与）
            'subtype': '',     # → subtype_label = -1（不参与）
            'image_paths': {'central': imgs, 'whorl': imgs},
        })
    return out


# ==================== pd 自监督 backbone 复用 ====================

def _pretrain_signature(cfg: SimCLRExperimentConfig, input_size: int) -> str:
    """根据 SimCLR 超参与 backbone 生成缓存签名（不含微调 seed）。"""
    key = {
        'backbone': cfg.BACKBONE,
        'pretrained': str(cfg.PRETRAINED),
        'simclr_epochs': cfg.SIMCLR_EPOCHS,
        'simclr_lr': cfg.SIMCLR_LR,
        'simclr_batch': cfg.SIMCLR_BATCH_SIZE,
        'temperature': cfg.SIMCLR_TEMPERATURE,
        'proj_dim': cfg.SIMCLR_PROJ_DIM,
        'proj_hidden_dim': cfg.SIMCLR_PROJ_HIDDEN_DIM,
        'simclr_wd': cfg.SIMCLR_WEIGHT_DECAY,
        'input_size': input_size,
        'vanillanet_act_num_init': cfg.VANILLANET_INIT_ACT_NUM,
        'pretrain_source': getattr(cfg, 'SIMCLR_PRETRAIN_DATASET', 'fold'),
        'corn1_root': getattr(cfg, 'CORN1_ROOT', ''),
        'corn1_splits': getattr(cfg, 'CORN1_PRETRAIN_SPLITS', ''),
        'pretrain_json': PD_PRETRAIN_JSON,
        'pretrain_mode': PD_PRETRAIN_MODE,
        'pretrain_seed': PRETRAIN_SEED,
    }
    raw = json.dumps(key, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(raw.encode('utf-8')).hexdigest()[:12]


def _build_simclr_model_from_state(cfg: SimCLRExperimentConfig,
                                   backbone_state: Dict[str, Any]) -> SimCLRModel:
    """重建 SimCLRModel 并载入 backbone 权重（投影头随机初始化，不影响下游）。"""
    model = SimCLRModel(
        cfg.BACKBONE,
        pretrained=False,
        proj_hidden_dim=cfg.SIMCLR_PROJ_HIDDEN_DIM,
        proj_out_dim=cfg.SIMCLR_PROJ_DIM,
    )
    model.get_backbone().load_state_dict(backbone_state)
    return model.to(cfg.DEVICE)


def get_or_build_pd_simclr(cfg: SimCLRExperimentConfig,
                           classifier: SimCLRClassifier,
                           log) -> SimCLRModel:
    """获取在 pd_dataset_en 上预训练好的 SimCLR backbone（复用优先，必要时训练一次）。

    复用优先级：
    1. 环境变量 ``RUN_SIMCLR_PRETRAIN_CKPT`` 指定的 backbone ckpt；
    2. 进程内缓存；
    3. 磁盘缓存（按签名）；
    4. 都没有则在完整 pd_dataset_en 上预训练一次并原子化写盘。
    """
    input_size = classifier._get_input_size()
    sig = _pretrain_signature(cfg, input_size)

    # 1) 显式 ckpt 覆盖
    env_ckpt = os.environ.get('RUN_SIMCLR_PRETRAIN_CKPT', '').strip()
    if env_ckpt:
        ckpt_abs = _resolve_project_path(env_ckpt)
        if os.path.exists(ckpt_abs):
            log(f"[dm-transfer] 复用显式 SimCLR backbone ckpt: {ckpt_abs}")
            state = torch.load(ckpt_abs, map_location=cfg.DEVICE)
            backbone_state = state.get('backbone_state', state) if isinstance(state, dict) else state
            return _build_simclr_model_from_state(cfg, backbone_state)
        log(f"[dm-transfer][warn] RUN_SIMCLR_PRETRAIN_CKPT 指定的文件不存在: {ckpt_abs}")

    # 2) 进程内缓存
    if sig in _INPROC_PRETRAIN_CACHE:
        return _INPROC_PRETRAIN_CACHE[sig]

    # 3) 磁盘缓存
    cache_dir = _resolve_project_path(PRETRAIN_CACHE_DIR)
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{cfg.BACKBONE}_{sig}.pt")
    if os.path.exists(cache_path) and os.path.getsize(cache_path) > 0:
        log(f"[dm-transfer] 复用 pd_dataset_en SimCLR backbone 缓存（跳过预训练）:\n"
            f"             {cache_path}")
        state = torch.load(cache_path, map_location=cfg.DEVICE)
        model = _build_simclr_model_from_state(cfg, state['backbone_state'])
        _INPROC_PRETRAIN_CACHE[sig] = model
        return model

    # 4) 训练一次：默认完整 pd_dataset_en；若配置为 corn1，则使用 CORN-1 图像。
    pretrain_source = str(getattr(cfg, 'SIMCLR_PRETRAIN_DATASET', 'fold')).lower()
    if pretrain_source in ('corn1', 'corn-1'):
        log(f"[dm-transfer] 未找到缓存，开始在 CORN-1 上做一次 SimCLR 预训练"
            f"（backbone={cfg.BACKBONE}, epochs={cfg.SIMCLR_EPOCHS}）...")
        pd_dataset = classifier._resolve_pretrain_dataset(None)
    else:
        log(f"[dm-transfer] 未找到缓存，开始在 pd_dataset_en（{PD_PRETRAIN_MODE}）上做一次 "
            f"SimCLR 预训练（backbone={cfg.BACKBONE}, epochs={cfg.SIMCLR_EPOCHS}）...")
        pd_json_abs = _resolve_project_path(PD_PRETRAIN_JSON)
        pd_base_path = _datasets_base_path(pd_json_abs)
        pd_patients = load_and_filter_dataset(pd_json_abs, PD_PRETRAIN_MODE)
        pd_dataset = PDDataset(
            pd_patients, 'dual_stream',
            classifier.train_transform, base_path=pd_base_path,
        )
        if len(pd_dataset) == 0:
            raise RuntimeError(f"pd 预训练数据集为空: {pd_json_abs}")

    set_seed(PRETRAIN_SEED)
    model = simclr_pretrain(
        backbone_name=cfg.BACKBONE,
        pretrained=cfg.PRETRAINED,
        train_dataset=pd_dataset,
        num_epochs=cfg.SIMCLR_EPOCHS,
        batch_size=cfg.SIMCLR_BATCH_SIZE,
        lr=cfg.SIMCLR_LR,
        temperature=cfg.SIMCLR_TEMPERATURE,
        device=cfg.DEVICE,
        proj_hidden_dim=cfg.SIMCLR_PROJ_HIDDEN_DIM,
        proj_out_dim=cfg.SIMCLR_PROJ_DIM,
        weight_decay=cfg.SIMCLR_WEIGHT_DECAY,
        num_workers=cfg.NUM_WORKERS,
        input_size=input_size,
        vanillanet_act_num_init=cfg.VANILLANET_INIT_ACT_NUM,
        verbose=True,
    )

    # 原子化写盘（容忍并发：tmp + replace）
    backbone_state = {k: v.detach().cpu() for k, v in model.get_backbone().state_dict().items()}
    payload = {
        'backbone_state': backbone_state,
        'meta': {
            'backbone': cfg.BACKBONE,
            'signature': sig,
            'pretrain_source': getattr(cfg, 'SIMCLR_PRETRAIN_DATASET', 'fold'),
            'pretrain_json': PD_PRETRAIN_JSON,
            'pretrain_mode': PD_PRETRAIN_MODE,
            'pretrain_seed': PRETRAIN_SEED,
            'simclr_epochs': cfg.SIMCLR_EPOCHS,
            'temperature': cfg.SIMCLR_TEMPERATURE,
            'proj_dim': cfg.SIMCLR_PROJ_DIM,
            'created': datetime.now().isoformat(timespec='seconds'),
        },
    }
    tmp_path = f"{cache_path}.tmp.{os.getpid()}"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, cache_path)
    log(f"[dm-transfer] 已缓存 pd SimCLR backbone: {cache_path}")

    _INPROC_PRETRAIN_CACHE[sig] = model
    return model


# ==================== 单实验运行 ====================

def _aggregate_fold_metrics(fold_metrics: List[Dict[str, float]]) -> Dict[str, float]:
    if not fold_metrics:
        return {}
    accs = [m['acc'] for m in fold_metrics]
    patient_accs = [m.get('patient_acc', 0.0) for m in fold_metrics]
    f1s = [m['f1'] for m in fold_metrics]
    train_accs = [m.get('train_acc', 0.0) for m in fold_metrics]
    return {
        'avg_train_acc': float(np.mean(train_accs)),
        'avg_acc': float(np.mean(accs)),
        'std_acc': float(np.std(accs)),
        'avg_patient_acc': float(np.mean(patient_accs)),
        'std_patient_acc': float(np.std(patient_accs)),
        'avg_f1': float(np.mean(f1s)),
        'std_f1': float(np.std(f1s)),
        'num_folds': len(fold_metrics),
    }


def _run_dm_task_kfold(classifier: SimCLRClassifier,
                       cfg: SimCLRExperimentConfig,
                       task: str) -> Dict[str, Any]:
    """对单个 DM 任务执行 patient-level K-fold（复用二分类 'pd' 头）。"""
    dm_json_abs = _resolve_project_path(DM_JSON)
    dm_base_path = _datasets_base_path(dm_json_abs)

    patients = build_dm_patients(task)
    dataset = PDDataset(
        patients, cfg.ARCHITECTURE,
        classifier.train_transform, base_path=dm_base_path,
    )
    print(f"\n{'='*60}\nDM 任务: {task} | 样本数: {len(dataset)} | "
          f"患者数: {len(set(dataset.patient_ids))}\n{'='*60}")
    if len(dataset) == 0:
        print(f"⚠ DM 任务 {task} 无可用样本，跳过")
        return {}

    splits = list(stratified_kfold(
        dataset.PD_labels, dataset.stage_labels, dataset.subtype_labels,
        n_splits=cfg.NUM_FOLDS, random_state=cfg.SEED,
        kfold_mode=cfg.KFOLD_MODE, patient_ids=dataset.patient_ids,
    ))

    fold_metrics: List[Dict[str, float]] = []
    all_labels: List[int] = []
    all_preds: List[int] = []

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        print(f"\n--- DM {task} Fold {fold_idx + 1}/{cfg.NUM_FOLDS} ---")
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        result = classifier.train(
            train_subset, val_subset, _DM_HEAD_TASK,
            fold_idx=fold_idx + 1, suffix=f"_{task}",
        )
        vr = result['val_result']
        if vr.get('labels'):
            th = result.get('train_history', {})
            patient_labels, patient_ids = task_labels_for_indices(dataset, val_idx, 'pd')
            patient_result = save_fold_patient_outputs(
                cfg.OUTPUT_DIR, _DM_HEAD_TASK, f"_{task}", fold_idx + 1,
                patient_ids, patient_labels,
                probs=vr.get('probs'), logits=vr.get('logits'),
            )
            fold_metrics.append({
                'fold': fold_idx + 1,
                'train_acc': th['train_acc'][-1] if th.get('train_acc') else 0.0,
                'acc': vr['acc'],
                'patient_acc': patient_result['patient_acc'],
                'n_patients': patient_result['n_patients'],
                'patient_logits_json': patient_result['patient_logits_json'],
                'patient_logits_csv': patient_result['patient_logits_csv'],
                'f1': vr['f1'],
            })
            all_labels.extend(vr['labels'])
            all_preds.extend(vr['preds'])
            print(f"  Acc: {vr['acc']:.4f}, PatientAcc: {patient_result['patient_acc']:.4f}, F1: {vr['f1']:.4f}")

    summary = _aggregate_fold_metrics(fold_metrics)
    if summary:
        print(f"\n📊 DM {task}: TrainAcc={summary['avg_train_acc']:.4f}, "
              f"ValAcc={summary['avg_acc']:.4f}±{summary['std_acc']:.4f}, "
              f"PatientAcc={summary.get('avg_patient_acc', 0.0):.4f}±{summary.get('std_patient_acc', 0.0):.4f}, "
              f"ValF1={summary['avg_f1']:.4f}±{summary['std_f1']:.4f}")
    return {
        'fold_results': fold_metrics,
        'final_results': summary,
        'all_predictions': {'labels': all_labels, 'preds': all_preds},
    }


def run_single_dm_transfer_experiment(cfg: SimCLRExperimentConfig,
                                      tasks: Optional[List[str]] = None,
                                      output_dir: Optional[str] = None) -> Dict[str, Any]:
    """运行单组 DM 迁移实验：复用 pd SimCLR backbone，仅微调下游头。

    参数：
        cfg: SimCLR 实验配置（由批量 runner 注入超参）。
        tasks: DM 任务列表，默认 ``DM_TASKS``。
        output_dir: 输出目录（由批量 runner 提供）。

    返回：
        结果字典（每个 DM 任务的 K-fold 汇总）。
    """
    tasks = [t for t in (tasks or DM_TASKS) if t in DM_TASKS] or list(DM_TASKS)

    output_dir = output_dir or os.path.join(
        _resolve_project_path('results/simclr_dm_transfer'),
        f"dm_transfer_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    os.makedirs(output_dir, exist_ok=True)
    cfg.OUTPUT_DIR = output_dir
    save_config(cfg, output_dir)

    log_file = open(os.path.join(output_dir, 'output.txt'), 'w', encoding='utf-8')
    original_stdout = sys.stdout
    sys.stdout = Tee(original_stdout, log_file)

    start_time = time.time()
    results: Dict[str, Any] = {}
    try:
        print("=" * 60)
        print("SimCLR → DM 迁移实验（复用 pd_dataset_en 自监督表征，仅微调下游头）")
        print("=" * 60)
        print(f"Backbone={cfg.BACKBONE} | Arch={cfg.ARCHITECTURE} | "
              f"Pretrained={cfg.PRETRAINED}")
        print(f"微调: epochs={cfg.NUM_EPOCHS}, lr={cfg.LEARNING_RATE}, "
              f"wd={cfg.WEIGHT_DECAY}, backbone_lr_ratio={cfg.BACKBONE_LR_RATIO}, "
              f"pd_loss={cfg.PD_LOSS_TYPE}")
        print(f"K-fold: {cfg.NUM_FOLDS} ({cfg.KFOLD_MODE}) | seed={cfg.SEED} | "
              f"tasks={tasks}")

        classifier = SimCLRClassifier(cfg)

        # 复用 pd 自监督 backbone（不重新训练）
        simclr_model = get_or_build_pd_simclr(cfg, classifier, log=print)
        classifier.precomputed_simclr = simclr_model

        # 复原微调阶段随机性（pd 预训练可能改过全局 seed）
        set_seed(cfg.SEED)

        for task in tasks:
            results[task] = _run_dm_task_kfold(classifier, cfg, task)

        _write_dm_report(output_dir, cfg, results, tasks)
    finally:
        sys.stdout = original_stdout
        log_file.close()

    duration = time.time() - start_time
    results['meta'] = {
        'output_dir': output_dir,
        'duration': duration,
        'tasks': tasks,
        'backbone': cfg.BACKBONE,
        'seed': cfg.SEED,
    }
    with open(os.path.join(output_dir, 'results.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=_json_default)

    print(f"DM 迁移实验完成（{cfg.BACKBONE}, seed={cfg.SEED}），用时 "
          f"{duration / 60:.1f}min，结果: {output_dir}")
    return results


def _write_dm_report(output_dir: str, cfg: SimCLRExperimentConfig,
                     results: Dict[str, Any], tasks: List[str]) -> None:
    path = os.path.join(output_dir, 'experiment_report.md')
    with open(path, 'w', encoding='utf-8') as f:
        f.write("# SimCLR → DM 迁移实验报告\n\n")
        f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("复用 `pd_dataset_en` 上的 SimCLR 自监督 backbone，仅在 "
                "`dm_dataset_en` 上微调下游二分类头。\n\n")
        f.write("## 配置\n\n| 参数 | 值 |\n|---|---|\n")
        f.write(f"| Backbone | `{cfg.BACKBONE}` |\n")
        f.write(f"| 架构 | `{cfg.ARCHITECTURE}` |\n")
        f.write(f"| ImageNet 预训练 | `{cfg.PRETRAINED}` |\n")
        f.write(f"| SimCLR 预训练源 | `{getattr(cfg, 'SIMCLR_PRETRAIN_DATASET', 'fold')}` |\n")
        f.write(f"| CORN-1 路径 | `{getattr(cfg, 'CORN1_ROOT', '')}` |\n")
        f.write(f"| CORN-1 splits | `{getattr(cfg, 'CORN1_PRETRAIN_SPLITS', '')}` |\n")
        f.write(f"| SimCLR Epochs/T/dim | {cfg.SIMCLR_EPOCHS} / "
                f"{cfg.SIMCLR_TEMPERATURE} / {cfg.SIMCLR_PROJ_DIM} |\n")
        f.write(f"| 微调 Epochs/LR/WD | {cfg.NUM_EPOCHS} / {cfg.LEARNING_RATE} / "
                f"{cfg.WEIGHT_DECAY} |\n")
        f.write(f"| Backbone LR 倍率 | {cfg.BACKBONE_LR_RATIO} |\n")
        f.write(f"| 微调损失 | `{cfg.PD_LOSS_TYPE}` (eps={cfg.LABEL_SMOOTH_EPS}) |\n")
        f.write(f"| K-fold | {cfg.NUM_FOLDS} ({cfg.KFOLD_MODE}) |\n")
        f.write(f"| Seed | {cfg.SEED} |\n\n")

        f.write("## 结果摘要\n\n| DM 任务 | Patient Accuracy | Image Accuracy | F1 |\n|---|---|---|---|\n")
        for task in tasks:
            fr = results.get(task, {}).get('final_results', {})
            if fr:
                f.write(
                    f"| {task} | {fr.get('avg_patient_acc', 0.0):.4f}±{fr.get('std_patient_acc', 0.0):.4f} | "
                    f"{fr['avg_acc']:.4f}±{fr['std_acc']:.4f} | "
                    f"{fr['avg_f1']:.4f}±{fr['std_f1']:.4f} |\n"
                )
            else:
                f.write(f"| {task} | - | - | - |\n")
        f.write("\n")
