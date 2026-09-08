"""MTL 到 DM 数据集的迁移运行器

目的
----
把在 ``codes/datasets/pd_dataset_en`` (+ CORN-1) 上多任务学习到的 **encoder 表征**
迁移到糖尿病视网膜数据集 ``codes/datasets/dm_dataset_en``，只重新训练下游分类头，
不重新做多任务预训练（"可以不重新训练，利用之前在 pd_dataset_en 中的训练结果"）。

与 SimCLR / topodise 的 ``*_dm_transfer.py`` 思路一致：
- SimCLR：复用 pd 自监督 backbone，重训下游头。
- topodise：复用 pd Stage1 表征 ckpt，重训 Stage2。
- 这里：复用在 pd_dataset_en + CORN-1 上联合训练得到的 UNet encoder，重训分类头。

DM 任务（均为二分类，复用二分类头）
-----------------------------------
- ``dm_binary`` ：control (0) vs NDR/NPDR (1)，全部 80 患者。
- ``dm_subtype``：NDR (0) vs NPDR (1)，排除 control（60 患者）。

复用机制
--------
1. 首次需要某条 MTL 线（backbone × branch_a_task）的 encoder 时，在 pd_dataset_en
   (+ CORN-1) 上用该线的最佳预训练超参做一次多任务训练，并把 **encoder 权重** 原子化
   缓存到磁盘。
2. 之后所有 DM 实验（不同 fold / seed / 任务）直接加载该缓存 encoder，跳过多任务
   预训练，仅训练 DM 下游分类头。
3. 缓存签名只取 backbone + branch_a_task + 固定预训练超参 + PRETRAIN_SEED（不含 DM
   微调超参），因此同一条线的所有 DM 实验复用同一份 pd encoder，最大化"不重新训练"。
4. 也支持用环境变量 ``RUN_MTL_PRETRAIN_CKPT`` 指定一个已有 encoder ckpt，直接复用、
   永不预训练。

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
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader

# ── 项目根目录 ─────────────────────────────────────────────────────────
_THIS_FILE = Path(__file__).resolve()
CODES_ROOT = str(_THIS_FILE.parents[2])
REPO_ROOT = str(_THIS_FILE.parents[3])
if CODES_ROOT not in sys.path:
    sys.path.insert(0, CODES_ROOT)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from multitasklearning.encoder import UNetEncoder  # noqa: E402
from multitasklearning.unet_multitask import UNetMultiTask  # noqa: E402
from multitasklearning.multitask_data_loader import (  # noqa: E402
    PDImageDataset,
    MultiTaskDataLoader,
    build_corn1_loaders,
    build_pd_patient_kfold_loaders,
    _patient_kfold_split,
)
from utils.misc import set_seed  # noqa: E402
from utils.patient_outputs import aggregate_patient_outputs, save_fold_patient_outputs  # noqa: E402

from experiments.mtl_experiments.main import (  # noqa: E402
    MTLExperimentConfig,
    FoldTrainer,
    _freeze_backbone_half,
)


# ==================== 路径与任务常量 ====================

DM_JSON = 'datasets/dm_dataset_en/dm_patients_dataset_en.json'
DM_TASKS: List[str] = ['dm_binary', 'dm_subtype']

# DM 原始 group → 二分类标签（dm_subtype 排除 control）
_DM_SUBTYPE_MAP = {'NDR': 0, 'NPDR': 1}

# pd 多任务预训练使用固定 seed，使所有 DM 实验共享同一份 pd encoder
PRETRAIN_SEED = 42

# 每条 MTL 线（branch_a_task）的最佳预训练超参（来自 pd_dataset_en 基线结论）
#   mae          : B03 (ep50, lr2e-4) —— 全实验验证集分数最高
#   segmentation : A05 (ep40, lr3e-4) —— Group A 最高
_PRETRAIN_HP: Dict[str, Dict[str, Any]] = {
    'mae':          dict(epochs=50, lr=2e-4, wd=3e-3, dropout=0.0, pretrained='Half'),
    'segmentation': dict(epochs=40, lr=3e-4, wd=3e-3, dropout=0.0, pretrained='Half'),
    'pseudo':       dict(epochs=50, lr=2e-4, wd=3e-3, dropout=0.0, pretrained='Half'),
}

# encoder 缓存目录（相对 CODES_ROOT）
PRETRAIN_CACHE_DIR = 'results/mtl_dm_transfer/pretrain_cache'

# 预训练时用于切分 pd 训练/验证的 fold（取 fold 0：~80% 训练）
_PRETRAIN_FOLD_IDX = 0
_PRETRAIN_N_FOLDS = 5

# 进程内 encoder 缓存：sig -> encoder_state_dict
_INPROC_ENCODER_CACHE: Dict[str, Dict[str, torch.Tensor]] = {}


# ==================== 路径辅助 ====================

def _resolve_project_path(p: str) -> str:
    """把相对路径解析为绝对路径（优先 CODES_ROOT，再 REPO_ROOT）。"""
    p_obj = Path(p)
    if p_obj.is_absolute():
        return str(p_obj)
    for root in (CODES_ROOT, REPO_ROOT):
        cand = Path(root) / p_obj
        if cand.exists():
            return str(cand)
    return str(Path(CODES_ROOT) / p_obj)


# ==================== DM 数据构造 ====================

def build_dm_patients(task: str) -> List[Dict[str, Any]]:
    """把 DM 数据集映射为 ``PDImageDataset`` 可用的患者字典列表。

    - dm_binary ：control→0(group='control')，NDR/NPDR→1(group='pd')，80 患者。
    - dm_subtype：排除 control；NDR→0(group='control')，NPDR→1(group='pd')，60 患者。

    所有 DM 图像放入 image_paths['central']，whorl 留空（PDImageDataset 仅按
    group 推断二分类标签，stage/subtype 取 -1 被忽略）。
    """
    if task not in DM_TASKS:
        raise ValueError(f"未知 DM 任务: {task}，可选: {DM_TASKS}")

    dm_json_abs = _resolve_project_path(DM_JSON)
    with open(dm_json_abs, 'r', encoding='utf-8') as f:
        raw = json.load(f)

    patients: List[Dict[str, Any]] = []
    for rec in raw:
        grp = rec['group']
        imgs = rec.get('image_paths', []) or []
        if not imgs:
            continue

        if task == 'dm_binary':
            label = 0 if grp == 'control' else 1
        else:  # DM 亚型任务
            if grp == 'control':
                continue
            label = _DM_SUBTYPE_MAP[grp]

        patients.append({
            'patient_id': rec.get('patient_id', rec.get('name', '')),
            'group': 'pd' if label == 1 else 'control',
            'stage': 0,
            'subtype': '',
            'image_paths': {'central': list(imgs), 'whorl': []},
        })
    return patients


# ==================== 预训练签名与缓存 ====================

def _pretrain_signature(backbone: str, branch_a_task: str) -> str:
    hp = _PRETRAIN_HP[branch_a_task]
    payload = {
        'backbone': backbone,
        'branch_a_task': branch_a_task,
        'epochs': hp['epochs'],
        'lr': hp['lr'],
        'wd': hp['wd'],
        'dropout': hp['dropout'],
        'pretrained': hp['pretrained'],
        'pretrain_seed': PRETRAIN_SEED,
        'fold_idx': _PRETRAIN_FOLD_IDX,
        'n_folds': _PRETRAIN_N_FOLDS,
    }
    raw = json.dumps(payload, sort_keys=True).encode('utf-8')
    return hashlib.sha1(raw).hexdigest()[:12]


def _pretrain_mtl_encoder(backbone: str, branch_a_task: str,
                          device: str, log) -> Dict[str, torch.Tensor]:
    """在 pd_dataset_en (+ CORN-1) 上做一次多任务训练，返回 encoder 权重。"""
    hp = _PRETRAIN_HP[branch_a_task]

    cfg = MTLExperimentConfig()
    cfg.BACKBONE = backbone
    cfg.BRANCH_A_TASK = branch_a_task
    cfg.NUM_EPOCHS = int(hp['epochs'])
    cfg.LEARNING_RATE = float(hp['lr'])
    cfg.WEIGHT_DECAY = float(hp['wd'])
    cfg.DROPOUT = float(hp['dropout'])
    cfg.PRETRAINED = hp['pretrained']
    cfg.NUM_FOLDS = _PRETRAIN_N_FOLDS
    cfg.SEED = PRETRAIN_SEED
    cfg.DEVICE = device
    cfg.NUM_WORKERS = 0

    corn1_root = _resolve_project_path(cfg.CORN1_ROOT)
    pd_json = _resolve_project_path(cfg.PD_DATASET_JSON)
    datasets_base = _resolve_project_path(cfg.DATASETS_BASE)

    set_seed(PRETRAIN_SEED)

    use_pseudo = (branch_a_task == 'pseudo')
    corn1_train, corn1_val = build_corn1_loaders(
        corn1_root,
        batch_size=cfg.BATCH_SIZE,
        num_workers=cfg.NUM_WORKERS,
        augment_train=cfg.DATA_AUGMENTATION,
        image_size=cfg.IMAGE_SIZE,
        generate_pseudo=use_pseudo,
    )
    pd_train, pd_val = build_pd_patient_kfold_loaders(
        dataset_json=pd_json,
        fold_idx=_PRETRAIN_FOLD_IDX,
        n_folds=_PRETRAIN_N_FOLDS,
        datasets_base=datasets_base,
        dataset_mode=cfg.DATASET_MODE,
        batch_size=cfg.BATCH_SIZE,
        num_workers=cfg.NUM_WORKERS,
        augment_train=cfg.DATA_AUGMENTATION,
        image_size=cfg.IMAGE_SIZE,
        random_state=PRETRAIN_SEED,
    )
    mtl_loader = MultiTaskDataLoader(corn1_train, pd_train)

    pretrained_bool = (cfg.PRETRAINED is True or cfg.PRETRAINED == 'Half')
    model = UNetMultiTask(
        encoder_type=cfg.BACKBONE,
        encoder_pretrained=pretrained_bool,
        classification_mode='dual_stream',
        num_pd_classes=2,
        num_stage_classes=5,
        num_subtype_classes=3,
        mae_enabled=(branch_a_task == 'mae'),
        dropout_rate=cfg.DROPOUT,
    )
    if cfg.PRETRAINED == 'Half':
        _freeze_backbone_half(model)

    cache_dir = _resolve_project_path(PRETRAIN_CACHE_DIR)
    os.makedirs(cache_dir, exist_ok=True)
    fold_dir = os.path.join(cache_dir, f'_pretrain_{backbone}_{branch_a_task}')

    log(f"[dm-transfer] 在 pd_dataset_en (+CORN-1) 上做一次 MTL 预训练 "
        f"(backbone={backbone}, branch_a={branch_a_task}, epochs={cfg.NUM_EPOCHS}) ...")
    trainer = FoldTrainer(
        model=model,
        config=cfg,
        mtl_loader=mtl_loader,
        val_corn1_loader=corn1_val,
        val_pd_loader=pd_val,
        fold_dir=fold_dir,
    )
    trainer.run()

    # 载入早停期间保存的最优权重（best_model.pth 为完整模型 state_dict）
    best_ckpt = os.path.join(fold_dir, 'best_model.pth')
    if os.path.exists(best_ckpt) and os.path.getsize(best_ckpt) > 0:
        full_state = torch.load(best_ckpt, map_location='cpu')
        model.load_state_dict(full_state)
        log(f"[dm-transfer] 已载入预训练最优权重: {best_ckpt}")

    encoder_state = {k: v.detach().cpu().clone()
                     for k, v in model.encoder.state_dict().items()}
    return encoder_state


def get_or_build_pd_encoder(backbone: str, branch_a_task: str,
                            device: str, log) -> Dict[str, torch.Tensor]:
    """获取在 pd_dataset_en 上预训练好的 encoder 权重（复用优先，必要时训练一次）。

    复用优先级：
    1. 环境变量 ``RUN_MTL_PRETRAIN_CKPT`` 指定的 encoder ckpt；
    2. 进程内缓存；
    3. 磁盘缓存（按签名）；
    4. 都没有则在 pd_dataset_en (+CORN-1) 上预训练一次并原子化写盘。
    """
    sig = _pretrain_signature(backbone, branch_a_task)

    # 1) 显式 ckpt 覆盖
    env_ckpt = os.environ.get('RUN_MTL_PRETRAIN_CKPT', '').strip()
    if env_ckpt:
        ckpt_abs = _resolve_project_path(env_ckpt)
        if os.path.exists(ckpt_abs):
            log(f"[dm-transfer] 复用显式 encoder ckpt: {ckpt_abs}")
            state = torch.load(ckpt_abs, map_location='cpu')
            enc_state = state.get('encoder_state', state) if isinstance(state, dict) else state
            return enc_state
        log(f"[dm-transfer][warn] RUN_MTL_PRETRAIN_CKPT 指定文件不存在: {ckpt_abs}")

    # 2) 进程内缓存
    if sig in _INPROC_ENCODER_CACHE:
        return _INPROC_ENCODER_CACHE[sig]

    # 3) 磁盘缓存
    cache_dir = _resolve_project_path(PRETRAIN_CACHE_DIR)
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{backbone}_{branch_a_task}_{sig}.pt")
    if os.path.exists(cache_path) and os.path.getsize(cache_path) > 0:
        log(f"[dm-transfer] 复用 pd_dataset_en MTL encoder 缓存（跳过预训练）:\n"
            f"             {cache_path}")
        payload = torch.load(cache_path, map_location='cpu')
        enc_state = payload['encoder_state']
        _INPROC_ENCODER_CACHE[sig] = enc_state
        return enc_state

    # 4) 预训练一次
    enc_state = _pretrain_mtl_encoder(backbone, branch_a_task, device, log)

    payload = {
        'encoder_state': enc_state,
        'meta': {
            'backbone': backbone,
            'branch_a_task': branch_a_task,
            'signature': sig,
            'pretrain_hp': _PRETRAIN_HP[branch_a_task],
            'pretrain_seed': PRETRAIN_SEED,
            'created': datetime.now().isoformat(timespec='seconds'),
        },
    }
    tmp_path = f"{cache_path}.tmp.{os.getpid()}"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, cache_path)
    log(f"[dm-transfer] 已缓存 pd MTL encoder: {cache_path}")

    _INPROC_ENCODER_CACHE[sig] = enc_state
    return enc_state


# ==================== 迁移模型（encoder + 分类头）====================

class MTLTransferModel(nn.Module):
    """复用 MTL encoder 表征 + 一个全局池化分类头，用于 DM 二分类迁移评估。"""

    def __init__(self, backbone: str, num_classes: int = 2,
                 dropout: float = 0.3, freeze_encoder: bool = True):
        super().__init__()
        self.encoder = UNetEncoder(backbone, pretrained=False)
        feat_dim = self.encoder.bottleneck_channels
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, num_classes),
        )
        self._freeze_encoder = freeze_encoder
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def load_encoder(self, encoder_state: Dict[str, torch.Tensor]):
        self.encoder.load_state_dict(encoder_state)

    def train(self, mode: bool = True):
        super().train(mode)
        # 冻结 encoder 时让其 BN/Dropout 始终处于 eval，稳定迁移表征
        if self._freeze_encoder:
            self.encoder.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._freeze_encoder:
            with torch.no_grad():
                feats = self.encoder(x)
        else:
            feats = self.encoder(x)
        pooled = self.pool(feats[-1])
        return self.head(pooled)


# ==================== DM 下游训练 ====================

def _compute_metrics(logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    preds = logits.argmax(dim=1).cpu().numpy()
    labs = labels.cpu().numpy()
    probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
    acc = float((preds == labs).mean())
    f1 = float(f1_score(labs, preds, average='macro', zero_division=0))
    try:
        auc = float(roc_auc_score(labs, probs))
    except ValueError:
        auc = 0.0
    return {'acc': acc, 'f1': f1, 'auc': auc}


def _make_loader(patients: List[Dict[str, Any]], datasets_base: str,
                 augment: bool, image_size: int, batch_size: int,
                 num_workers: int, shuffle: bool, drop_last: bool) -> DataLoader:
    ds = PDImageDataset(patients, datasets_base=datasets_base,
                        augment=augment, image_size=image_size)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True, drop_last=drop_last)


def _class_weights(patients: List[Dict[str, Any]], device: str) -> torch.Tensor:
    counts = [0, 0]
    for p in patients:
        counts[1 if p['group'] == 'pd' else 0] += 1
    total = sum(counts)
    w = [total / (2.0 * c) if c > 0 else 0.0 for c in counts]
    return torch.tensor(w, dtype=torch.float32, device=device)


def _train_one_fold(config: MTLExperimentConfig,
                    encoder_state: Dict[str, torch.Tensor],
                    train_patients: List[Dict[str, Any]],
                    val_patients: List[Dict[str, Any]],
                    datasets_base: str, log,
                    output_dir: Optional[str] = None,
                    task: str = 'dm',
                    fold_idx: int = 1) -> Dict[str, float]:
    device = config.DEVICE
    freeze = (config.BACKBONE_LR_RATIO <= 0.0)

    model = MTLTransferModel(
        backbone=config.BACKBONE,
        num_classes=2,
        dropout=config.DROPOUT,
        freeze_encoder=freeze,
    ).to(device)
    model.load_encoder(encoder_state)

    train_loader = _make_loader(
        train_patients, datasets_base, augment=config.DATA_AUGMENTATION,
        image_size=config.IMAGE_SIZE, batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS, shuffle=True, drop_last=True,
    )
    val_loader = _make_loader(
        val_patients, datasets_base, augment=False,
        image_size=config.IMAGE_SIZE, batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS, shuffle=False, drop_last=False,
    )

    # 优化器：冻结时只训练分类头；否则 encoder 用更小学习率
    if freeze:
        params = [{'params': model.head.parameters(), 'lr': config.LEARNING_RATE}]
    else:
        params = [
            {'params': model.encoder.parameters(),
             'lr': config.LEARNING_RATE * config.BACKBONE_LR_RATIO},
            {'params': model.head.parameters(), 'lr': config.LEARNING_RATE},
        ]
    optimizer = torch.optim.AdamW(params, weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.NUM_EPOCHS, eta_min=1e-7)

    criterion = nn.CrossEntropyLoss(weight=_class_weights(train_patients, device))

    best_score = -1.0
    best_metrics: Dict[str, float] = {'acc': 0.0, 'f1': 0.0, 'auc': 0.0}
    best_train_acc = 0.0
    patience_counter = 0

    for epoch in range(1, config.NUM_EPOCHS + 1):
        model.train()
        tr_correct, tr_total, run_loss = 0, 0, 0.0
        for batch in train_loader:
            imgs = batch['image'].to(device)
            labels = batch['pd_label'].to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            if config.GRAD_CLIP and config.GRAD_CLIP > 0:
                nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    config.GRAD_CLIP)
            optimizer.step()
            run_loss += float(loss.item())
            tr_correct += int((logits.argmax(1) == labels).sum().item())
            tr_total += labels.numel()
        scheduler.step()
        train_acc = tr_correct / max(tr_total, 1)

        # 验证
        model.eval()
        all_logits, all_labels, all_patient_ids = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch['image'].to(device)
                logits = model(imgs)
                all_logits.append(logits.cpu())
                all_labels.append(batch['pd_label'])
                dataset = getattr(val_loader, 'dataset', None)
                patients = getattr(dataset, 'patients', None)
                for idx in batch['patient_idx'].detach().cpu().tolist():
                    if patients is not None and 0 <= int(idx) < len(patients):
                        all_patient_ids.append(str(patients[int(idx)].get('patient_id', idx)))
                    else:
                        all_patient_ids.append(str(idx))
        v_logits = torch.cat(all_logits)
        v_labels = torch.cat(all_labels)
        metrics = _compute_metrics(v_logits, v_labels)
        patient_rows = aggregate_patient_outputs(
            all_patient_ids,
            v_labels.cpu().numpy().tolist(),
            logits=v_logits.cpu().numpy(),
        )
        metrics['patient_acc'] = float(np.mean([r['correct'] for r in patient_rows])) if patient_rows else 0.0
        score = metrics['patient_acc']

        improved = score > best_score
        if improved:
            if output_dir:
                patient_result = save_fold_patient_outputs(
                    output_dir, task, '', fold_idx,
                    all_patient_ids,
                    v_labels.cpu().numpy().tolist(),
                    logits=v_logits.cpu().numpy(),
                )
                metrics['patient_acc'] = patient_result['patient_acc']
                metrics['patient_logits_json'] = patient_result['patient_logits_json']
                metrics['patient_logits_csv'] = patient_result['patient_logits_csv']
            best_score = score
            best_metrics = metrics
            best_train_acc = train_acc
            patience_counter = 0
        else:
            patience_counter += 1

        log(f"    E{epoch:03d}/{config.NUM_EPOCHS:03d} loss={run_loss/max(len(train_loader),1):.4f} "
            f"tr_acc={train_acc:.3f} | val acc={metrics['acc']:.3f} "
            f"f1={metrics['f1']:.3f} auc={metrics['auc']:.3f} "
            f"pat={metrics['patient_acc']:.3f}{' ↑' if improved else ''}")

        if config.EARLY_STOPPING and patience_counter >= config.PATIENCE:
            log(f"    早停触发 (patience={config.PATIENCE})")
            break

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        'acc': best_metrics['acc'],
        'patient_acc': best_metrics.get('patient_acc', 0.0),
        'patient_logits_json': best_metrics.get('patient_logits_json', ''),
        'patient_logits_csv': best_metrics.get('patient_logits_csv', ''),
        'f1': best_metrics['f1'],
        'auc': best_metrics['auc'],
        'train_acc': best_train_acc,
    }


def _run_dm_task_kfold(config: MTLExperimentConfig,
                       encoder_state: Dict[str, torch.Tensor],
                       task: str, log,
                       output_dir: Optional[str] = None) -> Dict[str, Any]:
    """对单个 DM 任务执行 patient-level K-fold（复用 pd encoder）。"""
    datasets_base = _resolve_project_path(config.DATASETS_BASE)
    patients = build_dm_patients(task)

    fold_metrics: List[Dict[str, float]] = []
    for fold_idx in range(config.NUM_FOLDS):
        train_patients, val_patients = _patient_kfold_split(
            patients, fold_idx, config.NUM_FOLDS, random_state=config.SEED)
        log(f"\n  ── {task} | Fold {fold_idx + 1}/{config.NUM_FOLDS} "
            f"(train_pat={len(train_patients)}, val_pat={len(val_patients)}) ──")
        m = _train_one_fold(config, encoder_state, train_patients,
                            val_patients, datasets_base, log,
                            output_dir=output_dir, task=task, fold_idx=fold_idx + 1)
        m['fold'] = fold_idx
        fold_metrics.append(m)
        log(f"  Fold {fold_idx + 1} best: acc={m['acc']:.4f} f1={m['f1']:.4f} "
            f"auc={m['auc']:.4f} pat={m.get('patient_acc', 0.0):.4f}")

    accs = [m['acc'] for m in fold_metrics]
    f1s = [m['f1'] for m in fold_metrics]
    aucs = [m['auc'] for m in fold_metrics]
    patient_accs = [m.get('patient_acc', 0.0) for m in fold_metrics]
    train_accs = [m['train_acc'] for m in fold_metrics]
    return {
        'task': task,
        'fold_metrics': fold_metrics,
        'avg_train_acc': float(np.mean(train_accs)),
        'avg_acc': float(np.mean(accs)),
        'std_acc': float(np.std(accs)),
        'avg_patient_acc': float(np.mean(patient_accs)),
        'std_patient_acc': float(np.std(patient_accs)),
        'avg_f1': float(np.mean(f1s)),
        'std_f1': float(np.std(f1s)),
        'avg_auc': float(np.mean(aucs)),
        'num_folds': len(fold_metrics),
    }


# ==================== 单实验运行 ====================

class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


def _write_dm_report(report_path: str, config: MTLExperimentConfig,
                     task_results: List[Dict[str, Any]]):
    lines = [
        '# MTL → DM 迁移实验报告',
        '',
        f"- backbone: `{config.BACKBONE}`",
        f"- 预训练线 (branch_a_task): `{config.BRANCH_A_TASK}`",
        f"- 迁移策略: {'冻结 encoder（线性探针）' if config.BACKBONE_LR_RATIO <= 0 else f'微调 encoder (lr×{config.BACKBONE_LR_RATIO})'}",
        f"- DM 微调: epochs={config.NUM_EPOCHS}, lr={config.LEARNING_RATE:.0e}, "
        f"wd={config.WEIGHT_DECAY:.0e}, dropout={config.DROPOUT}, "
        f"folds={config.NUM_FOLDS}, seed={config.SEED}",
        '',
        '| 任务 | 验证 Acc | 验证 F1 | 验证 AUC | 训练 Acc | 折数 |',
        '|------|----------|---------|----------|----------|------|',
    ]
    for r in task_results:
        lines.append(
            f"| {r['task']} | {r['avg_acc']:.4f} ± {r['std_acc']:.4f} "
            f"| {r['avg_f1']:.4f} ± {r['std_f1']:.4f} "
            f"| {r['avg_auc']:.4f} | {r['avg_train_acc']:.4f} | {r['num_folds']} |"
        )
    lines.append('')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def run_single_dm_transfer_experiment(config: MTLExperimentConfig,
                                      output_dir: Optional[str] = None,
                                      tasks: Optional[List[str]] = None
                                      ) -> Dict[str, Any]:
    """运行单个 MTL→DM 迁移实验（复用 pd encoder，重训 DM 分类头）。"""
    tasks = tasks or DM_TASKS

    exp_dir = output_dir or os.path.join(
        CODES_ROOT, 'results/mtl_dm_transfer',
        datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(exp_dir, exist_ok=True)

    log_f = open(os.path.join(exp_dir, 'output.txt'), 'a', encoding='utf-8', buffering=1)
    _orig_stdout = sys.stdout
    sys.stdout = _Tee(_orig_stdout, log_f)

    def log(msg: str = ''):
        print(msg, flush=True)

    try:
        log('=' * 70)
        log(f"MTL→DM 迁移实验 | backbone={config.BACKBONE} "
            f"branch_a={config.BRANCH_A_TASK} tasks={tasks}")
        log(f"DM 微调: ep={config.NUM_EPOCHS} lr={config.LEARNING_RATE:.0e} "
            f"wd={config.WEIGHT_DECAY:.0e} drop={config.DROPOUT} "
            f"bblr_ratio={config.BACKBONE_LR_RATIO} folds={config.NUM_FOLDS} "
            f"seed={config.SEED}")
        log('=' * 70)

        # 1) 取/建 pd encoder（按线缓存复用）
        encoder_state = get_or_build_pd_encoder(
            config.BACKBONE, config.BRANCH_A_TASK, config.DEVICE, log)

        # 2) 逐任务迁移微调
        set_seed(config.SEED)
        task_results: List[Dict[str, Any]] = []
        for task in tasks:
            t0 = time.time()
            res = _run_dm_task_kfold(config, encoder_state, task, log, output_dir=exp_dir)
            res['duration_min'] = (time.time() - t0) / 60.0
            task_results.append(res)
            log(f"\n[{task}] 平均 acc={res['avg_acc']:.4f}±{res['std_acc']:.4f} "
                f"pat={res.get('avg_patient_acc', 0.0):.4f}±{res.get('std_patient_acc', 0.0):.4f} "
                f"f1={res['avg_f1']:.4f}±{res['std_f1']:.4f} "
                f"auc={res['avg_auc']:.4f}  ({res['duration_min']:.1f}min)")

        # 3) 保存结果
        summary = {
            'backbone': config.BACKBONE,
            'branch_a_task': config.BRANCH_A_TASK,
            'transfer': 'dm',
            'config': {
                'num_epochs': config.NUM_EPOCHS,
                'lr': config.LEARNING_RATE,
                'weight_decay': config.WEIGHT_DECAY,
                'dropout': config.DROPOUT,
                'backbone_lr_ratio': config.BACKBONE_LR_RATIO,
                'num_folds': config.NUM_FOLDS,
                'seed': config.SEED,
            },
            'task_results': task_results,
            'output_dir': exp_dir,
        }
        with open(os.path.join(exp_dir, 'results.json'), 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        _write_dm_report(os.path.join(exp_dir, 'experiment_report.md'),
                         config, task_results)
        log(f"\n结果保存至: {exp_dir}")
        return summary
    finally:
        sys.stdout = _orig_stdout
        log_f.close()
