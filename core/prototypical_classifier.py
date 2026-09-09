"""ResNet18 episodic prototype classifier and patient-level evaluation."""
import copy
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

from core.basic_models import (
    get_backbone as _get_basic_backbone,
    get_backbone_feature_dim as _get_basic_feature_dim,
    get_image_size as _get_basic_image_size,
)
from core.classification_trainer import EarlyStopping


# Backbone 路由：使用 basic_models 的名称集合


def _get_backbone(backbone: str, pretrained) -> nn.Module:
    return _get_basic_backbone(backbone, pretrained)


def _get_feature_dim(backbone: str) -> int:
    return _get_basic_feature_dim(backbone)


def _get_image_size(backbone: str, pretrained) -> tuple:
    return _get_basic_image_size(backbone, pretrained)


TASK_CONFIG = {
    'pd': {
        'name': 'PD_Classification',
        'num_classes': 2,
        'class_names': ['Non-PD', 'PD'],
        'label_key': 'pd',
        'label_attr': 'PD_labels',
    },
    'stage': {
        'name': 'Stage_Classification',
        'num_classes': 3,
        'class_names': ['Stage 1', 'Stage 2', 'Stage 3'],
        'label_key': 'stage',
        'label_attr': 'stage_labels',
    },
    'subtype': {
        'name': 'Subtype_Classification',
        'num_classes': 3,
        'class_names': ['TD', 'PIGD', 'Mixed'],
        'label_key': 'subtype',
        'label_attr': 'subtype_labels',
    },
    'pd_subtype_tdrest': {
        'name': 'Subtype_TDvsRest',
        'num_classes': 2,
        'class_names': ['TD', 'Rest'],
        'label_key': 'subtype',
        'label_attr': 'subtype_labels',
    },
    'pd_subtype_pigd_mixed': {
        'name': 'Subtype_PIGDvsMixed',
        'num_classes': 2,
        'class_names': ['PIGD', 'Mixed'],
        'label_key': 'subtype',
        'label_attr': 'subtype_labels',
    },
    'pd_subtype_hierarchy': {
        'name': 'Subtype_Hierarchy',
        'num_classes': 3,
        'class_names': ['TD', 'PIGD', 'Mixed'],
        'label_key': 'subtype',
        'label_attr': 'subtype_labels',
    },
}


class ProtoClassifierConfig:
    """原型学习分类器配置"""

    ARCHITECTURE: Literal['single_stream', 'mixed_single_stream', 'dual_stream'] = 'single_stream'
    BACKBONE: Literal['resnet18'] = 'resnet18'
    PRETRAINED: Literal[True, False, 'Half'] = True

    EMBEDDING_DIM: Literal[128, 256] = 128
    FEATURE_DROPOUT: float = 0.2

    N_WAY: int = 3
    N_SHOT: int = 3
    N_QUERY: int = 4
    TRAIN_EPISODES_PER_EPOCH: int = 80
    VAL_EPISODES: int = 80

    NUM_EPOCHS: int = 40
    LEARNING_RATE: float = 5e-4
    WEIGHT_DECAY: float = 1e-4
    LR_SCHEDULER: str = 'cosine'  # 'cosine' / 'none'

    EARLY_STOPPING: bool = True
    PATIENCE: int = 8

    DATA_AUGMENTATION: bool = True
    AUG_HORIZONTAL_FLIP: float = 0.5
    AUG_ROTATION: float = 10.0
    AUG_BRIGHTNESS: float = 0.2
    AUG_CONTRAST: float = 0.2
    AUG_SATURATION: float = 0.0
    AUG_HUE: float = 0.0

    BATCH_SIZE_INFERENCE: int = 64
    NUM_WORKERS: int = 4
    DEVICE: str = 'cuda' if torch.cuda.is_available() else 'cpu'


class PrototypicalEncoder(nn.Module):
    """轻量化特征编码器: Backbone -> Linear 投影 -> L2 归一化"""

    def __init__(
        self,
        backbone: str,
        embedding_dim: int = 128,
        pretrained: Literal[True, False, 'Half'] = True,
        feature_dropout: float = 0.2,
    ):
        super().__init__()
        self.backbone = _get_backbone(backbone, pretrained)
        feat_dim = _get_feature_dim(backbone)
        self.dropout = nn.Dropout(feature_dropout)
        self.proj = nn.Linear(feat_dim, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        feat = self.dropout(feat)
        emb = self.proj(feat)
        return F.normalize(emb, p=2, dim=1)


class DualStreamPrototypicalEncoder(nn.Module):
    """双流特征编码器: Central+Whorl 独立 Backbone → 拼接投影 → L2 归一化"""

    def __init__(
        self,
        backbone: str,
        embedding_dim: int = 128,
        pretrained: Literal[True, False, 'Half'] = True,
        feature_dropout: float = 0.2,
    ):
        super().__init__()
        self.backbone_central = _get_backbone(backbone, pretrained)
        self.backbone_whorl = _get_backbone(backbone, pretrained)
        feat_dim = _get_feature_dim(backbone) * 2  # 两路拼接后翻倍
        self.dropout = nn.Dropout(feature_dropout)
        self.proj = nn.Linear(feat_dim, embedding_dim)

    def forward(self, central: torch.Tensor, whorl: torch.Tensor) -> torch.Tensor:
        f_c = self.backbone_central(central)
        f_w = self.backbone_whorl(whorl)
        feat = torch.cat([f_c, f_w], dim=1)
        feat = self.dropout(feat)
        emb = self.proj(feat)
        return F.normalize(emb, p=2, dim=1)


def _pairwise_sq_euclidean(query: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
    """返回 query 到每个 prototype 的平方欧氏距离矩阵"""
    return torch.cdist(query, prototypes, p=2).pow(2)


def _compute_prototypes(support_emb: torch.Tensor, support_labels: torch.Tensor, n_way: int) -> torch.Tensor:
    """按 episode 内标签计算原型均值"""
    protos: List[torch.Tensor] = []
    for c in range(n_way):
        class_emb = support_emb[support_labels == c]
        protos.append(class_emb.mean(dim=0))
    return torch.stack(protos, dim=0)


def _get_task_labels(dataset: Dataset, task: str) -> List[int]:
    """从 Dataset/Subset 提取指定任务标签"""
    if task not in TASK_CONFIG:
        raise ValueError(f"未知任务: {task}")

    attr = TASK_CONFIG[task]['label_attr']

    if isinstance(dataset, Subset):
        base = dataset.dataset
        indices = list(dataset.indices)
        if hasattr(base, attr):
            all_labels = getattr(base, attr)
            return [_remap_task_label(task, int(all_labels[i])) for i in indices]
    else:
        if hasattr(dataset, attr):
            return [_remap_task_label(task, int(v)) for v in getattr(dataset, attr)]

    label_pos = {
        'pd': 1,
        'stage': 2,
        'subtype': 3,
        'pd_subtype_tdrest': 3,
        'pd_subtype_pigd_mixed': 3,
    }[task]
    labels: List[int] = []
    for i in range(len(dataset)):
        item = dataset[i]
        labels.append(_remap_task_label(task, int(item[label_pos])))
    return labels


def _remap_task_label(task: str, label: int) -> int:
    if task == 'pd_subtype_tdrest':
        if label == 0:
            return 0
        if label in (1, 2):
            return 1
        return -1
    if task == 'pd_subtype_pigd_mixed':
        if label == 1:
            return 0
        if label == 2:
            return 1
        return -1
    return label


class EpisodicIndexSampler:
    """在固定数据集上构建并采样 episodic 支持集/查询集索引"""

    def __init__(self, dataset: Dataset, task: str):
        labels = _get_task_labels(dataset, task)
        class_to_indices: Dict[int, List[int]] = {}
        for i, y in enumerate(labels):
            if y < 0:
                continue
            class_to_indices.setdefault(y, []).append(i)

        self.dataset = dataset
        self.task = task
        self.class_to_indices = class_to_indices

        valid_classes = sorted([c for c, idxs in class_to_indices.items() if len(idxs) > 0])
        if len(valid_classes) < 2:
            raise ValueError(
                f"任务 {task} 可用类别不足，至少需要 2 类，当前仅 {len(valid_classes)} 类"
            )

        self.valid_classes = valid_classes

    def sample_episode(
        self,
        n_way: int,
        n_shot: int,
        n_query: int,
        rng: np.random.RandomState,
    ) -> Tuple[List[int], List[int], List[int], List[int]]:
        if n_way > len(self.valid_classes):
            raise ValueError(
                f"n_way={n_way} 超过可用类别数 {len(self.valid_classes)}，请降低 N_WAY"
            )

        chosen_raw = rng.choice(self.valid_classes, size=n_way, replace=False)
        use_global_labels = (
            n_way == len(self.valid_classes)
            and self.valid_classes == list(range(n_way))
        )
        support_idx: List[int] = []
        support_y: List[int] = []
        query_idx: List[int] = []
        query_y: List[int] = []

        need = n_shot + n_query
        for epi_label, raw_label in enumerate(chosen_raw):
            label = int(raw_label) if use_global_labels else epi_label
            pool = self.class_to_indices[int(raw_label)]
            replace = len(pool) < need
            sampled = rng.choice(pool, size=need, replace=replace)

            support_part = sampled[:n_shot]
            query_part = sampled[n_shot:]

            support_idx.extend([int(i) for i in support_part])
            support_y.extend([label] * n_shot)
            query_idx.extend([int(i) for i in query_part])
            query_y.extend([label] * n_query)

        return support_idx, support_y, query_idx, query_y


def _gather_episode_images(
    dataset: Dataset, indices: List[int], device: str
):
    """收集 episode 图像；dual_stream 返回 (central_batch, whorl_batch) 元组，单流返回 Tensor。"""
    raw: List = []
    for i in indices:
        sample = dataset[i]
        raw.append(sample[0])

    if isinstance(raw[0], (tuple, list)):
        # dual_stream: 每个 sample[0] 是 (central_tensor, whorl_tensor)
        central_batch = torch.stack([img[0] for img in raw], dim=0).to(device)
        whorl_batch = torch.stack([img[1] for img in raw], dim=0).to(device)
        return central_batch, whorl_batch

    return torch.stack(raw, dim=0).to(device)


def _run_episode(
    model: nn.Module,
    dataset: Dataset,
    sampler: EpisodicIndexSampler,
    cfg: ProtoClassifierConfig,
    optimizer: Optional[torch.optim.Optimizer],
    rng: np.random.RandomState,
) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    support_idx, support_y, query_idx, query_y = sampler.sample_episode(
        cfg.N_WAY,
        cfg.N_SHOT,
        cfg.N_QUERY,
        rng,
    )

    support_x = _gather_episode_images(dataset, support_idx, cfg.DEVICE)
    query_x = _gather_episode_images(dataset, query_idx, cfg.DEVICE)

    support_y_t = torch.tensor(support_y, dtype=torch.long, device=cfg.DEVICE)
    query_y_t = torch.tensor(query_y, dtype=torch.long, device=cfg.DEVICE)

    if isinstance(support_x, tuple):
        # dual_stream: 解包 (central, whorl)
        support_emb = model(*support_x)
        query_emb = model(*query_x)
    else:
        support_emb = model(support_x)
        query_emb = model(query_x)

    prototypes = _compute_prototypes(support_emb, support_y_t, cfg.N_WAY)
    dists = _pairwise_sq_euclidean(query_emb, prototypes)
    logits = -dists

    loss = F.cross_entropy(logits, query_y_t)

    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    probs = torch.softmax(logits, dim=1)
    preds = probs.argmax(dim=1)
    acc = (preds == query_y_t).float().mean().item()

    return (
        float(loss.item()),
        float(acc),
        query_y_t.detach().cpu().numpy(),
        preds.detach().cpu().numpy(),
        probs.detach().cpu().numpy(),
    )


class FewShotPrototypicalClassifier:
    """小样本原型学习分类器"""

    def __init__(self, config: ProtoClassifierConfig):
        self.config = config
        self.models: Dict[str, Any] = {}
        self._init_transforms()

    def _init_transforms(self):
        resize, crop, use_original = _get_image_size(
            self.config.BACKBONE,
            self.config.PRETRAINED,
        )
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        base = [] if use_original else [
            transforms.Resize((resize, resize)),
            transforms.CenterCrop(crop),
        ]

        augment: List[Any] = []
        if self.config.DATA_AUGMENTATION:
            if self.config.AUG_HORIZONTAL_FLIP > 0:
                augment.append(transforms.RandomHorizontalFlip(p=self.config.AUG_HORIZONTAL_FLIP))
            if self.config.AUG_ROTATION > 0:
                augment.append(transforms.RandomRotation(self.config.AUG_ROTATION))
            if any([
                self.config.AUG_BRIGHTNESS > 0,
                self.config.AUG_CONTRAST > 0,
                self.config.AUG_SATURATION > 0,
                self.config.AUG_HUE > 0,
            ]):
                augment.append(
                    transforms.ColorJitter(
                        brightness=self.config.AUG_BRIGHTNESS,
                        contrast=self.config.AUG_CONTRAST,
                        saturation=self.config.AUG_SATURATION,
                        hue=self.config.AUG_HUE,
                    )
                )

        self.val_transform = transforms.Compose(base + [transforms.ToTensor(), normalize])
        self.train_transform = transforms.Compose(base + augment + [transforms.ToTensor(), normalize])

    def _create_model(self) -> nn.Module:
        if self.config.ARCHITECTURE == 'dual_stream':
            model = DualStreamPrototypicalEncoder(
                backbone=self.config.BACKBONE,
                embedding_dim=self.config.EMBEDDING_DIM,
                pretrained=self.config.PRETRAINED,
                feature_dropout=self.config.FEATURE_DROPOUT,
            )
        else:
            model = PrototypicalEncoder(
                backbone=self.config.BACKBONE,
                embedding_dim=self.config.EMBEDDING_DIM,
                pretrained=self.config.PRETRAINED,
                feature_dropout=self.config.FEATURE_DROPOUT,
            )
        return model.to(self.config.DEVICE)

    def _evaluate(self, model: nn.Module, dataset: Dataset, task: str, seed: int) -> Dict[str, Any]:
        cfg = self.config
        sampler = EpisodicIndexSampler(dataset, task)
        rng = np.random.RandomState(seed)

        available_classes = len(sampler.valid_classes)
        if available_classes < 2:
            raise ValueError(
                f"任务 {task} 在当前验证集可用类别数不足 2（仅 {available_classes} 类），无法进行 episodic 验证"
            )

        # 避免 N_WAY 超过当前数据集可用类别数（例如 pd 二分类）
        eval_cfg = copy.copy(cfg)
        eval_cfg.N_WAY = min(cfg.N_WAY, available_classes)

        model.eval()
        losses: List[float] = []
        accs: List[float] = []
        all_labels: List[int] = []
        all_preds: List[int] = []
        all_probs: List[np.ndarray] = []

        with torch.no_grad():
            for _ in range(cfg.VAL_EPISODES):
                loss, acc, labels, preds, probs = _run_episode(
                    model, dataset, sampler, eval_cfg, optimizer=None, rng=rng
                )
                losses.append(loss)
                accs.append(acc)
                all_labels.extend(labels.tolist())
                all_preds.extend(preds.tolist())
                all_probs.extend(list(probs))

        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels,
            all_preds,
            average='macro',
            zero_division=0,
        )

        return {
            'loss': float(np.mean(losses) if losses else 0.0),
            'acc': float(accuracy_score(all_labels, all_preds) if all_labels else 0.0),
            'episode_acc': float(np.mean(accs) if accs else 0.0),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'labels': all_labels,
            'preds': all_preds,
            'probs': np.array(all_probs),
        }

    def train(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset,
        task: str,
        fold_idx: int = 1,
        suffix: str = '',
    ) -> Dict[str, Any]:
        cfg = self.config
        model = self._create_model()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.LEARNING_RATE,
            weight_decay=cfg.WEIGHT_DECAY,
        )

        scheduler = None
        if cfg.LR_SCHEDULER == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(1, cfg.NUM_EPOCHS),
                eta_min=1e-6,
            )

        train_sampler = EpisodicIndexSampler(train_dataset, task)
        available_train_classes = len(train_sampler.valid_classes)
        if available_train_classes < 2:
            raise ValueError(
                f"任务 {task}{suffix} 在训练集可用类别数不足 2（仅 {available_train_classes} 类），无法进行 episodic 训练"
            )

        # 避免 N_WAY 超过当前训练集可用类别数（例如 pd 二分类）
        train_cfg = copy.copy(cfg)
        train_cfg.N_WAY = min(cfg.N_WAY, available_train_classes)

        early_stopper = EarlyStopping(patience=cfg.PATIENCE, min_delta=0.0005)

        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_f1': [],
        }

        best_state = copy.deepcopy(model.state_dict())
        best_val_acc = -1.0
        best_val_result: Dict[str, Any] = {}

        print(f"\n[Fold {fold_idx}] {task}{suffix} 原型学习训练开始")
        if train_cfg.N_WAY != cfg.N_WAY:
            print(
                f"N_WAY 自动调整: {cfg.N_WAY} -> {train_cfg.N_WAY} "
                f"(任务可用类别数={available_train_classes})"
            )
        print(
            f"Backbone={cfg.BACKBONE}, emb={cfg.EMBEDDING_DIM}, "
            f"episode={train_cfg.N_WAY}-way {cfg.N_SHOT}-shot {cfg.N_QUERY}-query"
        )

        for epoch in range(cfg.NUM_EPOCHS):
            model.train()
            epoch_losses: List[float] = []
            epoch_accs: List[float] = []

            rng = np.random.RandomState(2026 + fold_idx * 1000 + epoch)
            for _ in range(cfg.TRAIN_EPISODES_PER_EPOCH):
                loss, acc, _, _, _ = _run_episode(
                    model,
                    train_dataset,
                    train_sampler,
                    train_cfg,
                    optimizer=optimizer,
                    rng=rng,
                )
                epoch_losses.append(loss)
                epoch_accs.append(acc)

            if scheduler is not None:
                scheduler.step()

            train_loss = float(np.mean(epoch_losses) if epoch_losses else 0.0)
            train_acc = float(np.mean(epoch_accs) if epoch_accs else 0.0)
            val_result = self._evaluate(
                model,
                val_dataset,
                task,
                seed=9000 + fold_idx * 1000 + epoch,
            )

            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_result['loss'])
            history['val_acc'].append(val_result['acc'])
            history['val_f1'].append(val_result['f1'])

            print(
                f"Epoch {epoch+1:03d}/{cfg.NUM_EPOCHS} | "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                f"val_acc={val_result['acc']:.4f} val_f1={val_result['f1']:.4f}"
            )

            if val_result['acc'] > best_val_acc:
                best_val_acc = val_result['acc']
                best_state = copy.deepcopy(model.state_dict())
                best_val_result = val_result

            if cfg.EARLY_STOPPING and early_stopper(val_result['acc']):
                print(f"Early stopping at epoch {epoch+1}")
                break

        history['best_val_acc'] = float(best_val_acc if best_val_acc >= 0 else 0.0)

        model.load_state_dict(best_state)
        self.models[task + suffix] = model

        return {
            'model': model,
            'model_state': best_state,
            'val_result': best_val_result,
            'train_history': history,
        }

    def build_prototypes(
        self,
        model: nn.Module,
        support_dataset: Dataset,
        task: str,
    ) -> Tuple[torch.Tensor, List[int]]:
        """在给定支持集上计算全局类别原型"""
        labels = _get_task_labels(support_dataset, task)
        valid_indices = [i for i, y in enumerate(labels) if y >= 0]
        if not valid_indices:
            raise ValueError('支持集没有有效标签样本')

        class_to_local: Dict[int, List[int]] = {}
        for i in valid_indices:
            class_to_local.setdefault(labels[i], []).append(i)

        model.eval()
        class_ids = sorted(class_to_local.keys())
        prototypes: List[torch.Tensor] = []

        with torch.no_grad():
            for cid in class_ids:
                idxs = class_to_local[cid]
                loader = DataLoader(
                    Subset(support_dataset, idxs),
                    batch_size=self.config.BATCH_SIZE_INFERENCE,
                    shuffle=False,
                    num_workers=self.config.NUM_WORKERS,
                    pin_memory=(self.config.DEVICE != 'cpu'),
                )
                embs: List[torch.Tensor] = []
                for batch in loader:
                    x = batch[0]
                    if isinstance(x, (tuple, list)):
                        x = [part.to(self.config.DEVICE) for part in x]
                        embs.append(model(*x))
                    else:
                        x = x.to(self.config.DEVICE)
                        embs.append(model(x))
                proto = torch.cat(embs, dim=0).mean(dim=0)
                prototypes.append(proto)

        return torch.stack(prototypes, dim=0), class_ids

    def predict_with_prototypes(
        self,
        model: nn.Module,
        support_dataset: Dataset,
        query_dataset: Dataset,
        task: str,
    ) -> Dict[str, Any]:
        """用 support_dataset 的全局 prototype 预测 query_dataset 的每个样本。"""
        prototypes, class_ids = self.build_prototypes(model, support_dataset, task)
        labels_all = _get_task_labels(query_dataset, task)
        valid_indices = [i for i, y in enumerate(labels_all) if y >= 0]
        labels = [int(labels_all[i]) for i in valid_indices]
        if not valid_indices:
            raise ValueError(f"任务 {task} 在 query_dataset 中没有有效标签样本")
        if len(valid_indices) != len(labels_all):
            query_dataset = Subset(query_dataset, valid_indices)

        loader = DataLoader(
            query_dataset,
            batch_size=self.config.BATCH_SIZE_INFERENCE,
            shuffle=False,
            num_workers=self.config.NUM_WORKERS,
            pin_memory=(self.config.DEVICE != 'cpu'),
        )

        model.eval()
        preds: List[int] = []
        probs_out: List[List[float]] = []
        logits_out: List[List[float]] = []

        with torch.no_grad():
            for batch in loader:
                x = batch[0]
                if isinstance(x, (tuple, list)):
                    x = [part.to(self.config.DEVICE) for part in x]
                    emb = model(*x)
                else:
                    x = x.to(self.config.DEVICE)
                    emb = model(x)

                logits = -_pairwise_sq_euclidean(emb, prototypes)
                probs = torch.softmax(logits, dim=1)
                pred_pos = probs.argmax(dim=1).detach().cpu().numpy().tolist()
                preds.extend([int(class_ids[pos]) for pos in pred_pos])
                probs_out.extend(probs.detach().cpu().numpy().tolist())
                logits_out.extend(logits.detach().cpu().numpy().tolist())

        return {
            'labels': labels,
            'preds': preds,
            'probs': probs_out,
            'logits': logits_out,
            'class_ids': class_ids,
        }
