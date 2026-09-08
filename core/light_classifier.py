"""
light_classifier.py - 轻量化分类器

模块职责
--------
在 PDClassifier 基础上，专为轻量化骨干网络定制的分类器:
- 自动适配 MobileViT / VanillaNet / EfficientNet-B0 的输入尺寸
- 支持序数回归损失 (Stage 任务 ordinal 模式)
- 支持 Label Smoothing
- Dropout 提升至 0.3~0.5 范围
- AdamW 优化器 + warmup_cosine 调度器

主要类
------
- LightClassifierConfig: 轻量化分类器配置
- LightClassifier: 轻量化分类器

使用示例
--------
>>> config = LightClassifierConfig()
>>> config.BACKBONE = 'mobilevit_xxs'
>>> config.STAGE_LOSS_TYPE = 'ordinal'
>>> classifier = LightClassifier(config)
>>> result = classifier.train(train_dataset, val_dataset, task='stage')
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from typing import Literal, Optional, Dict, Any

from core.light_models import (
    LightSingleTaskModel, LightDualStreamModel,
    get_image_size_light, OrdinalRegressionLoss,
    LIGHT_BACKBONE_CONFIG
)
from core.light_trainer import (
    LightTrainerConfig,
    EarlyStopping,
    create_light_criterion,
    create_light_optimizer,
    create_light_scheduler,
    light_train_fold,
    light_evaluate,
)


# ==================== 任务配置 ====================

TASK_CONFIG = {
    'pd':      {'name': 'PD_Classification',      'num_classes': 2, 'class_names': ['Non-PD', 'PD'],                   'label_key': 'pd'},
    'stage':   {'name': 'Stage_Classification',   'num_classes': 3, 'class_names': ['Stage 1', 'Stage 2', 'Stage 3'],  'label_key': 'stage'},
    'subtype': {'name': 'Subtype_Classification', 'num_classes': 3, 'class_names': ['TD', 'PIGD', 'Mixed'],            'label_key': 'subtype'},
    'pd_subtype_tdrest':     {'name': 'Subtype_TDvsRest',    'num_classes': 2, 'class_names': ['TD', 'rest'],   'label_key': 'pd_subtype_tdrest'},
    'pd_subtype_pigd_mixed': {'name': 'Subtype_PIGDvsMixed', 'num_classes': 2, 'class_names': ['PIGD', 'mixed'], 'label_key': 'pd_subtype_pigd_mixed'},
}


# ==================== 配置类 ====================

class LightClassifierConfig:
    """
    轻量化分类器配置类

    架构配置
    --------
    ARCHITECTURE: 网络架构
        - 'single_stream': 单张图像输入（central 或 whorl）
        - 'mixed_single_stream': 混合所有区域单流训练
        - 'dual_stream': 双张图像输入（central + whorl）
    BACKBONE: 轻量化骨干网络
        - 'mobilevit_xxs': 最轻量 (~1.3M)
        - 'mobilevit_xs': 轻量 (~2.3M)
        - 'vanillanet_5': 中等 (~22M，激活退化策略)
        - 'efficientnet_b0': 小而强 (~5.3M，推荐首选)
        - 'mobilevit_xxs': 最轻量 (~1.3M)
        - 'mobilevit_xs': 轻量 (~2.3M)
        - 'vanillanet_5': 中等 (~22M，act_learn 激活，原生实现，特征维 4096)
        - 'efficientnet_b0': 小而强 (~5.3M，推荐首选)
    PRETRAINED: 预训练模式 (True=完整, False=随机, 'Half'=半冻结)

    正则化配置
    ----------
    DROPOUT: 特征层 Dropout 概率 (建议 0.3~0.5)

    损失函数配置
    ------------
    PD_LOSS_TYPE: PD 二分类任务损失类型 ('ce' / 'label_smooth')
    STAGE_LOSS_TYPE: Stage 有序分类损失 ('ce' / 'label_smooth' / 'ordinal')
    SUBTYPE_LOSS_TYPE: Subtype 分类损失 ('ce' / 'label_smooth')
    LABEL_SMOOTH_EPS: Label Smoothing epsilon

    训练配置
    --------
    BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY
    LR_SCHEDULER: 'step' / 'cosine' / 'warmup_cosine' (推荐 ViT 使用 warmup_cosine)
    BACKBONE_LR_RATIO: backbone 学习率倍率
    EARLY_STOPPING, PATIENCE: 早停配置

    数据增强
    --------
    DATA_AUGMENTATION: 是否启用数据增强
    AUG_*: 各增强参数
    """
    # ========== 架构 ==========
    ARCHITECTURE: Literal['single_stream', 'mixed_single_stream', 'dual_stream'] = 'single_stream'
    BACKBONE: Literal['mobilevit_xxs', 'mobilevit_xs', 'vanillanet_5', 'efficientnet_b0'] = 'efficientnet_b0'
    PRETRAINED: Literal[True, False, 'Half'] = True

    # ========== 正则化 ==========
    DROPOUT: float = 0.4

    # ========== 损失函数 ==========
    PD_LOSS_TYPE: str = 'ce'             # 'ce' / 'label_smooth'
    STAGE_LOSS_TYPE: str = 'label_smooth'  # 'ce' / 'label_smooth' / 'ordinal'
    SUBTYPE_LOSS_TYPE: str = 'ce'         # 'ce' / 'label_smooth'
    LABEL_SMOOTH_EPS: float = 0.1

    # ========== 数据 ==========
    DATASET_MODE: Literal['pd_control', 'pd_control_cataract', 'pd_all'] = 'pd_all'
    DATASET_JSON: str = 'datasets/pd_dataset_en/patients_dataset_en.json'

    # ========== 训练 ==========
    BATCH_SIZE: int = 16
    NUM_EPOCHS: int = 60
    LEARNING_RATE: float = 5e-4
    WEIGHT_DECAY: float = 1e-3       # 比 basic 更强的权重衰减
    NUM_WORKERS: int = 4
    DEVICE: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ========== 学习率调度 ==========
    LR_SCHEDULER: str = 'warmup_cosine'  # 'step' / 'cosine' / 'warmup_cosine'
    LR_STEP_SIZE: int = 15
    LR_GAMMA: float = 0.5
    BACKBONE_LR_RATIO: float = 0.1
    OPTIMIZER: str = 'adamw'             # 'adam' / 'adamw'

    # ========== 早停 ==========
    EARLY_STOPPING: bool = True
    PATIENCE: int = 15

    # ========== VanillaNet 训练调度 ==========
    # act_num 初始值，训练期间逐步递减到 0（逐步线性化激活函数）
    # 不使用 VanillaNet 时此配置项被忽略
    VANILLANET_INIT_ACT_NUM: int = 3

    # ========== 数据增强 ==========
    DATA_AUGMENTATION: bool = True
    AUG_HORIZONTAL_FLIP: float = 0.5
    AUG_ROTATION: float = 15.0
    AUG_BRIGHTNESS: float = 0.3
    AUG_CONTRAST: float = 0.3
    AUG_SATURATION: float = 0.1
    AUG_HUE: float = 0.0

    # ========== 交叉验证 ==========
    NUM_FOLDS: int = 5
    SEED: int = 42
    KFOLD_MODE: Literal['patient', 'image'] = 'patient'

    # ========== 输出 ==========
    OUTPUT_DIR: str = 'results'


# ==================== 分类器类 ====================

class LightClassifier:
    """
    轻量化帕金森病图像分类器

    功能概述
    --------
    1. 训练: 给定训练/验证数据集，训练模型并返回结果
    2. 模型管理: 保存/加载
    3. 序数回归: Stage 任务支持 ordinal loss，预测时自动转换

    接口与 PDClassifier 一致，可直接替换。

    属性
    ----
    config : LightClassifierConfig
    models : Dict[str, Any]
    train_transform : transforms.Compose
    val_transform : transforms.Compose
    """

    def __init__(self, config: LightClassifierConfig):
        self.config = config
        self.models: Dict[str, Any] = {}
        self._init_transforms()

    def _init_transforms(self):
        """
        初始化图像预处理

        根据 backbone 和预训练模式决定 resize/crop 尺寸，
        并构建带数据增强的训练变换和纯评估变换。
        """
        resize, crop, use_original = get_image_size_light(
            self.config.BACKBONE, self.config.PRETRAINED
        )
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        base = [] if use_original else [
            transforms.Resize((resize, resize)),
            transforms.CenterCrop(crop)
        ]

        augment = []
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
                augment.append(transforms.ColorJitter(
                    brightness=self.config.AUG_BRIGHTNESS,
                    contrast=self.config.AUG_CONTRAST,
                    saturation=self.config.AUG_SATURATION,
                    hue=self.config.AUG_HUE,
                ))
            # 随机仿射：针对视网膜图像的轻微几何扰动
            augment.append(transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)))

        self.val_transform = transforms.Compose(base + [transforms.ToTensor(), normalize])
        self.train_transform = transforms.Compose(base + augment + [transforms.ToTensor(), normalize])

    def _get_loss_type(self, task: str) -> str:
        """根据任务获取损失类型"""
        mapping = {
            'pd': self.config.PD_LOSS_TYPE,
            'stage': self.config.STAGE_LOSS_TYPE,
            'subtype': self.config.SUBTYPE_LOSS_TYPE,
            'pd_subtype_tdrest': self.config.SUBTYPE_LOSS_TYPE,
            'pd_subtype_pigd_mixed': self.config.SUBTYPE_LOSS_TYPE,
        }
        return mapping.get(task, 'ce')

    def _create_model(self, task: str) -> nn.Module:
        """
        创建神经网络模型

        Stage 任务在 ordinal 模式下输出维度为 num_classes-1。

        Args:
            task: 任务名称

        Returns:
            已移至 device 的模型
        """
        cfg = self.config
        task_cfg = TASK_CONFIG[task]
        num_classes = task_cfg['num_classes']
        loss_type = self._get_loss_type(task)
        use_ordinal = (loss_type == 'ordinal' and task == 'stage')

        if cfg.ARCHITECTURE == 'dual_stream':
            model = LightDualStreamModel(
                cfg.BACKBONE, num_classes,
                pretrained=cfg.PRETRAINED,
                dropout=cfg.DROPOUT,
                use_ordinal=use_ordinal,
            )
        else:
            model = LightSingleTaskModel(
                cfg.BACKBONE, num_classes,
                pretrained=cfg.PRETRAINED,
                dropout=cfg.DROPOUT,
                use_ordinal=use_ordinal,
            )

        return model.to(cfg.DEVICE)

    # ==================== 训练 ====================

    def train(self, train_dataset, val_dataset, task: str,
              fold_idx: int = 1, suffix: str = "",
              checkpoint_path: Optional[str] = None,
              reuse_checkpoint: bool = False) -> Dict[str, Any]:
        """
        训练模型 (单 fold)

        接口与 PDClassifier.train 完全一致，可直接替换。

        Args:
            train_dataset: 训练数据集
            val_dataset: 验证数据集
            task: 任务类型 ('pd'/'stage'/'subtype')
            fold_idx: fold 索引 (日志用)
            suffix: 任务名称后缀 (如 '_central')

        Returns:
            {
                'model': 模型实例,
                'model_state': 参数字典,
                'val_result': 验证评估结果,
                'train_history': 训练历史
            }
        """
        cfg = self.config
        task_cfg = TASK_CONFIG[task]
        loss_type = self._get_loss_type(task)

        # 数据加载器
        train_loader = DataLoader(
            train_dataset, batch_size=cfg.BATCH_SIZE,
            shuffle=True, num_workers=cfg.NUM_WORKERS,
            pin_memory=(cfg.DEVICE != 'cpu'),
            drop_last=True,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=cfg.BATCH_SIZE,
            shuffle=False, num_workers=cfg.NUM_WORKERS,
            pin_memory=(cfg.DEVICE != 'cpu'),
        )

        # 模型、损失、优化器、调度器
        model = self._create_model(task)
        criterion = create_light_criterion(
            task, task_cfg['num_classes'], loss_type,
            cfg.LABEL_SMOOTH_EPS, cfg.DEVICE
        )
        if checkpoint_path and reuse_checkpoint and os.path.exists(checkpoint_path):
            model.load_state_dict(torch.load(checkpoint_path, map_location=cfg.DEVICE))
            model.to(cfg.DEVICE)
            _, val_result = light_evaluate(
                model, val_loader, criterion, cfg.DEVICE, task_cfg['label_key']
            )
            print(f"  Fold {fold_idx} | 复用已有模型，跳过训练: {checkpoint_path}")
            return {
                'model': model,
                'model_state': {k: v.clone() for k, v in model.state_dict().items()},
                'val_result': val_result,
                'train_history': {'train_acc': [0.0], 'reused_checkpoint': checkpoint_path},
            }

        optimizer = create_light_optimizer(
            model, cfg.LEARNING_RATE, cfg.WEIGHT_DECAY,
            cfg.BACKBONE_LR_RATIO, cfg.OPTIMIZER
        )
        scheduler = create_light_scheduler(
            optimizer, cfg.LR_SCHEDULER, cfg.NUM_EPOCHS,
            cfg.LR_STEP_SIZE, cfg.LR_GAMMA
        )
        early_stopping = EarlyStopping(patience=cfg.PATIENCE) if cfg.EARLY_STOPPING else None

        # 执行训练
        history = light_train_fold(
            train_loader, val_loader, model, criterion, optimizer,
            cfg.NUM_EPOCHS, cfg.DEVICE,
            task_cfg['name'] + suffix, fold_idx, task_cfg['label_key'],
            scheduler=scheduler, early_stopping=early_stopping,
            vanillanet_act_num_init=cfg.VANILLANET_INIT_ACT_NUM,
        )

        # 最终验证评估
        _, val_result = light_evaluate(
            model, val_loader, criterion, cfg.DEVICE, task_cfg['label_key']
        )

        return {
            'model': model,
            'model_state': {k: v.clone() for k, v in model.state_dict().items()},
            'val_result': val_result,
            'train_history': history,
        }
