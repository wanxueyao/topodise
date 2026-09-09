"""ResNet18 SimCLR pretraining and downstream patient-level classification."""
import os
import json
import time
import hashlib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from typing import Literal, Optional, Dict, Any

from core.basic_models import (
    SingleTaskModel,
    DualStreamSingleTaskModel,
    get_image_size as get_image_size_basic,
    BACKBONE_CONFIG,
)
from core.classification_trainer import (
    EarlyStopping,
    create_classification_criterion,
    create_classification_optimizer,
    create_classification_scheduler,
    classification_train_fold,
    classification_evaluate,
)
from core.simclr_trainer import SimCLRModel, simclr_pretrain




class _ImagePathPretrainDataset:
    """Minimal dataset descriptor consumed by SimCLRDatasetWrapper."""

    def __init__(self, image_paths):
        self.samples = image_paths
        self.mode = 'single_stream'

    def __len__(self):
        return len(self.samples)


# ==================== 任务配置 ====================

TASK_CONFIG = {
    'pd':      {'name': 'PD_Classification',      'num_classes': 2,
                'class_names': ['Non-PD', 'PD'],   'label_key': 'pd'},
    'stage':   {'name': 'Stage_Classification',   'num_classes': 3,
                'class_names': ['Stage 1', 'Stage 2', 'Stage 3'], 'label_key': 'stage'},
    'subtype': {'name': 'Subtype_Classification', 'num_classes': 3,
                'class_names': ['TD', 'PIGD', 'Mixed'],           'label_key': 'subtype'},
    'pd_subtype_tdrest': {'name': 'Subtype_TDvsRest', 'num_classes': 2,
                          'class_names': ['TD', 'Rest'], 'label_key': 'pd_subtype_tdrest'},
    'pd_subtype_pigd_mixed': {'name': 'Subtype_PIGDvsMixed', 'num_classes': 2,
                              'class_names': ['PIGD', 'Mixed'], 'label_key': 'pd_subtype_pigd_mixed'},
    'pd_subtype_hierarchy': {'name': 'Subtype_Hierarchy', 'num_classes': 3,
                             'class_names': ['TD', 'PIGD', 'Mixed'], 'label_key': 'subtype'},
}


# ==================== 配置类 ====================

class SimCLRClassifierConfig:
    """ResNet18 SimCLR pretraining and classification configuration."""
    # ========== 架构 ==========
    ARCHITECTURE: Literal['single_stream', 'mixed_single_stream', 'dual_stream'] = 'single_stream'
    BACKBONE: Literal['resnet18'] = 'resnet18'
    PRETRAINED: Literal[True, False, 'Half'] = True
    DROPOUT: float = 0.4

    # ========== SimCLR 预训练 ==========
    SIMCLR_EPOCHS: int = 100
    SIMCLR_LR: float = 1e-3
    SIMCLR_BATCH_SIZE: int = 32
    SIMCLR_TEMPERATURE: float = 0.5
    SIMCLR_PROJ_DIM: int = 128
    SIMCLR_PROJ_HIDDEN_DIM: int = 256
    SIMCLR_WEIGHT_DECAY: float = 1e-4
    SIMCLR_PRETRAIN_DATASET: str = 'fold'
    CORN1_ROOT: str = 'datasets/CORN-1'
    CORN1_PRETRAIN_SPLITS: str = 'train,val,test'
    SHARED_PRETRAIN_CACHE_DIR: str = 'results/simclr_pretrain_cache'

    # ========== 损失函数 ==========
    PD_LOSS_TYPE: str = 'ce'
    STAGE_LOSS_TYPE: str = 'label_smooth'
    SUBTYPE_LOSS_TYPE: str = 'ce'
    LABEL_SMOOTH_EPS: float = 0.1

    # ========== 数据 ==========
    DATASET_MODE: Literal['pd_control', 'pd_control_cataract', 'pd_all'] = 'pd_all'
    DATASET_JSON: str = 'datasets/pd_dataset_en/patients_dataset_en.json'

    # ========== 微调训练 ==========
    BATCH_SIZE: int = 16
    NUM_EPOCHS: int = 60
    LEARNING_RATE: float = 5e-4
    WEIGHT_DECAY: float = 1e-3
    NUM_WORKERS: int = 4
    DEVICE: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ========== 学习率调度 ==========
    LR_SCHEDULER: str = 'warmup_cosine'
    LR_STEP_SIZE: int = 15
    LR_GAMMA: float = 0.5
    BACKBONE_LR_RATIO: float = 0.1
    OPTIMIZER: str = 'adamw'

    # ========== 早停 ==========
    EARLY_STOPPING: bool = True
    PATIENCE: int = 15


    # ========== 交叉验证 ==========
    NUM_FOLDS: int = 5
    SEED: int = 42
    KFOLD_MODE: Literal['patient', 'image'] = 'patient'

    # ========== 输出 ==========
    OUTPUT_DIR: str = 'results'


# ==================== 分类器类 ====================

class SimCLRClassifier:
    """
    SimCLR 增强的帕金森病图像分类器

    两阶段训练流程
    --------------
    1. SimCLR 自监督预训练：对训练集所有图像无监督预训练，
       backbone 学习视觉表征，不使用任何标签信息。
    2. 有监督微调：使用预训练 backbone 权重初始化，
       添加分类头后在下游任务上微调，不使用额外图像增强。

    接口与 LightClassifier 一致，可直接替换。

    属性
    ----
    config : SimCLRClassifierConfig
    models : Dict[str, Any]
    train_transform : 微调阶段变换（无增强）
    val_transform : 验证阶段变换
    """

    def __init__(self, config: SimCLRClassifierConfig):
        self.config = config
        self.models: Dict[str, Any] = {}
        # 迁移复用：若外部预先设置了已训练好的 SimCLRModel（例如在另一数据集上
        # 完成的自监督预训练），train() 会跳过 Phase 1 自监督预训练，直接复用该
        # backbone 表征，只做有监督微调。默认 None 时保持原有逐 fold 预训练行为。
        self.precomputed_simclr: Optional[SimCLRModel] = None
        self._shared_pretrain_simclr: Optional[SimCLRModel] = None
        self._init_transforms()

    def _init_transforms(self):
        """
        初始化微调阶段图像变换

        微调阶段训练集和验证集均不添加随机增强，
        仅做 Resize → CenterCrop → ToTensor → Normalize。
        SimCLR 预训练已通过对比增强学习了鲁棒表征。
        """
        resize, crop, use_original = get_image_size_basic(
            self.config.BACKBONE, self.config.PRETRAINED
        )
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        base = [] if use_original else [
            transforms.Resize((resize, resize)),
            transforms.CenterCrop(crop),
        ]

        # 训练和验证使用相同的纯净变换（无随机增强）
        clean_transform = transforms.Compose(base + [transforms.ToTensor(), normalize])
        self.train_transform = clean_transform
        self.val_transform = clean_transform

    def _get_loss_type(self, task: str) -> str:
        """根据任务获取损失类型"""
        mapping = {
            'pd':      self.config.PD_LOSS_TYPE,
            'stage':   self.config.STAGE_LOSS_TYPE,
            'subtype': self.config.SUBTYPE_LOSS_TYPE,
            'pd_subtype_tdrest': self.config.SUBTYPE_LOSS_TYPE,
            'pd_subtype_pigd_mixed': self.config.SUBTYPE_LOSS_TYPE,
        }
        return mapping.get(task, 'ce')

    def _get_input_size(self) -> int:
        """获取 backbone 推荐输入尺寸"""
        return BACKBONE_CONFIG[self.config.BACKBONE]['input_size']

    def _create_model_from_simclr(
        self, task: str, simclr_model: SimCLRModel
    ) -> nn.Module:
        """
        从 SimCLR 预训练模型创建下游分类模型

        将 SimCLR backbone 权重加载至分类模型中，替换随机初始化权重。
        dual_stream 模式下，central 和 whorl 两路 backbone 均使用相同的预训练权重初始化。

        Args:
            task: 下游任务名称
            simclr_model: 预训练完成的 SimCLRModel

        Returns:
            已迁移预训练权重的分类模型（位于 config.DEVICE）
        """
        cfg = self.config
        task_cfg = TASK_CONFIG[task]
        num_classes = task_cfg['num_classes']

        pretrained_backbone_state = simclr_model.get_backbone().state_dict()

        if cfg.ARCHITECTURE == 'dual_stream':
            model = DualStreamSingleTaskModel(
                cfg.BACKBONE, num_classes,
                pretrained=False,   # 不重新加载 ImageNet 权重
            )
            # 两路 backbone 均使用 SimCLR 预训练权重初始化
            model.backbone_central.load_state_dict(pretrained_backbone_state)
            model.backbone_whorl.load_state_dict(pretrained_backbone_state)
        else:
            model = SingleTaskModel(
                cfg.BACKBONE, num_classes,
                pretrained=False,
            )
            model.backbone.load_state_dict(pretrained_backbone_state)

        return model.to(cfg.DEVICE)

    # ==================== 训练 ====================

    def pretrain_backbone(self, train_dataset, fold_idx: int = 1) -> SimCLRModel:
        """只执行当前 fold 的 SimCLR 自监督预训练，供多个下游 head 复用。"""
        cfg = self.config
        if self.precomputed_simclr is not None:
            print(f"\n  [SimCLR] Fold {fold_idx} - 复用外部预训练 backbone ...")
            return self.precomputed_simclr
        if self._uses_shared_pretrain_source() and self._shared_pretrain_simclr is not None:
            print(f"\n  [SimCLR] Fold {fold_idx} - 复用 {cfg.SIMCLR_PRETRAIN_DATASET} 预训练 backbone ...")
            return self._shared_pretrain_simclr
        if self._uses_shared_pretrain_source():
            return self._get_or_train_shared_pretrain(train_dataset, fold_idx)

        pretrain_dataset = self._resolve_pretrain_dataset(train_dataset)

        print(f"\n  [SimCLR] Fold {fold_idx} - Shared self-supervised pretraining "
              f"(backbone={cfg.BACKBONE}, source={cfg.SIMCLR_PRETRAIN_DATASET}, "
              f"epochs={cfg.SIMCLR_EPOCHS}) ...")
        simclr_model = simclr_pretrain(
            backbone_name=cfg.BACKBONE,
            pretrained=cfg.PRETRAINED,
            train_dataset=pretrain_dataset,
            num_epochs=cfg.SIMCLR_EPOCHS,
            batch_size=cfg.SIMCLR_BATCH_SIZE,
            lr=cfg.SIMCLR_LR,
            temperature=cfg.SIMCLR_TEMPERATURE,
            device=cfg.DEVICE,
            proj_hidden_dim=cfg.SIMCLR_PROJ_HIDDEN_DIM,
            proj_out_dim=cfg.SIMCLR_PROJ_DIM,
            weight_decay=cfg.SIMCLR_WEIGHT_DECAY,
            num_workers=cfg.NUM_WORKERS,
            input_size=self._get_input_size(),
            verbose=True,
        )
        if self._uses_shared_pretrain_source():
            self._shared_pretrain_simclr = simclr_model
        return simclr_model

    def _uses_shared_pretrain_source(self) -> bool:
        source = str(getattr(self.config, 'SIMCLR_PRETRAIN_DATASET', 'fold')).lower()
        return source not in ('fold', 'pd_fold')

    def _get_or_train_shared_pretrain(self, train_dataset, fold_idx: int) -> SimCLRModel:
        cache_path = self._shared_pretrain_cache_path()
        cached = self._load_shared_pretrain_cache(cache_path)
        if cached is not None:
            print(f"\n  [SimCLR] Fold {fold_idx} - 复用磁盘缓存的 "
                  f"{self.config.SIMCLR_PRETRAIN_DATASET} backbone: {cache_path}")
            self._shared_pretrain_simclr = cached
            return cached

        lock_path = f"{cache_path}.lock"
        lock_fd = None
        while lock_fd is None:
            try:
                lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(lock_fd, str(os.getpid()).encode('utf-8'))
            except FileExistsError:
                cached = self._load_shared_pretrain_cache(cache_path)
                if cached is not None:
                    print(f"\n  [SimCLR] Fold {fold_idx} - 等待后复用磁盘缓存的 "
                          f"{self.config.SIMCLR_PRETRAIN_DATASET} backbone: {cache_path}")
                    self._shared_pretrain_simclr = cached
                    return cached
                if self._is_stale_lock(lock_path):
                    try:
                        os.remove(lock_path)
                    except OSError:
                        pass
                    continue
                print(f"\n  [SimCLR] Fold {fold_idx} - 等待其它进程生成 "
                      f"{self.config.SIMCLR_PRETRAIN_DATASET} 预训练缓存 ...")
                time.sleep(30)

        try:
            cached = self._load_shared_pretrain_cache(cache_path)
            if cached is not None:
                self._shared_pretrain_simclr = cached
                return cached

            pretrain_dataset = self._resolve_pretrain_dataset(train_dataset)
            print(f"\n  [SimCLR] Fold {fold_idx} - Shared self-supervised pretraining "
                  f"(backbone={self.config.BACKBONE}, source={self.config.SIMCLR_PRETRAIN_DATASET}, "
                  f"epochs={self.config.SIMCLR_EPOCHS}) ...")
            simclr_model = self._run_simclr_pretrain(pretrain_dataset)
            self._save_shared_pretrain_cache(cache_path, simclr_model)
            self._shared_pretrain_simclr = simclr_model
            return simclr_model
        finally:
            if lock_fd is not None:
                os.close(lock_fd)
            try:
                os.remove(lock_path)
            except OSError:
                pass

    def _run_simclr_pretrain(self, pretrain_dataset) -> SimCLRModel:
        cfg = self.config
        return simclr_pretrain(
            backbone_name=cfg.BACKBONE,
            pretrained=cfg.PRETRAINED,
            train_dataset=pretrain_dataset,
            num_epochs=cfg.SIMCLR_EPOCHS,
            batch_size=cfg.SIMCLR_BATCH_SIZE,
            lr=cfg.SIMCLR_LR,
            temperature=cfg.SIMCLR_TEMPERATURE,
            device=cfg.DEVICE,
            proj_hidden_dim=cfg.SIMCLR_PROJ_HIDDEN_DIM,
            proj_out_dim=cfg.SIMCLR_PROJ_DIM,
            weight_decay=cfg.SIMCLR_WEIGHT_DECAY,
            num_workers=cfg.NUM_WORKERS,
            input_size=self._get_input_size(),
            verbose=True,
        )

    def _shared_pretrain_cache_path(self) -> str:
        cfg = self.config
        cache_dir = self._resolve_existing_path(
            getattr(cfg, 'SHARED_PRETRAIN_CACHE_DIR', 'results/simclr_pretrain_cache')
        )
        os.makedirs(cache_dir, exist_ok=True)
        key = {
            'source': str(getattr(cfg, 'SIMCLR_PRETRAIN_DATASET', 'fold')).lower(),
            'backbone': cfg.BACKBONE,
            'pretrained': str(cfg.PRETRAINED),
            'simclr_epochs': cfg.SIMCLR_EPOCHS,
            'simclr_lr': cfg.SIMCLR_LR,
            'simclr_batch': cfg.SIMCLR_BATCH_SIZE,
            'temperature': cfg.SIMCLR_TEMPERATURE,
            'proj_dim': cfg.SIMCLR_PROJ_DIM,
            'proj_hidden_dim': cfg.SIMCLR_PROJ_HIDDEN_DIM,
            'simclr_weight_decay': cfg.SIMCLR_WEIGHT_DECAY,
            'input_size': self._get_input_size(),
            'corn1_root': os.path.abspath(self._resolve_existing_path(getattr(cfg, 'CORN1_ROOT', 'datasets/CORN-1'))),
            'corn1_splits': getattr(cfg, 'CORN1_PRETRAIN_SPLITS', 'train,val,test'),
        }
        raw = json.dumps(key, sort_keys=True, ensure_ascii=False)
        sig = hashlib.sha1(raw.encode('utf-8')).hexdigest()[:12]
        safe_source = key['source'].replace('-', '').replace('/', '_')
        return os.path.join(cache_dir, f"{safe_source}_{cfg.BACKBONE}_{sig}.pt")

    def _load_shared_pretrain_cache(self, cache_path: str) -> Optional[SimCLRModel]:
        if not os.path.exists(cache_path) or os.path.getsize(cache_path) <= 0:
            return None
        cfg = self.config
        state = torch.load(cache_path, map_location=cfg.DEVICE)
        backbone_state = state.get('backbone_state', state) if isinstance(state, dict) else state
        model = SimCLRModel(
            cfg.BACKBONE,
            pretrained=False,
            proj_hidden_dim=cfg.SIMCLR_PROJ_HIDDEN_DIM,
            proj_out_dim=cfg.SIMCLR_PROJ_DIM,
        )
        model.get_backbone().load_state_dict(backbone_state)
        return model.to(cfg.DEVICE)

    def _save_shared_pretrain_cache(self, cache_path: str, simclr_model: SimCLRModel):
        payload = {
            'backbone_state': {
                k: v.detach().cpu()
                for k, v in simclr_model.get_backbone().state_dict().items()
            },
            'meta': {
                'source': getattr(self.config, 'SIMCLR_PRETRAIN_DATASET', 'fold'),
                'backbone': self.config.BACKBONE,
                'simclr_epochs': self.config.SIMCLR_EPOCHS,
                'temperature': self.config.SIMCLR_TEMPERATURE,
                'proj_dim': self.config.SIMCLR_PROJ_DIM,
                'corn1_root': getattr(self.config, 'CORN1_ROOT', ''),
                'corn1_splits': getattr(self.config, 'CORN1_PRETRAIN_SPLITS', ''),
                'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            },
        }
        tmp_path = f"{cache_path}.tmp.{os.getpid()}"
        torch.save(payload, tmp_path)
        os.replace(tmp_path, cache_path)
        print(f"  [SimCLR] 已缓存 {self.config.SIMCLR_PRETRAIN_DATASET} backbone: {cache_path}")

    @staticmethod
    def _is_stale_lock(lock_path: str, stale_seconds: int = 12 * 60 * 60) -> bool:
        try:
            return (time.time() - os.path.getmtime(lock_path)) > stale_seconds
        except OSError:
            return False

    def _resolve_pretrain_dataset(self, train_dataset):
        source = str(getattr(self.config, 'SIMCLR_PRETRAIN_DATASET', 'fold')).lower()
        if source in ('fold', 'pd_fold'):
            return train_dataset
        if source in ('corn1', 'corn-1'):
            image_paths = self._collect_corn1_image_paths()
            print(f"  [SimCLR] 使用 CORN-1 图像做自监督预训练: {len(image_paths)} images")
            return _ImagePathPretrainDataset(image_paths)
        raise ValueError(f"未知 SIMCLR_PRETRAIN_DATASET: {self.config.SIMCLR_PRETRAIN_DATASET}")

    def _collect_corn1_image_paths(self):
        root = self._resolve_existing_path(getattr(self.config, 'CORN1_ROOT', 'datasets/CORN-1'))
        raw_splits = getattr(self.config, 'CORN1_PRETRAIN_SPLITS', 'train,val,test')
        splits = [s.strip() for s in str(raw_splits).split(',') if s.strip()]
        exts = ('.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp')
        image_paths = []
        for split in splits:
            image_dir = os.path.join(root, 'images', f'{split}_orig')
            if not os.path.isdir(image_dir):
                raise FileNotFoundError(f"CORN-1 split 图像目录不存在: {image_dir}")
            image_paths.extend(
                os.path.join(image_dir, name)
                for name in sorted(os.listdir(image_dir))
                if name.lower().endswith(exts)
            )
        if not image_paths:
            raise RuntimeError(f"CORN-1 预训练图像为空: {root}, splits={splits}")
        return image_paths

    @staticmethod
    def _resolve_existing_path(path: str) -> str:
        if os.path.isabs(path):
            return path
        candidates = [
            os.path.abspath(path),
            os.path.join(os.getcwd(), path),
            os.path.join(os.path.dirname(os.getcwd()), path),
        ]
        for cand in candidates:
            if os.path.exists(cand):
                return cand
        return candidates[0]

    def train(self, train_dataset, val_dataset, task: str,
              fold_idx: int = 1, suffix: str = "") -> Dict[str, Any]:
        """
        两阶段训练（SimCLR 预训练 + 有监督微调）

        Args:
            train_dataset: 训练数据集（PDDataset 实例，transform 已设置）
            val_dataset: 验证数据集（PDDataset 实例）
            task: 任务类型 ('pd' / 'stage' / 'subtype')
            fold_idx: fold 索引（日志显示用）
            suffix: 任务名称后缀（如 '_central'）

        Returns:
            Dict 包含:
                'model': 微调后的模型实例
                'model_state': 参数字典
                'val_result': 最终验证评估结果
                'train_history': 微调阶段训练历史
        """
        cfg = self.config
        task_cfg = TASK_CONFIG[task]
        loss_type = self._get_loss_type(task)

        # ---- 阶段 1: SimCLR 自监督预训练 ----
        if self.precomputed_simclr is not None:
            # 迁移复用：直接使用外部传入的、已在其它数据集上完成自监督预训练的
            # backbone，跳过本数据集的逐 fold 预训练。
            print(f"\n  [SimCLR] Fold {fold_idx} - Phase 1: 复用外部预训练 backbone "
                  f"(backbone={cfg.BACKBONE}，跳过自监督预训练) ...")
            simclr_model = self.precomputed_simclr
        elif self._uses_shared_pretrain_source() and self._shared_pretrain_simclr is not None:
            print(f"\n  [SimCLR] Fold {fold_idx} - Phase 1: 复用 {cfg.SIMCLR_PRETRAIN_DATASET} "
                  f"预训练 backbone (backbone={cfg.BACKBONE}) ...")
            simclr_model = self._shared_pretrain_simclr
        elif self._uses_shared_pretrain_source():
            simclr_model = self._get_or_train_shared_pretrain(train_dataset, fold_idx)
        else:
            print(f"\n  [SimCLR] Fold {fold_idx} - Phase 1: Self-supervised pretraining "
                  f"(backbone={cfg.BACKBONE}, source={cfg.SIMCLR_PRETRAIN_DATASET}, "
                  f"epochs={cfg.SIMCLR_EPOCHS}) ...")

            pretrain_dataset = self._resolve_pretrain_dataset(train_dataset)

            simclr_model = self._run_simclr_pretrain(pretrain_dataset)
            if self._uses_shared_pretrain_source():
                self._shared_pretrain_simclr = simclr_model

        # ---- 阶段 2: 有监督微调（无增强）----
        print(f"\n  [SimCLR] Fold {fold_idx} - Phase 2: Supervised fine-tuning "
              f"(task={task}, epochs={cfg.NUM_EPOCHS}) ...")

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

        # 从 SimCLR 预训练权重初始化分类模型
        model = self._create_model_from_simclr(task, simclr_model)

        criterion = create_classification_criterion(
            task, task_cfg['num_classes'], loss_type,
            cfg.LABEL_SMOOTH_EPS, cfg.DEVICE,
        )
        optimizer = create_classification_optimizer(
            model, cfg.LEARNING_RATE, cfg.WEIGHT_DECAY,
            cfg.BACKBONE_LR_RATIO, cfg.OPTIMIZER,
        )
        scheduler = create_classification_scheduler(
            optimizer, cfg.LR_SCHEDULER, cfg.NUM_EPOCHS,
            cfg.LR_STEP_SIZE, cfg.LR_GAMMA,
        )
        early_stopping = EarlyStopping(patience=cfg.PATIENCE) if cfg.EARLY_STOPPING else None

        history = classification_train_fold(
            train_loader, val_loader, model, criterion, optimizer,
            cfg.NUM_EPOCHS, cfg.DEVICE,
            task_cfg['name'] + suffix, fold_idx, task_cfg['label_key'],
            scheduler=scheduler, early_stopping=early_stopping,
        )

        _, val_result = classification_evaluate(
            model, val_loader, criterion, cfg.DEVICE, task_cfg['label_key'],
        )

        return {
            'model': model,
            'model_state': {k: v.clone() for k, v in model.state_dict().items()},
            'simclr_model': simclr_model,
            'val_result': val_result,
            'train_history': history,
        }

    def predict_dataset(self, model: nn.Module, dataset, task: str) -> Dict[str, Any]:
        """对 dataset 全量推理；二分类 subtype 任务不会用真实标签过滤 TD。"""
        cfg = self.config
        task_cfg = TASK_CONFIG[task]
        loader = DataLoader(
            dataset,
            batch_size=cfg.BATCH_SIZE,
            shuffle=False,
            num_workers=cfg.NUM_WORKERS,
            pin_memory=(cfg.DEVICE != 'cpu'),
        )

        model.eval()
        all_preds, all_labels, all_probs = [], [], []
        with torch.no_grad():
            for batch in loader:
                is_dual = isinstance(batch[0], (tuple, list))
                if is_dual:
                    (central, whorl), _, _, subtype = batch
                    images = (central.to(cfg.DEVICE), whorl.to(cfg.DEVICE))
                    outputs = model(*images)
                else:
                    img, _, _, subtype = batch
                    outputs = model(img.to(cfg.DEVICE))

                if task == 'pd_subtype_tdrest':
                    labels = torch.where(subtype == 0, 0, torch.where(subtype > 0, 1, -1))
                elif task == 'pd_subtype_pigd_mixed':
                    labels = torch.where(subtype == 1, 0, torch.where(subtype == 2, 1, -1))
                else:
                    labels = subtype

                probs = torch.softmax(outputs, dim=1)
                preds = probs.argmax(dim=1)
                all_labels.extend(labels.cpu().numpy().tolist())
                all_preds.extend(preds.cpu().numpy().tolist())
                all_probs.extend(probs.cpu().numpy().tolist())

        return {
            'labels': all_labels,
            'preds': all_preds,
            'probs': all_probs,
            'class_ids': list(range(task_cfg['num_classes'])),
        }
