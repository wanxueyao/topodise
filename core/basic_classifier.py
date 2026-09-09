"""
basic_classifier.py - 帕金森病图像分类器

模块职责
--------
提供模型训练和分类预测功能，不包含:
- K-fold 交叉验证（由实验层负责）
- 指标汇总和可视化（由实验层负责）

主要类
------
- PDClassifierConfig: 分类器配置类
- PDClassifier: 分类器主类

使用示例
--------
>>> config = PDClassifierConfig()
>>> classifier = PDClassifier(config)
>>> result = classifier.train(train_dataset, val_dataset, task='pd')
>>> prediction = classifier.predict('image.jpg', task='pd')
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from typing import Literal, Optional, Dict, Any

from core.basic_models import SingleTaskModel, DualStreamSingleTaskModel, get_image_size
from core.basic_trainer import (
    train_fold_single_task, evaluate, 
    create_optimizer, create_scheduler, EarlyStopping
)


# ==================== 任务配置 ====================

TASK_CONFIG = {
    'pd':      {'name': 'PD_Classification',      'num_classes': 2, 'class_names': ['Non-PD', 'PD'],              'label_key': 'pd'},
    'stage':   {'name': 'Stage_Classification',   'num_classes': 3, 'class_names': ['Stage 1', 'Stage 2', 'Stage 3'], 'label_key': 'stage'},
    'subtype': {'name': 'Subtype_Classification', 'num_classes': 3, 'class_names': ['TD', 'PIGD', 'Mixed'],       'label_key': 'subtype'},
    # subtype 层级三分类的两个二分类子任务（标签写入 subtype_labels，复用 mask 机制）
    'pd_subtype_tdrest':     {'name': 'Subtype_TDvsRest',    'num_classes': 2, 'class_names': ['TD', 'rest'],   'label_key': 'pd_subtype_tdrest'},
    'pd_subtype_pigd_mixed': {'name': 'Subtype_PIGDvsMixed', 'num_classes': 2, 'class_names': ['PIGD', 'mixed'], 'label_key': 'pd_subtype_pigd_mixed'},
}
"""
任务配置字典

Key:
    - 'pd': 帕金森病二分类 (有PD/无PD)
    - 'stage': 病程阶段三分类 (Stage 1/2/3)
    - 'subtype': 亚型三分类 (TD/PIGD/mixed)
    
Value:
    - name: 任务显示名称
    - num_classes: 类别数量
    - class_names: 类别名称列表
    - label_key: 数据集中对应的标签键名
"""


# ==================== 配置类 ====================

class PDClassifierConfig:
    """
    分类器配置类
    
    属性分组
    --------
    架构配置:
        ARCHITECTURE: 网络架构 ('single_stream'/'mixed_single_stream'/'dual_stream')
        BACKBONE: 特征提取网络 ('resnet18')
        PRETRAINED: 预训练模式 (True=完整预训练, False=随机初始化, 'Half'=半冻结)
    
    训练配置:
        BATCH_SIZE: 批次大小
        NUM_EPOCHS: 训练轮数
        LEARNING_RATE: 基础学习率
        WEIGHT_DECAY: L2 正则化系数
    
    学习率调度:
        LR_SCHEDULER: 调度器类型 ('step'/'cosine'/'none')
        LR_STEP_SIZE: StepLR 步长
        LR_GAMMA: StepLR 衰减因子
        BACKBONE_LR_RATIO: backbone 学习率倍率 (相对于 LEARNING_RATE)
    
    早停配置:
        EARLY_STOPPING: 是否启用早停
        PATIENCE: 早停耐心值 (无改善的 epoch 数)
    
    数据增强配置:
        DATA_AUGMENTATION: 是否启用数据增强
        AUG_HORIZONTAL_FLIP: 水平翻转概率
        AUG_ROTATION: 旋转角度范围 (度)
        AUG_BRIGHTNESS: 亮度抖动范围
        AUG_CONTRAST: 对比度抖动范围
        AUG_SATURATION: 饱和度抖动范围
        AUG_HUE: 色调抖动范围
    """
    # 架构
    ARCHITECTURE: Literal['single_stream', 'mixed_single_stream', 'dual_stream'] = 'single_stream'
    BACKBONE: Literal['resnet18'] = 'resnet18'
    PRETRAINED: Literal[True, False, 'Half'] = True
    
    # 训练
    BATCH_SIZE: int = 16
    NUM_EPOCHS: int = 50
    LEARNING_RATE: float = 0.001
    WEIGHT_DECAY: float = 1e-4
    NUM_WORKERS: int = 4
    DEVICE: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 学习率调度
    LR_SCHEDULER: str = 'step'
    LR_STEP_SIZE: int = 15
    LR_GAMMA: float = 0.5
    BACKBONE_LR_RATIO: float = 0.1
    
    # 早停
    EARLY_STOPPING: bool = True
    PATIENCE: int = 10
    
    # 数据增强
    DATA_AUGMENTATION: bool = True              # 是否启用数据增强
    AUG_HORIZONTAL_FLIP: float = 0.5            # 水平翻转概率
    AUG_ROTATION: float = 10.0                  # 旋转角度范围 (度)
    AUG_BRIGHTNESS: float = 0.2                 # 亮度抖动范围
    AUG_CONTRAST: float = 0.2                   # 对比度抖动范围
    AUG_SATURATION: float = 0.0                 # 饱和度抖动范围
    AUG_HUE: float = 0.0                        # 色调抖动范围


# ==================== 分类器类 ====================

class PDClassifier:
    """
    帕金森病图像分类器
    
    功能概述
    --------
    1. 训练: 给定训练/验证数据集，训练模型并返回结果
    2. 预测: 给定图像路径，返回分类结果
    3. 模型管理: 加载/保存模型
    
    架构支持
    --------
    - single_stream: 单张图像输入 (central 或 whorl 区域)
    - mixed_single_stream: 混合所有区域的单流训练
    - dual_stream: 双张图像输入 (central + whorl)
    
    属性
    ----
    config : PDClassifierConfig
        分类器配置
    models : Dict[str, Any]
        已加载的模型字典 {task: model} 或 {task: {region: model}}
    train_transform : transforms.Compose
        训练时的图像变换 (含数据增强)
    val_transform : transforms.Compose
        验证/预测时的图像变换 (无数据增强)
    """
    
    def __init__(self, config: PDClassifierConfig):
        """
        初始化分类器
        
        Args:
            config: 分类器配置对象
        """
        self.config = config
        self.models: Dict[str, Any] = {}
        self._init_transforms()
    
    def _init_transforms(self):
        """
        初始化图像预处理流程
        
        根据 backbone 类型和预训练模式确定:
        - 是否需要 resize/crop
        - 数据增强策略
        - ImageNet 标准化参数
        """
        resize, crop, use_original = get_image_size(self.config.BACKBONE, self.config.PRETRAINED)
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        
        # 基础变换: resize + crop (迁移学习时使用)
        base = [] if use_original else [transforms.Resize((resize, resize)), transforms.CenterCrop(crop)]
        
        # 数据增强: 根据配置构建增强策略
        augment = []
        if self.config.DATA_AUGMENTATION:
            # 水平翻转
            if self.config.AUG_HORIZONTAL_FLIP > 0:
                augment.append(transforms.RandomHorizontalFlip(p=self.config.AUG_HORIZONTAL_FLIP))
            
            # 旋转
            if self.config.AUG_ROTATION > 0:
                augment.append(transforms.RandomRotation(self.config.AUG_ROTATION))
            
            # 颜色抖动
            if any([self.config.AUG_BRIGHTNESS > 0, self.config.AUG_CONTRAST > 0,
                    self.config.AUG_SATURATION > 0, self.config.AUG_HUE > 0]):
                augment.append(transforms.ColorJitter(
                    brightness=self.config.AUG_BRIGHTNESS,
                    contrast=self.config.AUG_CONTRAST,
                    saturation=self.config.AUG_SATURATION,
                    hue=self.config.AUG_HUE
                ))
        
        self.val_transform = transforms.Compose(base + [transforms.ToTensor(), normalize])
        self.train_transform = transforms.Compose(base + augment + [transforms.ToTensor(), normalize])
    
    def _create_model(self, num_classes: int) -> nn.Module:
        """
        创建神经网络模型
        
        Args:
            num_classes: 输出类别数
            
        Returns:
            已移至指定设备的模型实例
        """
        Model = SingleTaskModel if self.config.ARCHITECTURE != 'dual_stream' else DualStreamSingleTaskModel
        return Model(self.config.BACKBONE, num_classes, self.config.PRETRAINED).to(self.config.DEVICE)
    
    def _get_model(self, task: str, region: Optional[str] = None) -> nn.Module:
        """
        获取已注册的模型
        
        Args:
            task: 任务类型 ('pd'/'stage'/'subtype')
            region: 区域 (仅 single_stream 需要)
            
        Returns:
            模型实例
            
        Raises:
            ValueError: 模型未加载时抛出
        """
        if region and isinstance(self.models.get(task), dict):
            model = self.models[task].get(region)
        else:
            model = self.models.get(task)
        if model is None:
            raise ValueError(f"模型未加载: task={task}, region={region}")
        return model
    
    def _set_model(self, model: nn.Module, task: str, region: Optional[str] = None):
        """
        注册模型到分类器
        
        Args:
            model: 模型实例
            task: 任务类型
            region: 区域 (可选)
        """
        if region:
            self.models.setdefault(task, {})[region] = model
        else:
            self.models[task] = model
    
    # ==================== 训练 ====================
    
    def train(self, train_dataset, val_dataset, task: str, 
              fold_idx: int = 1, suffix: str = "",
              checkpoint_path: Optional[str] = None,
              reuse_checkpoint: bool = False) -> Dict[str, Any]:
        """
        训练模型
        
        Args:
            train_dataset: 训练数据集 (torch.utils.data.Dataset)
            val_dataset: 验证数据集
            task: 任务类型 ('pd'/'stage'/'subtype')
            fold_idx: fold 索引 (用于日志显示)
            suffix: 任务名称后缀 (如 '_central')
            
        Returns:
            训练结果字典:
            - model: 训练好的模型实例
            - model_state: 模型参数字典 (可用于保存)
            - val_result: 验证集评估结果
            - train_history: 训练历史 (loss/acc/f1 曲线数据)
        """
        cfg, task_cfg = self.config, TASK_CONFIG[task]
        
        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=cfg.NUM_WORKERS)
        val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=cfg.NUM_WORKERS)
        
        # 创建模型、优化器、调度器
        model = self._create_model(task_cfg['num_classes'])
        if checkpoint_path and reuse_checkpoint and os.path.exists(checkpoint_path):
            model.load_state_dict(torch.load(checkpoint_path, map_location=cfg.DEVICE))
            model.to(cfg.DEVICE)
            criterion = nn.CrossEntropyLoss()
            _, val_result = evaluate(model, val_loader, criterion, cfg.DEVICE, task_cfg['label_key'])
            print(f"  Fold {fold_idx} | 复用已有模型，跳过训练: {checkpoint_path}")
            return {
                'model': model,
                'model_state': model.state_dict().copy(),
                'val_result': val_result,
                'train_history': {'train_acc': [0.0], 'reused_checkpoint': checkpoint_path},
            }

        optimizer = create_optimizer(model, cfg.LEARNING_RATE, cfg.WEIGHT_DECAY, cfg.BACKBONE_LR_RATIO)
        scheduler = create_scheduler(optimizer, cfg.LR_SCHEDULER, cfg.NUM_EPOCHS, cfg.LR_STEP_SIZE, cfg.LR_GAMMA)
        early_stopping = EarlyStopping(patience=cfg.PATIENCE) if cfg.EARLY_STOPPING else None
        criterion = nn.CrossEntropyLoss()
        
        # 执行训练
        history = train_fold_single_task(
            train_loader, val_loader, model, criterion, optimizer,
            cfg.NUM_EPOCHS, cfg.DEVICE, task_cfg['name'] + suffix, fold_idx, task_cfg['label_key'],
            scheduler=scheduler, early_stopping=early_stopping
        )
        
        # 最终评估
        _, val_result = evaluate(model, val_loader, criterion, cfg.DEVICE, task_cfg['label_key'])
        
        return {
            'model': model,
            'model_state': model.state_dict().copy(),
            'val_result': val_result,
            'train_history': history
        }
    
    # ==================== 预测 ====================
    
    def predict(self, image_path: str, task: str, 
                region: Optional[str] = None, whorl_path: Optional[str] = None) -> Dict[str, Any]:
        """
        分类预测 (自动选择单流/双流模式)
        
        Args:
            image_path: 图像路径 (单流时为唯一图像, 双流时为 central 图像)
            task: 任务类型 ('pd'/'stage'/'subtype')
            region: 区域 (仅 single_stream 需要指定 'central'/'whorl')
            whorl_path: whorl 图像路径 (仅 dual_stream 需要)
            
        Returns:
            预测结果字典:
            - class_id: 预测类别 ID
            - class_name: 预测类别名称
            - probabilities: 各类别概率列表
            
        Raises:
            ValueError: dual_stream 模式未提供 whorl_path 时抛出
        """
        if self.config.ARCHITECTURE == 'dual_stream':
            if not whorl_path:
                raise ValueError("dual_stream 架构需要提供 whorl_path")
            return self._predict_dual(image_path, whorl_path, task)
        return self._predict_single(image_path, task, region)
    
    def _predict_single(self, image_path: str, task: str, 
                        region: Optional[str] = None) -> Dict[str, Any]:
        """单流预测 (内部方法)"""
        model = self._get_model(task, region)
        model.eval()
        
        img = self.val_transform(Image.open(image_path).convert('RGB')).unsqueeze(0).to(self.config.DEVICE)
        with torch.no_grad():
            probs = torch.softmax(model(img), dim=1)
        
        class_id = probs.argmax(1).item()
        return {
            'class_id': class_id,
            'class_name': TASK_CONFIG[task]['class_names'][class_id],
            'probabilities': probs[0].cpu().tolist()
        }
    
    def _predict_dual(self, central_path: str, whorl_path: str, task: str) -> Dict[str, Any]:
        """双流预测 (内部方法)"""
        model = self._get_model(task)
        model.eval()
        
        c = self.val_transform(Image.open(central_path).convert('RGB')).unsqueeze(0).to(self.config.DEVICE)
        w = self.val_transform(Image.open(whorl_path).convert('RGB')).unsqueeze(0).to(self.config.DEVICE)
        with torch.no_grad():
            probs = torch.softmax(model(c, w), dim=1)
        
        class_id = probs.argmax(1).item()
        return {
            'class_id': class_id,
            'class_name': TASK_CONFIG[task]['class_names'][class_id],
            'probabilities': probs[0].cpu().tolist()
        }
    
    # ==================== 模型管理 ====================
    
    def load_model(self, path: str, task: str, region: Optional[str] = None):
        """
        加载已保存的模型
        
        Args:
            path: 模型文件路径 (.pth)
            task: 任务类型
            region: 区域 (仅 single_stream 需要)
        """
        model = self._create_model(TASK_CONFIG[task]['num_classes'])
        model.load_state_dict(torch.load(path, map_location=self.config.DEVICE))
        model.eval()
        self._set_model(model, task, region)
        print(f"已加载模型: {path}")
    
    def save_model(self, model: nn.Module, path: str):
        """
        保存模型到文件
        
        Args:
            model: 模型实例
            path: 保存路径 (.pth)
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(model.state_dict(), path)
        print(f"已保存模型: {path}")
