"""
多任务学习示例训练脚本
====================

演示如何使用多任务学习系统进行训练

使用方法:
    python train_multitask.py --mode single_stream --epochs 100 --batch_size 32
"""

import argparse
import logging
import sys
import torch
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from multitasklearning.config import MTLConfig, get_config_standard, get_config_lightweight
from multitasklearning.unet_multitask import UNetMultiTask, MTLBuilder
from multitasklearning.multitask_data_loader import load_corn1_dataset, load_pd_dataset
from multitasklearning.multitask_trainer import MultiTaskTrainer
from multitasklearning.helpers import set_seed

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='多任务学习训练脚本')
    
    # 模型配置
    parser.add_argument('--mode', type=str, default='dual_stream',
                       choices=['single_stream', 'dual_stream', 'mixed_single_stream'],
                       help='分类架构模式')
    parser.add_argument('--encoder', type=str, default='resnet50',
                       choices=['resnet18', 'resnet50', 'efficientnet_b4'],
                       help='编码器类型')
    parser.add_argument('--config', type=str, default='standard',
                       choices=['lightweight', 'standard', 'large'],
                       help='预定义配置')
    
    # 训练配置
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32, help='批大小')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout比例')
    
    # 数据配置
    parser.add_argument('--corn1_root', type=str, default='datasets/CORN-1',
                       help='CORN-1数据集路径')
    parser.add_argument('--pd_root', type=str, default='datasets/pd_dataset_en',
                       help='PD数据集路径')
    parser.add_argument('--kfold', type=int, default=0, help='K-fold索引')
    parser.add_argument('--kfold_splits', type=int, default=5, help='K-fold分割数')
    
    # 其他配置
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='设备')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--output_dir', type=str, default='checkpoints',
                       help='输出目录')
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 创建配置
    if args.config == 'lightweight':
        config = get_config_lightweight()
    elif args.config == 'large':
        from config import get_config_large
        config = get_config_large()
    else:
        config = get_config_standard()
    
    # 更新配置
    config.encoder_type = args.encoder
    config.classification_mode = args.mode
    config.num_epochs = args.epochs
    config.batch_size = args.batch_size
    config.learning_rate = args.lr
    config.device = args.device
    config.checkpoint_dir = args.output_dir
    
    logger.info(f"配置: {config}")
    
    # 构建模型
    logger.info(f"构建 {args.mode} 架构的模型...")
    model = UNetMultiTask(
        encoder_type=config.encoder_type,
        encoder_pretrained=True,
        classification_mode=config.classification_mode,
        dropout_rate=args.dropout
    )
    
    # 打印模型信息
    params = model.count_parameters()
    logger.info(f"模型参数数量: {params['total']:,}")
    
    # 加载数据
    logger.info("加载数据...")
    
    # 加载 CORN-1 数据
    try:
        corn1_train, corn1_val = load_corn1_dataset(
            args.corn1_root,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            augment=True
        )
        logger.info(f"CORN-1 数据加载成功")
    except Exception as e:
        logger.warning(f"CORN-1 数据加载失败: {e}")
        corn1_train, corn1_val = None, None
    
    # 加载 PD 数据
    try:
        pd_train = load_pd_dataset(
            args.pd_root,
            split='train',
            arch_mode=args.mode,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            kfold_idx=args.kfold,
            kfold_splits=args.kfold_splits,
            augment=True
        )
        pd_val = load_pd_dataset(
            args.pd_root,
            split='val',
            arch_mode=args.mode,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            kfold_idx=args.kfold,
            kfold_splits=args.kfold_splits,
            augment=False
        )
        logger.info(f"PD 数据加载成功 (K-fold={args.kfold})")
    except Exception as e:
        logger.warning(f"PD 数据加载失败: {e}")
        pd_train, pd_val = None, None
    
    # 使用可用的数据加载器
    if corn1_train is not None and pd_train is not None:
        # 使用多任务数据加载器
        from multitask_data_loader import MultiTaskDataLoader
        train_loader = MultiTaskDataLoader(corn1_train, pd_train, sampling_ratio=0.5)
        val_loader = None  # 可以创建多任务验证加载器
    elif pd_train is not None:
        train_loader = pd_train
        val_loader = pd_val
    elif corn1_train is not None:
        train_loader = corn1_train
        val_loader = corn1_val
    else:
        raise RuntimeError("无法加载任何数据集")
    
    # 创建训练器
    logger.info("创建训练器...")
    trainer = MultiTaskTrainer(
        config=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader
    )
    
    # 训练
    logger.info("开始训练...")
    trainer.train()
    
    logger.info("训练完成！")
    logger.info(f"输出目录: {trainer.output_dir}")


if __name__ == '__main__':
    main()
