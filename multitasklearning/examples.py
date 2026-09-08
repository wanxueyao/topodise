"""
完整使用示例
===========

演示多任务学习系统的各个功能
"""

import torch
import numpy as np
from pathlib import Path

# ==================== 导入 ====================
from multitasklearning import (
    MTLConfig, 
    UNetMultiTask,
    FrangiFilter,
)
from multitasklearning.config import get_config_standard
from multitasklearning.inference import MTLInference, MTLEvaluator
from multitasklearning.multitask_utils import batch_pseudo_labels, HessianPseudoLabeler
from multitasklearning.helpers import set_seed


def example_1_model_creation():
    """示例1：创建模型"""
    print("\n" + "="*60)
    print("示例1：创建模型")
    print("="*60)
    
    # 方式1：使用默认配置
    model = UNetMultiTask()
    print("✓ 创建默认双流模型")
    
    # 方式2：使用预定义配置
    config = get_config_standard()
    model = UNetMultiTask(
        encoder_type=config.encoder_type,
        classification_mode=config.classification_mode,
        dropout_rate=0.5
    )
    print("✓ 使用标准配置创建模型")
    
    # 方式3：自定义架构
    model = UNetMultiTask(
        encoder_type='efficientnet_b4',
        classification_mode='mixed_single_stream',
        mae_enabled=True,
        dropout_rate=0.6
    )
    print("✓ 创建自定义架构的大型模型")
    
    # 打印模型信息
    params = model.count_parameters()
    print(f"\n模型参数数量:")
    for key, val in params.items():
        print(f"  {key}: {val:,}")


def example_2_forward_pass():
    """示例2：前向传播"""
    print("\n" + "="*60)
    print("示例2：前向传播")
    print("="*60)
    
    model = UNetMultiTask()
    
    # 准备数据
    B, C, H, W = 4, 3, 256, 256
    images = torch.randn(B, C, H, W)
    
    # 训练模式
    print("\n[训练模式]")
    outputs = model(images, training=True, use_mae=True)
    
    print(f"  分割输出: {outputs['segmentation'].shape}")
    print(f"  PD分类: {outputs['pd_logits'].shape}")
    print(f"  HY分级: {outputs['stage_logits'].shape}")
    print(f"  亚型分类: {outputs['subtype_logits'].shape}")
    print(f"  MAE重构: {outputs['mae_reconstructed'].shape}")
    
    # 推理模式
    print("\n[推理模式]")
    with torch.no_grad():
        outputs = model(images, training=False, use_mae=False)
    
    print(f"  分割输出: {outputs['segmentation'].shape}")
    print(f"  分类输出确认: {len(outputs['losses']) == 0}")


def example_3_inference():
    """示例3：模型推理"""
    print("\n" + "="*60)
    print("示例3：模型推理")
    print("="*60)
    
    model = UNetMultiTask()
    model.eval()
    
    # 创建推理器
    inferencer = MTLInference(model, device='cpu')
    
    # 准备测试数据
    images = torch.randn(2, 3, 256, 256)
    
    # 分割推理
    print("\n[分割推理]")
    segmentation = inferencer.infer_segmentation(images)
    print(f"  输出形状: {segmentation.shape}")
    print(f"  值范围: [{segmentation.min():.3f}, {segmentation.max():.3f}]")
    
    # 分类推理
    print("\n[分类推理]")
    classification = inferencer.infer_classification(images)
    print(f"  PD概率: {classification['pd_probs'].shape}")
    print(f"  PD预测: {classification['pd_pred']}")
    print(f"  HY分级预测: {classification['stage_pred']}")
    print(f"  亚型预测: {classification['subtype_pred']}")
    
    # 联合推理
    print("\n[联合推理]")
    results = inferencer.infer_joint(images)
    print(f"  分割 + 分类完成")


def example_4_frangi_filter():
    """示例4：Frangi Filter伪标签"""
    print("\n" + "="*60)
    print("示例4：Frangi Filter伪标签生成")
    print("="*60)
    
    # 创建Frangi滤波器
    frangi = FrangiFilter(sigma_range=(1, 10), num_scales=30)
    
    # 创建测试图像 (模拟血管/神经纤维)
    np.random.seed(42)
    image = np.random.rand(256, 256) * 255
    
    # 应用Frangi滤波
    print("\n[单张图像处理]")
    enhanced = frangi(image)
    print(f"  输入形状: {image.shape}")
    print(f"  输出形状: {enhanced.shape}")
    print(f"  输出范围: [{enhanced.min():.3f}, {enhanced.max():.3f}]")
    
    # 批量伪标签生成
    print("\n[批量处理]")
    images = np.random.rand(4, 256, 256, 3) * 255
    pseudo_labels = batch_pseudo_labels(images, method='frangi')
    print(f"  输入批次大小: {images.shape}")
    print(f"  伪标签形状: {pseudo_labels.shape}")
    print(f"  伪标签值: {torch.unique(pseudo_labels)}")
    
    # Hessian伪标签
    print("\n[Hessian伪标签]")
    hessian_labeler = HessianPseudoLabeler(sigma=2.0)
    hessian_label = hessian_labeler(image)
    print(f"  Hessian伪标签形状: {hessian_label.shape}")
    print(f"  Hessian伪标签值: {np.unique(hessian_label)}")


def example_5_configuration():
    """示例5：配置管理"""
    print("\n" + "="*60)
    print("示例5：配置管理")
    print("="*60)
    
    # 创建配置
    config = MTLConfig(
        encoder_type='resnet50',
        batch_size=32,
        num_epochs=100,
        learning_rate=1e-4,
        loss_weight_segmentation=0.3,
        loss_weight_mae=0.2,
        loss_weight_pd_classification=0.2,
        loss_weight_stage=0.1,
        loss_weight_subtype=0.1,
    )
    
    # 打印配置
    print("\n[基本配置]")
    print(f"  编码器: {config.encoder_type}")
    print(f"  批大小: {config.batch_size}")
    print(f"  训练轮数: {config.num_epochs}")
    print(f"  学习率: {config.learning_rate}")
    
    # 归一化权重
    print("\n[任务权重 (归一化)]")
    weights = config.normalize_weights()
    for task, weight in weights.items():
        print(f"  {task}: {weight:.3f}")
    
    # 验证配置
    print("\n[配置验证]")
    print(f"  ✓ MAE mask ratio: {config.mae_mask_ratio}")
    print(f"  ✓ PD子类型数: {config.num_pd_subtypes}")
    print(f"  ✓ 总权重: {config.get_total_weight():.3f}")


def example_6_multi_architecture():
    """示例6：比较多种架构"""
    print("\n" + "="*60)
    print("示例6：比较多种架构")
    print("="*60)
    
    architectures = {
        'single_stream': 'single_stream',
        'dual_stream': 'dual_stream',
        'mixed_single_stream': 'mixed_single_stream',
    }
    
    images = torch.randn(2, 3, 256, 256)
    
    for arch_name, arch_mode in architectures.items():
        print(f"\n[{arch_name.upper()}]")
        
        model = UNetMultiTask(
            encoder_type='resnet50',
            classification_mode=arch_mode,
            dropout_rate=0.5
        )
        
        with torch.no_grad():
            outputs = model(images, training=False)
        
        print(f"  PD分类输出: {outputs['pd_logits'].shape}")
        print(f"  输出维度: {outputs['pd_logits'].size(1)} (2=PD vs Control)")
        
        if arch_name == 'dual_stream':
            print(f"  分割流输出: {outputs.get('seg_pd_logits', 'N/A')}")
            print(f"  解码流输出: {outputs.get('dec_pd_logits', 'N/A')}")


def example_7_training_setup():
    """示例7：训练设置"""
    print("\n" + "="*60)
    print("示例7：训练设置 (准备阶段)")
    print("="*60)
    
    # 设置随机种子
    set_seed(42)
    print("✓ 随机种子设置完成")
    
    # 创建配置
    config = get_config_standard()
    print(f"✓ 配置创建完成: {config.encoder_type}")
    
    # 创建模型
    model = UNetMultiTask(
        encoder_type=config.encoder_type,
        classification_mode=config.classification_mode,
    )
    model.to(config.device)
    print(f"✓ 模型创建完成，设备: {config.device}")
    
    # 创建优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    print(f"✓ 优化器创建完成: AdamW")
    
    # 创建学习率调度
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.num_epochs,
        eta_min=1e-7
    )
    print(f"✓ 学习率调度创建完成: CosineAnnealingLR")
    
    print("\n[训练超参数]")
    print(f"  批大小: {config.batch_size}")
    print(f"  轮数: {config.num_epochs}")
    print(f"  初始学习率: {config.learning_rate}")
    print(f"  优化器: {type(optimizer).__name__}")


def example_8_model_export():
    """示例8：模型导出和加载"""
    print("\n" + "="*60)
    print("示例8：模型导出和加载")
    print("="*60)
    
    # 创建并保存模型
    model = UNetMultiTask()
    checkpoint_path = Path('temp_checkpoint.pt')
    
    print("\n[保存模型]")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"  ✓ 模型保存到: {checkpoint_path}")
    print(f"  文件大小: {checkpoint_path.stat().st_size / (1024**2):.1f} MB")
    
    # 加载模型
    print("\n[加载模型]")
    new_model = UNetMultiTask()
    new_model.load_state_dict(torch.load(checkpoint_path))
    print(f"  ✓ 模型加载完成")
    
    # 验证
    print("\n[验证]")
    x = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        out1 = model(x, training=False)
        out2 = new_model(x, training=False)
    
    diff = torch.abs(out1['pd_logits'] - out2['pd_logits']).max().item()
    print(f"  ✓ 输出差异: {diff:.1e} (接近0表示加载正确)")
    
    # 清理
    checkpoint_path.unlink()
    print(f"  ✓ 临时文件已删除")


# ==================== 主程序 ====================

if __name__ == '__main__':
    print("\n" + "="*60)
    print("多任务学习系统 - 完整使用示例")
    print("="*60)
    
    # 确保在CPU上运行这些示例
    torch.set_printoptions(precision=3)
    
    # 运行所有示例
    example_1_model_creation()
    example_2_forward_pass()
    example_3_inference()
    example_4_frangi_filter()
    example_5_configuration()
    example_6_multi_architecture()
    example_7_training_setup()
    example_8_model_export()
    
    print("\n" + "="*60)
    print("所有示例运行完成！")
    print("="*60)
