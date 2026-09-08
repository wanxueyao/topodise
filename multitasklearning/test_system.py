"""
多任务学习系统测试脚本
====================

验证所有模块的功能完整性
"""

import torch
import sys
import traceback
from pathlib import Path

# 测试颜色
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
RESET = '\033[0m'

def test_result(test_name: str, success: bool, error_msg: str = None):
    """打印测试结果"""
    if success:
        print(f'{GREEN}✓{RESET} {test_name}')
    else:
        print(f'{RED}✗{RESET} {test_name}')
        if error_msg:
            print(f'  {RED}错误: {error_msg}{RESET}')


def test_imports():
    """测试所有模块导入"""
    print('\n' + '='*60)
    print('测试 1: 模块导入')
    print('='*60)
    
    tests = {
        '配置模块': 'from multitasklearning import MTLConfig',
        '编码器': 'from multitasklearning.encoder import UNetEncoder',
        '分割分支': 'from multitasklearning.segmentation_branch import SegmentationBranch',
        '分类分支': 'from multitasklearning.classification_branch import ClassificationBranch',
        '主模型': 'from multitasklearning import UNetMultiTask',
        '数据加载': 'from multitasklearning.multitask_data_loader import load_pd_dataset',
        '训练器': 'from multitasklearning.multitask_trainer import MultiTaskTrainer',
        '推理': 'from multitasklearning.inference import MTLInference',
        '工具函数': 'from multitasklearning.multitask_utils import FrangiFilter',
    }
    
    passed = 0
    for name, import_stmt in tests.items():
        try:
            exec(import_stmt)
            test_result(name, True)
            passed += 1
        except Exception as e:
            test_result(name, False, str(e))
    
    return passed, len(tests)


def test_model_creation():
    """测试模型创建"""
    print('\n' + '='*60)
    print('测试 2: 模型创建')
    print('='*60)
    
    from multitasklearning import UNetMultiTask
    
    tests = []
    
    # 测试默认模型
    try:
        model = UNetMultiTask()
        test_result('默认模型创建', True)
        tests.append(True)
    except Exception as e:
        test_result('默认模型创建', False, str(e))
        tests.append(False)
    
    # 测试单流模型
    try:
        model = UNetMultiTask(classification_mode='single_stream')
        test_result('单流模型创建', True)
        tests.append(True)
    except Exception as e:
        test_result('单流模型创建', False, str(e))
        tests.append(False)
    
    # 测试双流模型
    try:
        model = UNetMultiTask(classification_mode='dual_stream')
        test_result('双流模型创建', True)
        tests.append(True)
    except Exception as e:
        test_result('双流模型创建', False, str(e))
        tests.append(False)
    
    # 测试混合单流模型
    try:
        model = UNetMultiTask(classification_mode='mixed_single_stream')
        test_result('混合单流模型创建', True)
        tests.append(True)
    except Exception as e:
        test_result('混合单流模型创建', False, str(e))
        tests.append(False)
    
    # 测试不同编码器
    try:
        model = UNetMultiTask(encoder_type='resnet50', encoder_pretrained=False)
        test_result('ResNet50编码器', True)
        tests.append(True)
    except Exception as e:
        test_result('ResNet50编码器', False, str(e))
        tests.append(False)
    
    return sum(tests), len(tests)


def test_forward_pass():
    """测试前向传播"""
    print('\n' + '='*60)
    print('测试 3: 前向传播')
    print('='*60)
    
    from multitasklearning import UNetMultiTask
    
    model = UNetMultiTask()
    model.eval()
    
    tests = []
    
    # 创建测试数据
    images = torch.randn(2, 3, 256, 256)
    
    # 测试推理模式
    try:
        with torch.no_grad():
            outputs = model(images, training=False, use_mae=False)
        
        assert 'segmentation' in outputs
        assert 'pd_logits' in outputs
        assert outputs['segmentation'].shape == (2, 1, 256, 256)
        assert outputs['pd_logits'].shape == (2, 2)
        
        test_result('推理模式', True)
        tests.append(True)
    except Exception as e:
        test_result('推理模式', False, str(e))
        tests.append(False)
    
    # 测试训练模式
    model.train()
    try:
        outputs = model(images, training=True, use_mae=True)
        
        assert 'losses' in outputs
        assert len(outputs['losses']) > 0
        
        test_result('训练模式', True)
        tests.append(True)
    except Exception as e:
        test_result('训练模式', False, str(e))
        tests.append(False)
    
    # 测试不同输入大小
    try:
        images = torch.randn(4, 3, 512, 512)
        with torch.no_grad():
            outputs = model(images, training=False)
        test_result('不同输入大小', True)
        tests.append(True)
    except Exception as e:
        test_result('不同输入大小', False, str(e))
        tests.append(False)
    
    return sum(tests), len(tests)


def test_data_loading():
    """测试数据加载"""
    print('\n' + '='*60)
    print('测试 4: 数据加载')
    print('='*60)
    
    from multitasklearning.multitask_data_loader import PDDataset
    from torch.utils.data import DataLoader
    
    tests = []
    
    # 创建虚拟数据集
    try:
        patients = [
            {'patient_id': 'p1', 'group': 'pd', 'image_path': 'dummy.jpg', 
             'stage': 2, 'subtype': 'TD'},
            {'patient_id': 'p2', 'group': 'control', 'image_path': 'dummy.jpg'},
        ]
        
        dataset = PDDataset(patients, 'dummy_dir')
        test_result('PDDataset创建', True)
        tests.append(True)
    except Exception as e:
        test_result('PDDataset创建', False, str(e))
        tests.append(False)
    
    # 测试数据加载器
    try:
        from torch.utils.data import TensorDataset
        dummy_data = TensorDataset(torch.randn(10, 3, 256, 256))
        loader = DataLoader(dummy_data, batch_size=2)
        batch = next(iter(loader))
        test_result('DataLoader创建', True)
        tests.append(True)
    except Exception as e:
        test_result('DataLoader创建', False, str(e))
        tests.append(False)
    
    return sum(tests), len(tests)


def test_utils():
    """测试工具函数"""
    print('\n' + '='*60)
    print('测试 5: 工具函数')
    print('='*60)
    
    tests = []
    
    # 测试 Frangi Filter
    try:
        import numpy as np
        from multitasklearning.multitask_utils import FrangiFilter
        
        frangi = FrangiFilter(sigma_range=(1, 5), num_scales=10)
        image = np.random.rand(128, 128).astype(np.float32) * 255
        result = frangi(image)
        
        assert result.shape == image.shape
        assert result.min() >= 0
        assert result.max() <= 1
        
        test_result('FrangiFilter', True)
        tests.append(True)
    except Exception as e:
        test_result('FrangiFilter', False, str(e))
        tests.append(False)
    
    # 测试 Hessian 伪标签
    try:
        import numpy as np
        from multitasklearning.multitask_utils import HessianPseudoLabeler
        
        hessian = HessianPseudoLabeler()
        image = np.random.rand(128, 128).astype(np.float32) * 255
        label = hessian(image)
        
        assert label.shape == image.shape
        assert np.all((label == 0) | (label == 1))
        
        test_result('HessianPseudoLabeler', True)
        tests.append(True)
    except Exception as e:
        test_result('HessianPseudoLabeler', False, str(e))
        tests.append(False)
    
    # 测试评估指标
    try:
        from multitasklearning.multitask_utils import (
            compute_dice_score,
            compute_iou_score,
            compute_classification_metrics
        )
        
        pred = torch.rand(2, 1, 128, 128)
        target = torch.randint(0, 2, (2, 1, 128, 128)).float()
        
        dice = compute_dice_score(pred, target)
        iou = compute_iou_score(pred, target)
        
        assert 0 <= dice <= 1
        assert 0 <= iou <= 1
        
        test_result('分割指标', True)
        tests.append(True)
    except Exception as e:
        test_result('分割指标', False, str(e))
        tests.append(False)
    
    # 测试分类指标
    try:
        pred_cls = torch.randn(10, 2)
        target_cls = torch.randint(0, 2, (10,))
        
        metrics = compute_classification_metrics(pred_cls, target_cls, 2)
        assert 'accuracy' in metrics
        assert 0 <= metrics['accuracy'] <= 1
        
        test_result('分类指标', True)
        tests.append(True)
    except Exception as e:
        test_result('分类指标', False, str(e))
        tests.append(False)
    
    return sum(tests), len(tests)


def test_inference():
    """测试推理模块"""
    print('\n' + '='*60)
    print('测试 6: 推理模块')
    print('='*60)
    
    from multitasklearning import UNetMultiTask
    from multitasklearning.inference import MTLInference
    
    tests = []
    
    model = UNetMultiTask()
    inferencer = MTLInference(model, device='cpu')
    images = torch.randn(2, 3, 256, 256)
    
    # 测试分割推理
    try:
        seg = inferencer.infer_segmentation(images)
        assert seg.shape == (2, 1, 256, 256)
        test_result('分割推理', True)
        tests.append(True)
    except Exception as e:
        test_result('分割推理', False, str(e))
        tests.append(False)
    
    # 测试分类推理
    try:
        cls = inferencer.infer_classification(images)
        assert 'pd_probs' in cls
        assert 'pd_pred' in cls
        test_result('分类推理', True)
        tests.append(True)
    except Exception as e:
        test_result('分类推理', False, str(e))
        tests.append(False)
    
    # 测试联合推理
    try:
        results = inferencer.infer_joint(images)
        assert 'segmentation' in results
        assert 'classification' in results
        test_result('联合推理', True)
        tests.append(True)
    except Exception as e:
        test_result('联合推理', False, str(e))
        tests.append(False)
    
    return sum(tests), len(tests)


def test_config():
    """测试配置管理"""
    print('\n' + '='*60)
    print('测试 7: 配置管理')
    print('='*60)
    
    from multitasklearning import MTLConfig
    from multitasklearning.config import get_config_standard, get_config_lightweight
    
    tests = []
    
    # 测试默认配置
    try:
        config = MTLConfig()
        test_result('默认配置', True)
        tests.append(True)
    except Exception as e:
        test_result('默认配置', False, str(e))
        tests.append(False)
    
    # 测试标准预设
    try:
        config = get_config_standard()
        test_result('标准预设', True)
        tests.append(True)
    except Exception as e:
        test_result('标准预设', False, str(e))
        tests.append(False)
    
    # 测试轻量级预设
    try:
        config = get_config_lightweight()
        test_result('轻量级预设', True)
        tests.append(True)
    except Exception as e:
        test_result('轻量级预设', False, str(e))
        tests.append(False)
    
    # 测试权重归一化
    try:
        config = MTLConfig()
        weights = config.normalize_weights()
        total = sum(weights.values())
        assert abs(total - 1.0) < 1e-5
        test_result('权重归一化', True)
        tests.append(True)
    except Exception as e:
        test_result('权重归一化', False, str(e))
        tests.append(False)
    
    return sum(tests), len(tests)


def test_model_serialization():
    """测试模型保存和加载"""
    print('\n' + '='*60)
    print('测试 8: 模型序列化')
    print('='*60)
    
    from multitasklearning import UNetMultiTask
    import tempfile
    
    tests = []
    
    # 创建临时文件
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        temp_path = f.name
    
    try:
        # 保存模型
        model1 = UNetMultiTask()
        torch.save(model1.state_dict(), temp_path)
        test_result('模型保存', True)
        tests.append(True)
    except Exception as e:
        test_result('模型保存', False, str(e))
        tests.append(False)
    
    try:
        # 加载模型
        model2 = UNetMultiTask()
        model2.load_state_dict(torch.load(temp_path))
        test_result('模型加载', True)
        tests.append(True)
    except Exception as e:
        test_result('模型加载', False, str(e))
        tests.append(False)
    
    try:
        # 验证输出一致性
        x = torch.randn(1, 3, 256, 256)
        with torch.no_grad():
            out1 = model1(x, training=False)
            out2 = model2(x, training=False)
        
        diff = torch.abs(out1['pd_logits'] - out2['pd_logits']).max().item()
        assert diff < 1e-4
        test_result('输出一致性', True)
        tests.append(True)
    except Exception as e:
        test_result('输出一致性', False, str(e))
        tests.append(False)
    
    # 清理
    Path(temp_path).unlink()
    
    return sum(tests), len(tests)


def run_all_tests():
    """运行所有测试"""
    print('\\n' + '='*60)
    print('多任务学习系统 - 完整功能测试')
    print('='*60)
    
    test_functions = [
        test_imports,
        test_model_creation,
        test_forward_pass,
        test_data_loading,
        test_utils,
        test_inference,
        test_config,
        test_model_serialization,
    ]
    
    total_passed = 0
    total_tests = 0
    
    for test_func in test_functions:
        try:
            passed, total = test_func()
            total_passed += passed
            total_tests += total
        except Exception as e:
            print(f'{RED}测试执行出错: {e}{RESET}')
            traceback.print_exc()
    
    # 打印总结
    print('\\n' + '='*60)
    print(f'测试总结: {GREEN}{total_passed}/{total_tests}{RESET} 通过')
    print('='*60)
    
    if total_passed == total_tests:
        print(f'{GREEN}✓ 所有测试通过！系统已准备就绪。{RESET}')
    else:
        print(f'{YELLOW}⚠ 有 {total_tests - total_passed} 个测试未通过{RESET}')
    
    return total_passed, total_tests


if __name__ == '__main__':
    # 禁用梯度计算，加快测试
    torch.set_grad_enabled(False)
    
    passed, total = run_all_tests()
    
    # 退出码
    sys.exit(0 if passed == total else 1)
