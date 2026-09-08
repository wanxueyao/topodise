"""
main.py - 帕金森病图像分类实验主程序

模块职责
--------
执行单组配置下的完整 K-fold 交叉验证实验:
1. 加载数据并生成 K-fold 划分
2. 逐 fold 训练和验证
3. 汇总指标 (Acc, F1, AUC)
4. 生成可视化 (训练曲线、混淆矩阵、ROC)
5. 输出实验报告 (Markdown 格式)

接口设计
--------
本模块为单次实验入口，为未来批量实验预留以下接口:

输入接口:
    - ExperimentConfig: 可通过修改类属性或继承来自定义配置
    - run_single_experiment(config): 接受配置对象，返回结果字典
    
输出接口:
    - 返回值: Dict[str, Any] 包含所有任务的详细结果
    - 输出目录: results/<arch>_<dataset>_<backbone>_<epochs>_<timestamp>/
        - config.txt: 配置记录
        - output.txt: 控制台日志
        - experiment_report.md: 实验报告
        - *.png: 可视化图表
        - best_model_*.pth: 最佳模型

未来批量实验调用示例:
    >>> from main import run_single_experiment, ExperimentConfig
    >>> 
    >>> configs = []
    >>> for backbone in ['resnet18', 'resnet34']:
    ...     for pretrained in [True, 'Half']:
    ...         cfg = ExperimentConfig()
    ...         cfg.BACKBONE = backbone
    ...         cfg.PRETRAINED = pretrained
    ...         configs.append(cfg)
    >>> 
    >>> all_results = {str(i): run_single_experiment(cfg) for i, cfg in enumerate(configs)}
"""
import sys
import os
import time
import argparse
import torch
import numpy as np
from datetime import datetime
from typing import Literal, List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.basic_classifier import PDClassifier, PDClassifierConfig, TASK_CONFIG
from core.data_loader import PDDataset, load_and_filter_dataset, stratified_kfold, KFoldMode
from utils.patient_outputs import save_fold_patient_outputs, task_labels_for_indices
from utils.misc import set_seed, Tee, create_output_dir, save_config
from utils.visualize import plot_confusion_matrix, plot_roc_auc, plot_training_curves


# ==================== 实验配置 ====================

class ExperimentConfig(PDClassifierConfig):
    """
    实验配置类 (继承自 PDClassifierConfig)
    
    通过修改类属性来配置实验参数。
    批量实验时可创建多个配置实例。
    
    配置分组
    --------
    架构配置:
        ARCHITECTURE: 'single_stream' / 'mixed_single_stream' / 'dual_stream'
            - single_stream: 每个区域独立训练
            - mixed_single_stream: 混合所有区域
            - dual_stream: 双流输入
        BACKBONE: 'resnet18' / 'resnet34' / 'densenet121' / 'efficientnet'
        PRETRAINED: True / False / 'Half'
    
    数据配置:
        DATASET_MODE: 'pd_control' / 'pd_control_cataract' / 'pd_all'
        DATASET_JSON: 数据集 JSON 文件路径
    
    训练配置:
        BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY
    
    交叉验证:
        NUM_FOLDS: K-fold 的 K 值
        SEED: 随机种子 (确保可复现)
    
    输出配置:
        OUTPUT_DIR: 结果输出根目录
    """
    # ========== 架构 ==========
    ARCHITECTURE: Literal['single_stream', 'mixed_single_stream', 'dual_stream'] = 'single_stream'
    BACKBONE: Literal['resnet18', 'resnet34', 'densenet121', 'efficientnet'] = 'resnet18'
    PRETRAINED: Literal[True, False, 'Half'] = True
    
    # ========== 数据 ==========
    DATASET_MODE: Literal['pd_control', 'pd_control_cataract', 'pd_all'] = 'pd_all'
    DATASET_JSON: str = 'datasets/pd_dataset_en/patients_dataset_en.json'
    
    # ========== 训练 ==========
    BATCH_SIZE: int = 16
    NUM_EPOCHS: int = 50
    LEARNING_RATE: float = 0.001
    WEIGHT_DECAY: float = 1e-4
    
    # ========== 学习率调度 ==========
    LR_SCHEDULER: str = 'cosine'        # 可选 step、cosine 或 none
    LR_STEP_SIZE: int = 15              # StepLR 步长
    LR_GAMMA: float = 0.5               # StepLR 衰减因子
    BACKBONE_LR_RATIO: float = 0.1      # backbone 学习率 = LR * ratio
    
    # ========== 早停 ==========
    EARLY_STOPPING: bool = True
    PATIENCE: int = 10
    
    # ========== 数据增强 ==========
    DATA_AUGMENTATION: bool = True              # 是否启用数据增强
    AUG_HORIZONTAL_FLIP: float = 0.5            # 水平翻转概率
    AUG_ROTATION: float = 10.0                  # 旋转角度范围 (度)
    AUG_BRIGHTNESS: float = 0.2                 # 亮度抖动范围
    AUG_CONTRAST: float = 0.2                   # 对比度抖动范围
    AUG_SATURATION: float = 0.0                 # 饱和度抖动范围
    AUG_HUE: float = 0.0                        # 色调抖动范围
    
    # ========== 交叉验证 ==========
    NUM_FOLDS: int = 5
    SEED: int = 42
    KFOLD_MODE: Literal['patient', 'image'] = 'patient'  # 'patient': 按患者划分（避免数据泄露）, 'image': 按图像划分
    
    # ========== 系统 ==========
    NUM_WORKERS: int = 4
    DEVICE: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    OUTPUT_DIR: str = 'results'


# ==================== 实验运行器 ====================

class ExperimentRunner:
    """
    单次实验运行器
    
    执行一组配置下的完整 K-fold 实验流程。
    
    使用方法
    --------
    >>> config = ExperimentConfig()
    >>> config.BACKBONE = 'resnet34'
    >>> runner = ExperimentRunner(config)
    >>> results = runner.run(tasks=['pd'])
    
    属性
    ----
    config : ExperimentConfig
        实验配置
    classifier : PDClassifier
        分类器实例
    output_dir : str
        输出目录路径
    """
    
    def __init__(self, config: ExperimentConfig):
        """
        初始化实验运行器
        
        Args:
            config: 实验配置对象
        """
        self.config = config
        self.classifier = PDClassifier(config)
        self.output_dir: Optional[str] = None
    
    def run(self, tasks: Optional[List[str]] = None, 
            output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        运行完整 K-fold 实验
        
        Args:
            tasks: 要执行的任务列表，默认 ['pd', 'stage', 'subtype']
            output_dir: 自定义输出目录，默认自动生成带时间戳的目录
            
        Returns:
            实验结果字典，结构:
            {
                'pd': {
                    'regions': {...},  # single_stream 时按区域分组
                    'final_results': {...},
                    'fold_results': [...]
                },
                'stage': {...},
                'subtype': {...},
                'meta': {
                    'config': {...},
                    'duration': float,
                    'output_dir': str
                }
            }
        """
        set_seed(self.config.SEED)
        start_time = time.time()
        
        if tasks is None:
            tasks = ['pd', 'stage', 'subtype']
        
        # 创建输出目录
        self.output_dir = output_dir or create_output_dir('results', self.config)
        os.makedirs(self.output_dir, exist_ok=True)
        self.config.OUTPUT_DIR = self.output_dir
        save_config(self.config, self.output_dir)
        
        # 设置日志双写
        log_file = open(os.path.join(self.output_dir, 'output.txt'), 'w', encoding='utf-8')
        original_stdout = sys.stdout
        sys.stdout = Tee(original_stdout, log_file)
        
        try:
            results = self._execute_experiment(tasks)
        finally:
            sys.stdout = original_stdout
            log_file.close()
        
        # 添加元数据
        duration = time.time() - start_time
        results['meta'] = {
            'config': self._config_to_dict(),
            'duration': duration,
            'output_dir': self.output_dir
        }
        
        print(f"\n实验完成！用时 {duration/60:.2f} 分钟")
        print(f"结果保存到: {self.output_dir}")
        
        return results
    
    def _execute_experiment(self, tasks: List[str]) -> Dict[str, Any]:
        """执行实验主流程"""
        config = self.config
        self._print_header()
        
        # 加载数据
        patients = load_and_filter_dataset(config.DATASET_JSON, config.DATASET_MODE)
        print(f"\n加载 {len(patients)} 个患者")
        
        results = {}
        
        if config.ARCHITECTURE == 'single_stream':
            # 每个任务每个区域分别训练
            for task in tasks:
                results[task] = {'regions': {}}
                for region in ['central', 'whorl']:
                    print(f"\n{'='*60}")
                    print(f"任务: {TASK_CONFIG[task]['name']} | 区域: {region}")
                    print('='*60)
                    
                    dataset = PDDataset(
                        patients, 'single_stream',
                        self.classifier.train_transform,
                        region=region, base_path='datasets'
                    )
                    if len(dataset) == 0:
                        print(f"⚠ 区域 {region} 无数据，跳过")
                        continue
                    
                    print(f"样本数: {len(dataset)}")
                    region_result = self._run_kfold(dataset, task, f"_{region}")
                    results[task]['regions'][region] = region_result
        else:
            # mixed_single_stream 或 dual_stream
            dataset = PDDataset(
                patients, config.ARCHITECTURE,
                self.classifier.train_transform,
                base_path='datasets'
            )
            
            if len(dataset) == 0:
                print("⚠ 数据集为空")
                return results
            
            print(f"样本数: {len(dataset)}")
            
            for task in tasks:
                print(f"\n{'='*60}")
                print(f"任务: {TASK_CONFIG[task]['name']}")
                print('='*60)
                
                results[task] = self._run_kfold(dataset, task)
        
        # 生成报告
        self._generate_report(results)
        
        return results
    
    def _run_kfold(self, dataset, task: str, suffix: str = "") -> Dict[str, Any]:
        """执行单任务 K-fold 交叉验证"""
        config = self.config
        task_cfg = TASK_CONFIG[task]
        
        # K-fold 划分
        splits = list(stratified_kfold(
            dataset.PD_labels, dataset.stage_labels, dataset.subtype_labels,
            n_splits=config.NUM_FOLDS, random_state=config.SEED,
            kfold_mode=config.KFOLD_MODE, patient_ids=dataset.patient_ids
        ))
        
        print(f"K-fold 模式: {config.KFOLD_MODE} (按{'患者' if config.KFOLD_MODE == 'patient' else '图像'}划分)")
        
        fold_results = []
        all_preds = {'labels': [], 'preds': [], 'probs': []}
        best_models = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            print(f"\n--- Fold {fold_idx+1}/{config.NUM_FOLDS} ---")
            
            train_subset = torch.utils.data.Subset(dataset, train_idx)
            val_subset = torch.utils.data.Subset(dataset, val_idx)
            
            # 训练
            result = self.classifier.train(
                train_subset, val_subset, task,
                fold_idx=fold_idx+1, suffix=suffix
            )
            
            # 绘制训练曲线
            plot_training_curves(
                result['train_history'],
                task_cfg['name'] + suffix,
                fold_idx+1, self.output_dir
            )
            
            val_result = result['val_result']
            
            if val_result['labels']:
                patient_labels, patient_ids = task_labels_for_indices(dataset, val_idx, task)
                patient_result = save_fold_patient_outputs(
                    self.output_dir, task, suffix, fold_idx+1,
                    patient_ids, patient_labels,
                    probs=val_result.get('probs'), logits=val_result.get('logits')
                )
                fold_results.append({
                    'fold': fold_idx+1,
                    'train_acc': result['train_history']['train_acc'][-1] if 'train_acc' in result['train_history'] else 0.0,
                    'acc': val_result['acc'],
                    'patient_acc': patient_result['patient_acc'],
                    'n_patients': patient_result['n_patients'],
                    'patient_logits_json': patient_result['patient_logits_json'],
                    'patient_logits_csv': patient_result['patient_logits_csv'],
                    'precision': val_result['precision'],
                    'recall': val_result['recall'],
                    'f1': val_result['f1']
                })
                all_preds['labels'].extend(val_result['labels'])
                all_preds['preds'].extend(val_result['preds'])
                if len(val_result['probs']) > 0:
                    all_preds['probs'].extend(val_result['probs'].tolist())
                
                print(f"  Acc: {val_result['acc']:.4f}, PatientAcc: {patient_result['patient_acc']:.4f}, F1: {val_result['f1']:.4f}")
            
            best_models.append({
                'fold': fold_idx+1,
                'state': result['model_state'],
                'acc': val_result['acc']
            })
        
        if not fold_results:
            return {}
        
        # 汇总结果
        summary = self._summarize_results(fold_results, all_preds, task_cfg, suffix)
        
        # 保存最佳模型
        best = max(best_models, key=lambda x: x['acc'])
        model_path = os.path.join(self.output_dir, f"best_model_{task}{suffix}.pth")
        torch.save(best['state'], model_path)
        print(f"✓ 保存最佳模型: {model_path}")
        
        return summary
    
    def _summarize_results(self, fold_results: List[Dict], all_preds: Dict,
                           task_cfg: Dict, suffix: str) -> Dict[str, Any]:
        """汇总 K-fold 结果并生成可视化"""
        task_name = task_cfg['name'] + suffix
        class_names = task_cfg['class_names']
        
        # 计算统计量
        avg_train_acc = np.mean([r.get('train_acc', 0) for r in fold_results])
        avg_acc = np.mean([r['acc'] for r in fold_results])
        std_acc = np.std([r['acc'] for r in fold_results])
        avg_patient_acc = np.mean([r.get('patient_acc', 0.0) for r in fold_results])
        std_patient_acc = np.std([r.get('patient_acc', 0.0) for r in fold_results])
        avg_f1 = np.mean([r['f1'] for r in fold_results])
        std_f1 = np.std([r['f1'] for r in fold_results])
        
        print(f"\n📊 {task_name}: TrainAcc={avg_train_acc:.4f}, ValAcc={avg_acc:.4f}±{std_acc:.4f}, PatientAcc={avg_patient_acc:.4f}±{std_patient_acc:.4f}, ValF1={avg_f1:.4f}±{std_f1:.4f}")
        
        # 生成可视化
        auc_scores = {}
        if all_preds['labels']:
            safe_name = task_name.replace(' ', '_').replace('/', '_')
            
            # 获取实际出现的类别
            unique_labels = sorted(set(all_preds['labels'] + all_preds['preds']))
            actual_names = [class_names[l] if l < len(class_names) else str(l) for l in unique_labels]
            
            # 混淆矩阵
            plot_confusion_matrix(
                all_preds['labels'], all_preds['preds'],
                actual_names, safe_name, 0, self.output_dir, is_test=True
            )
            
            # ROC 曲线
            if all_preds['probs']:
                auc = plot_roc_auc(
                    all_preds['labels'],
                    np.array(all_preds['probs']),
                    actual_names, safe_name, self.output_dir
                )
                if isinstance(auc, dict):
                    auc_scores = {actual_names[i]: auc[i] for i in auc if isinstance(i, int)}
                else:
                    auc_scores = {'overall': auc}
                print(f"   AUC: {auc_scores}")
        
        return {
            'fold_results': fold_results,
            'final_results': {
                'avg_train_acc': avg_train_acc,
                'avg_acc': avg_acc, 'std_acc': std_acc,
                'avg_patient_acc': avg_patient_acc, 'std_patient_acc': std_patient_acc,
                'avg_f1': avg_f1, 'std_f1': std_f1,
                'auc_scores': auc_scores
            },
            'all_predictions': all_preds
        }
    
    def _print_header(self):
        """打印实验信息头"""
        cfg = self.config
        print("="*60)
        print("帕金森病图像分类实验")
        print("="*60)
        print(f"架构: {cfg.ARCHITECTURE} | Backbone: {cfg.BACKBONE}")
        print(f"预训练: {cfg.PRETRAINED} | 数据集: {cfg.DATASET_MODE}")
        print(f"Epochs: {cfg.NUM_EPOCHS} | K-fold: {cfg.NUM_FOLDS}")
        print(f"K-fold模式: {cfg.KFOLD_MODE} (按{'患者' if cfg.KFOLD_MODE == 'patient' else '图像'}划分)")
        print(f"LR: {cfg.LEARNING_RATE} | Batch: {cfg.BATCH_SIZE}")
        if cfg.DATA_AUGMENTATION:
            print(f"数据增强: 开启 (翻转={cfg.AUG_HORIZONTAL_FLIP}, 旋转={cfg.AUG_ROTATION}°, "
                  f"亮度={cfg.AUG_BRIGHTNESS}, 对比度={cfg.AUG_CONTRAST})")
        else:
            print("数据增强: 关闭")
        print("="*60)
    
    def _generate_report(self, results: Dict):
        """生成 Markdown 实验报告 (含嵌入图像)"""
        report_path = os.path.join(self.output_dir, 'experiment_report.md')
        cfg = self.config
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 实验报告\n\n")
            f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # 配置表
            f.write("## 实验配置\n\n")
            f.write("| 参数 | 值 |\n|---|---|\n")
            f.write(f"| 架构 | `{cfg.ARCHITECTURE}` |\n")
            f.write(f"| Backbone | `{cfg.BACKBONE}` |\n")
            f.write(f"| 预训练 | `{cfg.PRETRAINED}` |\n")
            f.write(f"| 数据集 | `{cfg.DATASET_MODE}` |\n")
            f.write(f"| K-fold | {cfg.NUM_FOLDS} |\n")
            f.write(f"| K-fold模式 | `{cfg.KFOLD_MODE}` (按{'患者' if cfg.KFOLD_MODE == 'patient' else '图像'}划分) |\n")
            f.write(f"| Epochs | {cfg.NUM_EPOCHS} |\n")
            f.write(f"| 学习率 | {cfg.LEARNING_RATE} |\n")
            f.write(f"| Batch Size | {cfg.BATCH_SIZE} |\n")
            f.write(f"| 数据增强 | `{'开启' if cfg.DATA_AUGMENTATION else '关闭'}` |\n")
            if cfg.DATA_AUGMENTATION:
                f.write(f"| - 水平翻转 | {cfg.AUG_HORIZONTAL_FLIP} |\n")
                f.write(f"| - 旋转 | ±{cfg.AUG_ROTATION}° |\n")
                f.write(f"| - 亮度抖动 | {cfg.AUG_BRIGHTNESS} |\n")
                f.write(f"| - 对比度抖动 | {cfg.AUG_CONTRAST} |\n")
                if cfg.AUG_SATURATION > 0:
                    f.write(f"| - 饱和度抖动 | {cfg.AUG_SATURATION} |\n")
                if cfg.AUG_HUE > 0:
                    f.write(f"| - 色调抖动 | {cfg.AUG_HUE} |\n")
            f.write("\n")
            
            # 结果摘要表
            f.write("## 结果摘要\n\n")
            f.write("| 任务 | 区域 | Accuracy | F1 Score |\n")
            f.write("|---|---|---|---|\n")
            
            for task, result in results.items():
                if task == 'meta':
                    continue
                task_name = TASK_CONFIG[task]['name']
                
                if 'regions' in result:
                    for region, rr in result['regions'].items():
                        if 'final_results' in rr:
                            fr = rr['final_results']
                            f.write(f"| {task_name} | {region} | "
                                   f"{fr['avg_acc']:.4f}±{fr['std_acc']:.4f} | "
                                   f"{fr['avg_f1']:.4f}±{fr['std_f1']:.4f} |\n")
                elif 'final_results' in result:
                    fr = result['final_results']
                    f.write(f"| {task_name} | - | "
                           f"{fr['avg_acc']:.4f}±{fr['std_acc']:.4f} | "
                           f"{fr['avg_f1']:.4f}±{fr['std_f1']:.4f} |\n")
            
            # 详细结果 (含图像)
            f.write("\n---\n\n## 详细结果\n\n")
            
            for task, result in results.items():
                if task == 'meta':
                    continue
                task_name = TASK_CONFIG[task]['name']
                
                if 'regions' in result:
                    for region, rr in result['regions'].items():
                        self._write_task_section(f, task_name, region, rr)
                elif 'final_results' in result:
                    self._write_task_section(f, task_name, None, result)
            
            # 输出文件
            f.write("\n---\n\n## 附件说明\n\n")
            f.write("| 文件 | 说明 |\n|---|---|\n")
            f.write("| `config.txt` | 完整配置 |\n")
            f.write("| `output.txt` | 训练日志 |\n")
            f.write("| `best_model_*.pth` | 最佳模型权重 |\n")
        
        print(f"✓ 生成报告: {report_path}")
    
    def _write_task_section(self, f, task_name: str, region: str, result: Dict):
        """为单个任务/区域写入详细结果段落"""
        suffix = f"_{region}" if region else ""
        display_name = f"{task_name}{suffix}" if region else task_name
        
        f.write(f"### {display_name}\n\n")
        
        if 'final_results' not in result:
            f.write("*无有效结果*\n\n")
            return
        
        fr = result['final_results']
        
        # 指标表
        f.write("| 指标 | 值 |\n|---|---|\n")
        f.write(f"| Accuracy | {fr['avg_acc']:.4f} ± {fr['std_acc']:.4f} |\n")
        f.write(f"| F1 Score | {fr['avg_f1']:.4f} ± {fr['std_f1']:.4f} |\n")
        if 'avg_precision' in fr:
            f.write(f"| Precision | {fr['avg_precision']:.4f} ± {fr['std_precision']:.4f} |\n")
        if 'avg_recall' in fr:
            f.write(f"| Recall | {fr['avg_recall']:.4f} ± {fr['std_recall']:.4f} |\n")
        f.write("\n")
        
        # AUC 信息
        if 'auc' in fr:
            f.write("**AUC**:\n")
            for cls, auc_val in fr['auc'].items():
                f.write(f"- {cls}: {auc_val:.4f}\n")
            f.write("\n")
        
        # 混淆矩阵和 ROC 曲线
        cm_file = f"{task_name}{suffix}_Test_cm.png"
        roc_file = f"{task_name}{suffix}_Test_roc.png"
        
        if os.path.exists(os.path.join(self.output_dir, cm_file)):
            f.write(f"**混淆矩阵**:\n\n![混淆矩阵]({cm_file})\n\n")
        
        if os.path.exists(os.path.join(self.output_dir, roc_file)):
            f.write(f"**ROC 曲线**:\n\n![ROC曲线]({roc_file})\n\n")
        
        # 训练曲线 (所有 fold)
        curve_files = [fn for fn in os.listdir(self.output_dir) 
                      if fn.startswith(f"{task_name}{suffix}_fold") and fn.endswith("_curves.png")]
        
        if curve_files:
            f.write("**训练曲线**:\n\n")
            for curve_file in sorted(curve_files):
                fold_num = curve_file.split('_fold')[1].split('_')[0]
                f.write(f"Fold {fold_num}:\n\n![Fold{fold_num}训练曲线]({curve_file})\n\n")
    
    def _config_to_dict(self) -> Dict[str, Any]:
        """将配置转为字典 (用于序列化)"""
        return {
            attr: getattr(self.config, attr)
            for attr in dir(self.config)
            if attr.isupper() and not attr.startswith('_')
        }


# ==================== 公共接口 ====================

def run_single_experiment(config: Optional[ExperimentConfig] = None,
                          tasks: Optional[List[str]] = None,
                          output_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    运行单次实验 (批量实验入口)
    
    这是为批量实验设计的主要接口函数。
    
    Args:
        config: 实验配置，默认使用 ExperimentConfig()
        tasks: 任务列表，默认 ['pd', 'stage', 'subtype']
        output_dir: 输出目录，默认自动生成
        
    Returns:
        实验结果字典
        
    使用示例
    --------
    单次实验:
        >>> result = run_single_experiment()
    
    自定义配置:
        >>> cfg = ExperimentConfig()
        >>> cfg.BACKBONE = 'resnet34'
        >>> cfg.NUM_EPOCHS = 100
        >>> result = run_single_experiment(cfg, tasks=['pd'])
    
    批量实验:
        >>> results = {}
        >>> for backbone in ['resnet18', 'resnet34', 'densenet121']:
        ...     cfg = ExperimentConfig()
        ...     cfg.BACKBONE = backbone
        ...     results[backbone] = run_single_experiment(cfg)
    """
    if config is None:
        config = ExperimentConfig()
    
    runner = ExperimentRunner(config)
    return runner.run(tasks=tasks, output_dir=output_dir)


# ==================== 命令行入口 ====================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="帕金森病图像分类实验")
    parser.add_argument('--backbone', default=None,
                        help='覆盖默认 backbone (resnet18/resnet34/densenet121/efficientnet)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='覆盖默认训练轮数')
    parser.add_argument('--tasks', nargs='+', default=None,
                        help='指定任务 (pd stage subtype)')
    parser.add_argument('--kfold-mode', choices=['patient', 'image'], default=None,
                        help='K-fold划分模式: patient(按患者，避免数据泄露) 或 image(按图像)')
    args = parser.parse_args()
    
    config = ExperimentConfig()
    
    # 应用命令行参数覆盖
    if args.backbone:
        config.BACKBONE = args.backbone
    if args.epochs:
        config.NUM_EPOCHS = args.epochs
    if args.kfold_mode:
        config.KFOLD_MODE = args.kfold_mode
    
    run_single_experiment(config, tasks=args.tasks)
