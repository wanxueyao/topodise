"""
main.py - GNN 实验主程序（使用 pd_dataset_unet_en 数据集）

模块职责
--------
执行单组配置下的完整 K-fold 交叉验证实验:
1. 加载 pd_dataset_unet_en 数据并生成 K-fold 划分
2. 根据 backbone 类型自动选择分类器（basic / light）
3. 逐 fold 训练和验证
4. 汇总指标 (Acc, F1, AUC)
5. 生成可视化 (训练曲线、混淆矩阵、ROC)
6. 输出实验报告 (Markdown 格式)

支持的 Backbone
---------------
- resnet18     → basic_classifier (无 dropout 参数，cosine scheduler)
- efficientnet_b0 → light_classifier (dropout 0.3, label_smooth, adamw)

数据集
------
默认使用 datasets/pd_dataset_unet_en/patients_dataset_en.json

接口设计
--------
输入接口:
    - GNNExperimentConfig: 可通过修改类属性或继承来自定义配置
    - run_single_experiment(config): 接受配置对象，返回结果字典

批量实验调用示例:
    >>> from main import GNNExperimentConfig, run_single_experiment
    >>> cfg = GNNExperimentConfig()
    >>> cfg.BACKBONE = 'resnet18'
    >>> cfg.NUM_FOLDS = 2
    >>> run_single_experiment(cfg, output_dir='results/test')
"""
import sys
import os
import time
import torch
import numpy as np
from datetime import datetime
from typing import Literal, List, Dict, Any, Optional

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.basic_classifier import PDClassifier, PDClassifierConfig, TASK_CONFIG
from core.light_classifier import LightClassifier, LightClassifierConfig
from core.data_loader import PDDataset, load_and_filter_dataset, stratified_kfold, KFoldMode
from utils.patient_outputs import save_fold_patient_outputs, task_labels_for_indices
from utils.misc import set_seed, Tee, create_output_dir, save_config
from utils.visualize import plot_confusion_matrix, plot_roc_auc, plot_training_curves


# ==================== Backbone 分组 ====================

# 使用 basic_classifier 的 backbone
BASIC_BACKBONES = {'resnet18', 'resnet34', 'densenet121', 'efficientnet'}

# 使用 light_classifier 的 backbone
LIGHT_BACKBONES = {'efficientnet_b0', 'mobilevit_xxs', 'mobilevit_xs', 'vanillanet_5'}


# ==================== 实验配置 ====================

class GNNExperimentConfig:
    """
    GNN 实验配置类（支持 basic / light 两类 backbone）

    通过修改类属性来配置实验参数，批量实验时创建多个实例。

    配置分组
    --------
    架构配置:
        ARCHITECTURE: 'single_stream' / 'mixed_single_stream' / 'dual_stream'
        BACKBONE: 'resnet18' / 'efficientnet_b0' 等
        PRETRAINED: True / False / 'Half'

    数据配置:
        DATASET_MODE: 'pd_control' / 'pd_control_cataract' / 'pd_all'
        DATASET_JSON: 数据集 JSON 文件路径（默认 pd_dataset_unet_en）

    训练配置:
        BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY

    学习率调度:
        LR_SCHEDULER: 'cosine' / 'step' / 'warmup_cosine' / 'none'
        LR_STEP_SIZE, LR_GAMMA
        BACKBONE_LR_RATIO: backbone 学习率倍率（仅 basic 模式）
        OPTIMIZER: 'adamw' / 'adam'（仅 light 模式）

    正则化（仅 light backbone 生效）:
        DROPOUT: 特征 Dropout

    损失函数（仅 light backbone 生效）:
        PD_LOSS_TYPE: 'ce' / 'label_smooth'
        STAGE_LOSS_TYPE: 'ce' / 'label_smooth' / 'ordinal'
        SUBTYPE_LOSS_TYPE: 'ce' / 'label_smooth'
        LABEL_SMOOTH_EPS: 标签平滑 epsilon

    早停:
        EARLY_STOPPING: bool
        PATIENCE: int

    数据增强:
        DATA_AUGMENTATION: bool
        AUG_HORIZONTAL_FLIP, AUG_ROTATION, AUG_BRIGHTNESS, AUG_CONTRAST

    交叉验证:
        NUM_FOLDS: K-fold 的 K 值（默认 2）
        SEED: 随机种子
        KFOLD_MODE: 'patient' / 'image'

    系统:
        NUM_WORKERS: DataLoader 工作进程数
        DEVICE: 'cuda' / 'cpu'
        OUTPUT_DIR: 输出根目录
    """
    # 架构
    ARCHITECTURE: Literal['single_stream', 'mixed_single_stream', 'dual_stream'] = 'dual_stream'
    BACKBONE: str = 'resnet18'
    PRETRAINED: Literal[True, False, 'Half'] = 'Half'

    # 数据（默认使用 unet 分割版数据集）
    DATASET_MODE: Literal['pd_control', 'pd_control_cataract', 'pd_all'] = 'pd_all'
    DATASET_JSON: str = 'datasets/pd_dataset_unet_en/patients_dataset_en.json'

    # 训练
    BATCH_SIZE: int = 8
    NUM_EPOCHS: int = 40
    LEARNING_RATE: float = 2e-4
    WEIGHT_DECAY: float = 3e-3

    # 学习率调度
    LR_SCHEDULER: str = 'cosine'     # basic 支持: step/cosine/none；light 支持: warmup_cosine/cosine/step
    LR_STEP_SIZE: int = 15
    LR_GAMMA: float = 0.5
    BACKBONE_LR_RATIO: float = 0.05  # basic 模式下 backbone 学习率倍率
    OPTIMIZER: str = 'adamw'         # light 模式下优化器

    # 早停
    EARLY_STOPPING: bool = True
    PATIENCE: int = 6

    # 正则化（light backbone 生效）
    DROPOUT: float = 0.3

    # 损失函数（light backbone 生效）
    PD_LOSS_TYPE: str = 'ce'
    STAGE_LOSS_TYPE: str = 'label_smooth'
    SUBTYPE_LOSS_TYPE: str = 'ce'
    LABEL_SMOOTH_EPS: float = 0.1

    # 数据增强
    DATA_AUGMENTATION: bool = True
    AUG_HORIZONTAL_FLIP: float = 0.5
    AUG_ROTATION: float = 6.0
    AUG_BRIGHTNESS: float = 0.12
    AUG_CONTRAST: float = 0.12
    AUG_SATURATION: float = 0.0
    AUG_HUE: float = 0.0

    # 交叉验证
    NUM_FOLDS: int = 2
    SEED: int = 42
    KFOLD_MODE: Literal['patient', 'image'] = 'patient'

    # 系统
    NUM_WORKERS: int = 4
    DEVICE: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    OUTPUT_DIR: str = 'results'
    REUSE_CHECKPOINTS: bool = True


# ==================== 分类器工厂 ====================

def _build_basic_classifier(cfg: GNNExperimentConfig) -> PDClassifier:
    """根据 GNNExperimentConfig 构建 basic_classifier"""
    basic_cfg = PDClassifierConfig()
    basic_cfg.ARCHITECTURE = cfg.ARCHITECTURE
    basic_cfg.BACKBONE = cfg.BACKBONE
    basic_cfg.PRETRAINED = cfg.PRETRAINED
    basic_cfg.BATCH_SIZE = cfg.BATCH_SIZE
    basic_cfg.NUM_EPOCHS = cfg.NUM_EPOCHS
    basic_cfg.LEARNING_RATE = cfg.LEARNING_RATE
    basic_cfg.WEIGHT_DECAY = cfg.WEIGHT_DECAY
    basic_cfg.LR_SCHEDULER = cfg.LR_SCHEDULER
    basic_cfg.LR_STEP_SIZE = cfg.LR_STEP_SIZE
    basic_cfg.LR_GAMMA = cfg.LR_GAMMA
    basic_cfg.BACKBONE_LR_RATIO = cfg.BACKBONE_LR_RATIO
    basic_cfg.EARLY_STOPPING = cfg.EARLY_STOPPING
    basic_cfg.PATIENCE = cfg.PATIENCE
    basic_cfg.DATA_AUGMENTATION = cfg.DATA_AUGMENTATION
    basic_cfg.AUG_HORIZONTAL_FLIP = cfg.AUG_HORIZONTAL_FLIP
    basic_cfg.AUG_ROTATION = cfg.AUG_ROTATION
    basic_cfg.AUG_BRIGHTNESS = cfg.AUG_BRIGHTNESS
    basic_cfg.AUG_CONTRAST = cfg.AUG_CONTRAST
    basic_cfg.AUG_SATURATION = cfg.AUG_SATURATION
    basic_cfg.AUG_HUE = cfg.AUG_HUE
    basic_cfg.NUM_WORKERS = cfg.NUM_WORKERS
    basic_cfg.DEVICE = cfg.DEVICE
    return PDClassifier(basic_cfg)


def _build_light_classifier(cfg: GNNExperimentConfig) -> LightClassifier:
    """根据 GNNExperimentConfig 构建 light_classifier"""
    light_cfg = LightClassifierConfig()
    light_cfg.ARCHITECTURE = cfg.ARCHITECTURE
    light_cfg.BACKBONE = cfg.BACKBONE
    light_cfg.PRETRAINED = cfg.PRETRAINED
    light_cfg.BATCH_SIZE = cfg.BATCH_SIZE
    light_cfg.NUM_EPOCHS = cfg.NUM_EPOCHS
    light_cfg.LEARNING_RATE = cfg.LEARNING_RATE
    light_cfg.WEIGHT_DECAY = cfg.WEIGHT_DECAY
    light_cfg.LR_SCHEDULER = cfg.LR_SCHEDULER
    light_cfg.LR_STEP_SIZE = cfg.LR_STEP_SIZE
    light_cfg.LR_GAMMA = cfg.LR_GAMMA
    light_cfg.OPTIMIZER = cfg.OPTIMIZER
    light_cfg.EARLY_STOPPING = cfg.EARLY_STOPPING
    light_cfg.PATIENCE = cfg.PATIENCE
    light_cfg.DROPOUT = cfg.DROPOUT
    light_cfg.PD_LOSS_TYPE = cfg.PD_LOSS_TYPE
    light_cfg.STAGE_LOSS_TYPE = cfg.STAGE_LOSS_TYPE
    light_cfg.SUBTYPE_LOSS_TYPE = cfg.SUBTYPE_LOSS_TYPE
    light_cfg.LABEL_SMOOTH_EPS = cfg.LABEL_SMOOTH_EPS
    light_cfg.DATA_AUGMENTATION = cfg.DATA_AUGMENTATION
    light_cfg.AUG_HORIZONTAL_FLIP = cfg.AUG_HORIZONTAL_FLIP
    light_cfg.AUG_ROTATION = cfg.AUG_ROTATION
    light_cfg.AUG_BRIGHTNESS = cfg.AUG_BRIGHTNESS
    light_cfg.AUG_CONTRAST = cfg.AUG_CONTRAST
    light_cfg.AUG_SATURATION = cfg.AUG_SATURATION
    light_cfg.AUG_HUE = cfg.AUG_HUE
    light_cfg.NUM_WORKERS = cfg.NUM_WORKERS
    light_cfg.DEVICE = cfg.DEVICE
    return LightClassifier(light_cfg)


# ==================== 实验运行器 ====================

class GNNExperimentRunner:
    """
    GNN 实验运行器（统一 basic / light 两类 backbone）

    根据 backbone 自动选择对应分类器，其余流程与 basic_experiments 保持一致。
    """

    def __init__(self, config: GNNExperimentConfig):
        self.config = config
        self.output_dir: Optional[str] = None

        if config.BACKBONE in LIGHT_BACKBONES:
            self.classifier = _build_light_classifier(config)
            self._mode = 'light'
        else:
            self.classifier = _build_basic_classifier(config)
            self._mode = 'basic'

    def run(self, tasks: Optional[List[str]] = None,
            output_dir: Optional[str] = None) -> Dict[str, Any]:
        """运行完整 K-fold 实验"""
        set_seed(self.config.SEED)
        start_time = time.time()

        if tasks is None:
            tasks = ['pd', 'stage', 'subtype']

        self.output_dir = output_dir or create_output_dir('results', self.config)
        os.makedirs(self.output_dir, exist_ok=True)
        self.config.OUTPUT_DIR = self.output_dir
        save_config(self.config, self.output_dir)

        log_file = open(os.path.join(self.output_dir, 'output.txt'), 'w', encoding='utf-8')
        original_stdout = sys.stdout
        sys.stdout = Tee(original_stdout, log_file)

        try:
            results = self._execute_experiment(tasks)
        finally:
            sys.stdout = original_stdout
            log_file.close()

        duration = time.time() - start_time
        results['meta'] = {
            'config': self._config_to_dict(),
            'duration': duration,
            'output_dir': self.output_dir,
        }

        print(f"\n实验完成！用时 {duration / 60:.2f} 分钟")
        print(f"结果保存到: {self.output_dir}")
        return results

    def _execute_experiment(self, tasks: List[str]) -> Dict[str, Any]:
        config = self.config
        self._print_header()

        patients = load_and_filter_dataset(config.DATASET_JSON, config.DATASET_MODE)
        print(f"\n加载 {len(patients)} 个患者（来自 {config.DATASET_JSON}）")

        results = {}

        if config.ARCHITECTURE == 'single_stream':
            for task in tasks:
                results[task] = {'regions': {}}
                for region in ['central', 'whorl']:
                    print(f"\n{'=' * 60}")
                    print(f"任务: {TASK_CONFIG[task]['name']} | 区域: {region}")
                    print('=' * 60)

                    dataset = PDDataset(
                        patients, 'single_stream',
                        self.classifier.train_transform,
                        region=region, base_path='datasets'
                    )
                    if len(dataset) == 0:
                        print(f"⚠ 区域 {region} 无数据，跳过")
                        continue

                    print(f"样本数: {len(dataset)}")
                    results[task]['regions'][region] = self._run_kfold(dataset, task, f"_{region}")
        else:
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
                print(f"\n{'=' * 60}")
                print(f"任务: {TASK_CONFIG[task]['name']}")
                print('=' * 60)
                results[task] = self._run_kfold(dataset, task)

        self._generate_report(results)
        return results

    def _run_kfold(self, dataset, task: str, suffix: str = '') -> Dict[str, Any]:
        config = self.config
        task_cfg = TASK_CONFIG[task]

        splits = list(stratified_kfold(
            dataset.PD_labels, dataset.stage_labels, dataset.subtype_labels,
            n_splits=config.NUM_FOLDS, random_state=config.SEED,
            kfold_mode=config.KFOLD_MODE, patient_ids=dataset.patient_ids
        ))

        print(f"K-fold 模式: {config.KFOLD_MODE} (按{'患者' if config.KFOLD_MODE == 'patient' else '图像'}划分), "
              f"K={config.NUM_FOLDS}")

        fold_results = []
        all_preds = {'labels': [], 'preds': [], 'probs': []}
        best_models = []

        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            print(f"\n--- Fold {fold_idx + 1}/{config.NUM_FOLDS} ---")

            train_subset = torch.utils.data.Subset(dataset, train_idx)
            val_subset = torch.utils.data.Subset(dataset, val_idx)

            safe_suffix = suffix.replace('/', '_')
            checkpoint_path = os.path.join(
                self.output_dir, f"best_model_{task}{safe_suffix}_fold{fold_idx + 1}.pth"
            )
            result = self.classifier.train(
                train_subset, val_subset, task,
                fold_idx=fold_idx + 1, suffix=suffix,
                checkpoint_path=checkpoint_path,
                reuse_checkpoint=config.REUSE_CHECKPOINTS,
            )

            plot_training_curves(
                result['train_history'],
                task_cfg['name'] + suffix,
                fold_idx + 1, self.output_dir
            )

            val_result = result['val_result']

            if val_result['labels']:
                patient_labels, patient_ids = task_labels_for_indices(dataset, val_idx, task)
                patient_result = save_fold_patient_outputs(
                    self.output_dir, task, suffix, fold_idx + 1,
                    patient_ids, patient_labels,
                    probs=val_result.get('probs'), logits=val_result.get('logits')
                )
                fold_results.append({
                    'fold': fold_idx + 1,
                    'train_acc': result['train_history']['train_acc'][-1] if 'train_acc' in result['train_history'] else 0.0,
                    'acc': val_result['acc'],
                    'patient_acc': patient_result['patient_acc'],
                    'n_patients': patient_result['n_patients'],
                    'patient_logits_json': patient_result['patient_logits_json'],
                    'patient_logits_csv': patient_result['patient_logits_csv'],
                    'precision': val_result['precision'],
                    'recall': val_result['recall'],
                    'f1': val_result['f1'],
                })
                all_preds['labels'].extend(val_result['labels'])
                all_preds['preds'].extend(val_result['preds'])
                if len(val_result['probs']) > 0:
                    all_preds['probs'].extend(val_result['probs'].tolist())

                print(f"  Acc: {val_result['acc']:.4f}, PatientAcc: {patient_result['patient_acc']:.4f}, F1: {val_result['f1']:.4f}")

            best_models.append({
                'fold': fold_idx + 1,
                'state': result['model_state'],
                'acc': val_result['acc'],
            })
            torch.save(result['model_state'], checkpoint_path)
            print(f"✓ 保存 Fold {fold_idx + 1} 模型: {checkpoint_path}")

        if not fold_results:
            return {}

        summary = self._summarize_results(fold_results, all_preds, task_cfg, suffix)

        best = max(best_models, key=lambda x: x['acc'])
        model_path = os.path.join(self.output_dir, f"best_model_{task}{suffix}.pth")
        torch.save(best['state'], model_path)
        print(f"✓ 保存最佳模型: {model_path}")

        return summary

    def _summarize_results(self, fold_results: List[Dict], all_preds: Dict,
                           task_cfg: Dict, suffix: str) -> Dict[str, Any]:
        task_name = task_cfg['name'] + suffix
        class_names = task_cfg['class_names']

        avg_train_acc = np.mean([r.get('train_acc', 0) for r in fold_results])
        avg_acc = np.mean([r['acc'] for r in fold_results])
        std_acc = np.std([r['acc'] for r in fold_results])
        avg_patient_acc = np.mean([r.get('patient_acc', 0.0) for r in fold_results])
        std_patient_acc = np.std([r.get('patient_acc', 0.0) for r in fold_results])
        avg_f1 = np.mean([r['f1'] for r in fold_results])
        std_f1 = np.std([r['f1'] for r in fold_results])

        print(f"\n📊 {task_name}: TrainAcc={avg_train_acc:.4f}, "
              f"ValAcc={avg_acc:.4f}±{std_acc:.4f}, "
              f"PatientAcc={avg_patient_acc:.4f}±{std_patient_acc:.4f}, "
              f"ValF1={avg_f1:.4f}±{std_f1:.4f}")

        auc_scores = {}
        if all_preds['labels']:
            safe_name = task_name.replace(' ', '_').replace('/', '_')
            unique_labels = sorted(set(all_preds['labels'] + all_preds['preds']))
            actual_names = [class_names[l] if l < len(class_names) else str(l) for l in unique_labels]

            plot_confusion_matrix(
                all_preds['labels'], all_preds['preds'],
                actual_names, safe_name, 0, self.output_dir, is_test=True
            )

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
                'auc_scores': auc_scores,
            },
            'all_predictions': all_preds,
        }

    def _print_header(self):
        cfg = self.config
        print("=" * 60)
        print("GNN 实验 - pd_dataset_unet_en")
        print("=" * 60)
        print(f"架构: {cfg.ARCHITECTURE} | Backbone: {cfg.BACKBONE} | 分类器模式: {self._mode}")
        print(f"预训练: {cfg.PRETRAINED} | 数据集: {cfg.DATASET_MODE}")
        print(f"Epochs: {cfg.NUM_EPOCHS} | K-fold: {cfg.NUM_FOLDS} | K-fold模式: {cfg.KFOLD_MODE}")
        print(f"LR: {cfg.LEARNING_RATE} | WD: {cfg.WEIGHT_DECAY} | Batch: {cfg.BATCH_SIZE}")
        if self._mode == 'light':
            print(f"Dropout: {cfg.DROPOUT} | Optimizer: {cfg.OPTIMIZER} | Stage Loss: {cfg.STAGE_LOSS_TYPE}")
        print(f"LR Scheduler: {cfg.LR_SCHEDULER} | Early Stopping: patience={cfg.PATIENCE}")
        if cfg.DATA_AUGMENTATION:
            print(f"数据增强: 开启 (翻转={cfg.AUG_HORIZONTAL_FLIP}, 旋转={cfg.AUG_ROTATION}°, "
                  f"亮度={cfg.AUG_BRIGHTNESS}, 对比度={cfg.AUG_CONTRAST})")
        else:
            print("数据增强: 关闭")
        print("=" * 60)

    def _generate_report(self, results: Dict):
        report_path = os.path.join(self.output_dir, 'experiment_report.md')
        cfg = self.config

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 实验报告（GNN / pd_dataset_unet_en）\n\n")
            f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**数据集**: `{cfg.DATASET_JSON}`\n\n")

            f.write("## 实验配置\n\n")
            f.write("| 参数 | 值 |\n|---|---|\n")
            f.write(f"| 架构 | `{cfg.ARCHITECTURE}` |\n")
            f.write(f"| Backbone | `{cfg.BACKBONE}` |\n")
            f.write(f"| 分类器模式 | `{self._mode}` |\n")
            f.write(f"| 预训练 | `{cfg.PRETRAINED}` |\n")
            f.write(f"| 数据集 | `{cfg.DATASET_MODE}` |\n")
            f.write(f"| K-fold | {cfg.NUM_FOLDS} |\n")
            f.write(f"| K-fold模式 | `{cfg.KFOLD_MODE}` |\n")
            f.write(f"| Epochs | {cfg.NUM_EPOCHS} |\n")
            f.write(f"| 学习率 | {cfg.LEARNING_RATE} |\n")
            f.write(f"| Weight Decay | {cfg.WEIGHT_DECAY} |\n")
            f.write(f"| Batch Size | {cfg.BATCH_SIZE} |\n")
            f.write(f"| LR Scheduler | `{cfg.LR_SCHEDULER}` |\n")
            f.write(f"| Early Stopping patience | {cfg.PATIENCE} |\n")
            if self._mode == 'light':
                f.write(f"| Dropout | {cfg.DROPOUT} |\n")
                f.write(f"| Optimizer | `{cfg.OPTIMIZER}` |\n")
                f.write(f"| Stage Loss | `{cfg.STAGE_LOSS_TYPE}` |\n")
            f.write(f"| 数据增强 | `{'开启' if cfg.DATA_AUGMENTATION else '关闭'}` |\n")
            if cfg.DATA_AUGMENTATION:
                f.write(f"| - 水平翻转 | {cfg.AUG_HORIZONTAL_FLIP} |\n")
                f.write(f"| - 旋转 | ±{cfg.AUG_ROTATION}° |\n")
                f.write(f"| - 亮度抖动 | {cfg.AUG_BRIGHTNESS} |\n")
                f.write(f"| - 对比度抖动 | {cfg.AUG_CONTRAST} |\n")
            f.write("\n")

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

        print(f"✓ 生成报告: {report_path}")

    def _write_task_section(self, f, task_name: str, region: Optional[str], result: Dict):
        suffix = f"_{region}" if region else ""
        display_name = f"{task_name}{suffix}" if region else task_name

        f.write(f"### {display_name}\n\n")

        if 'final_results' not in result:
            f.write("*无有效结果*\n\n")
            return

        fr = result['final_results']
        f.write("| 指标 | 值 |\n|---|---|\n")
        f.write(f"| Accuracy | {fr['avg_acc']:.4f} ± {fr['std_acc']:.4f} |\n")
        f.write(f"| F1 Score | {fr['avg_f1']:.4f} ± {fr['std_f1']:.4f} |\n")
        f.write("\n")

        if fr.get('auc_scores'):
            f.write("**AUC**:\n")
            for cls, auc_val in fr['auc_scores'].items():
                f.write(f"- {cls}: {auc_val:.4f}\n")
            f.write("\n")

        safe_name = (task_name + suffix).replace(' ', '_').replace('/', '_')
        cm_file = f"{safe_name}_Test_cm.png"
        roc_file = f"{safe_name}_Test_roc.png"

        if os.path.exists(os.path.join(self.output_dir, cm_file)):
            f.write(f"**混淆矩阵**:\n\n![混淆矩阵]({cm_file})\n\n")
        if os.path.exists(os.path.join(self.output_dir, roc_file)):
            f.write(f"**ROC 曲线**:\n\n![ROC曲线]({roc_file})\n\n")

        curve_files = [fn for fn in os.listdir(self.output_dir)
                       if fn.startswith(f"{task_name}{suffix}_fold") and fn.endswith("_curves.png")]
        if curve_files:
            f.write("**训练曲线**:\n\n")
            for curve_file in sorted(curve_files):
                fold_num = curve_file.split('_fold')[1].split('_')[0]
                f.write(f"Fold {fold_num}:\n\n![Fold{fold_num}训练曲线]({curve_file})\n\n")

    def _config_to_dict(self) -> Dict[str, Any]:
        return {
            attr: getattr(self.config, attr)
            for attr in dir(self.config)
            if attr.isupper() and not attr.startswith('_')
        }


# ==================== 公共接口 ====================

def run_single_experiment(config: Optional[GNNExperimentConfig] = None,
                          tasks: Optional[List[str]] = None,
                          output_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    运行单次实验（批量实验入口）

    Args:
        config: 实验配置，默认使用 GNNExperimentConfig()
        tasks: 任务列表，默认 ['pd', 'stage', 'subtype']
        output_dir: 输出目录，默认自动生成

    Returns:
        实验结果字典
    """
    if config is None:
        config = GNNExperimentConfig()

    runner = GNNExperimentRunner(config)
    return runner.run(tasks=tasks, output_dir=output_dir)


# ==================== 命令行入口 ====================

if __name__ == '__main__':
    cfg = GNNExperimentConfig()
    run_single_experiment(cfg)
