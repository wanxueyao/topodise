"""U-Net-only/dual 实验实现：患者划分、训练、评估及亚型层级合成。"""
from __future__ import annotations
if __name__ == '__main__':
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from experiments._execution import single_main
    single_main('experiments.unet_experiments.main')
    raise SystemExit(0)

from utils.td_rest import resolve_baseline_tasks, td_rest_rows, td_rest_fold, save_td_rest_results
import sys
import os
import time
import torch
import numpy as np
from datetime import datetime
from typing import Literal, List, Dict, Any, Optional
from core.basic_classifier import PDClassifier, PDClassifierConfig, TASK_CONFIG
from core.data_loader import PDDataset, load_and_filter_dataset, stratified_kfold
from utils.patient_outputs import save_fold_patient_outputs, task_labels_for_indices
from utils.misc import set_seed, Tee, create_output_dir, save_config
from utils.visualize import plot_confusion_matrix, plot_roc_auc, plot_training_curves
import json
from dataclasses import dataclass, asdict
from core.data_loader import SUBTYPE_MAP
from unet.dual_view_classifier import DualViewConfig, DualViewClassifier
from unet.dual_view_model import DUAL_VIEW_TASK_CONFIG
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Tuple
from torch.utils.data import DataLoader
from utils.subtype_hierarchy import (
    SUBTYPE_TO_3CLS,
    build_patient_5x3_subtype_folds,
)
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from unet.dual_view_classifier import _build_transforms
from unet.dual_view_dataset import DualViewDataset
from unet.dual_view_model import DualViewModel
from utils.subtype_hierarchy import (
    subtype_pigd_mixed_label,
    subtype_tdrest_label,
)

CODES_ROOT = Path(__file__).resolve().parents[2]

class UNetExperimentConfig:
    """U-Net-only classification configuration."""
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
    LR_SCHEDULER: str = 'cosine'
    LR_STEP_SIZE: int = 15
    LR_GAMMA: float = 0.5
    BACKBONE_LR_RATIO: float = 0.05  # basic 模式下 backbone 学习率倍率
    OPTIMIZER: str = 'adamw'

    # 早停
    EARLY_STOPPING: bool = True
    PATIENCE: int = 6

    DROPOUT: float = 0.3

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


def _only_build_basic_classifier(cfg: UNetExperimentConfig) -> PDClassifier:
    """根据 UNetExperimentConfig 构建 basic_classifier"""
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


class UNetExperimentRunner:
    """Run patient-level U-Net-only classification and export metrics."""

    def __init__(self, config: UNetExperimentConfig):
        self.config = config
        self.output_dir: Optional[str] = None

        self.classifier = _only_build_basic_classifier(config)
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
        print("U-Net-only 实验 - pd_dataset_unet_en")
        print("=" * 60)
        print(f"架构: {cfg.ARCHITECTURE} | Backbone: {cfg.BACKBONE} | 分类器模式: {self._mode}")
        print(f"预训练: {cfg.PRETRAINED} | 数据集: {cfg.DATASET_MODE}")
        print(f"Epochs: {cfg.NUM_EPOCHS} | K-fold: {cfg.NUM_FOLDS} | K-fold模式: {cfg.KFOLD_MODE}")
        print(f"LR: {cfg.LEARNING_RATE} | WD: {cfg.WEIGHT_DECAY} | Batch: {cfg.BATCH_SIZE}")
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
            f.write("# 实验报告（U-Net-only / pd_dataset_unet_en）\n\n")
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


def only_run_single_experiment(config: Optional[UNetExperimentConfig] = None,
                          tasks: Optional[List[str]] = None,
                          output_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    运行单次实验（批量实验入口）

    Args:
        config: 实验配置，默认使用 UNetExperimentConfig()
        tasks: 任务列表，默认 ['pd', 'stage', 'subtype']
        output_dir: 输出目录，默认自动生成

    Returns:
        实验结果字典
    """
    if config is None:
        config = UNetExperimentConfig()

    runner = UNetExperimentRunner(config)
    return runner.run(tasks=tasks, output_dir=output_dir)


def only_build_paper_config(exp, device="cpu", num_workers=0):
    """Translate a selected paper recipe without changing its hyperparameters."""
    cfg = UNetExperimentConfig()
    cfg.BACKBONE = exp.get('backbone', 'resnet18')
    cfg.ARCHITECTURE = exp.get('architecture', 'dual_stream')
    cfg.PRETRAINED = exp.get('pretrained', 'Half')
    cfg.BATCH_SIZE = exp.get('batch_size', cfg.BATCH_SIZE)
    cfg.NUM_EPOCHS = exp.get('epochs', cfg.NUM_EPOCHS)
    cfg.LEARNING_RATE = exp.get('learning_rate', cfg.LEARNING_RATE)
    cfg.WEIGHT_DECAY = exp.get('weight_decay', cfg.WEIGHT_DECAY)
    cfg.LR_SCHEDULER = exp.get('lr_scheduler', cfg.LR_SCHEDULER)
    cfg.LR_STEP_SIZE = exp.get('lr_step_size', cfg.LR_STEP_SIZE)
    cfg.LR_GAMMA = exp.get('lr_gamma', cfg.LR_GAMMA)
    cfg.BACKBONE_LR_RATIO = exp.get('backbone_lr_ratio', cfg.BACKBONE_LR_RATIO)
    cfg.OPTIMIZER = exp.get('optimizer', cfg.OPTIMIZER)
    cfg.EARLY_STOPPING = exp.get('early_stopping', cfg.EARLY_STOPPING)
    cfg.PATIENCE = exp.get('patience', cfg.PATIENCE)
    cfg.DROPOUT = exp.get('dropout', cfg.DROPOUT)
    cfg.PD_LOSS_TYPE = exp.get('pd_loss_type', cfg.PD_LOSS_TYPE)
    cfg.STAGE_LOSS_TYPE = exp.get('stage_loss_type', cfg.STAGE_LOSS_TYPE)
    cfg.SUBTYPE_LOSS_TYPE = exp.get('subtype_loss_type', cfg.SUBTYPE_LOSS_TYPE)
    cfg.LABEL_SMOOTH_EPS = exp.get('label_smooth_eps', cfg.LABEL_SMOOTH_EPS)
    cfg.DATASET_MODE = exp.get('dataset_mode', 'pd_all')
    cfg.NUM_FOLDS = exp.get('num_folds', 2)
    cfg.SEED = exp.get('seed', cfg.SEED)
    cfg.KFOLD_MODE = exp.get('kfold_mode', 'patient')
    cfg.DATA_AUGMENTATION = exp.get('data_augmentation', True)
    cfg.AUG_HORIZONTAL_FLIP = exp.get('aug_horizontal_flip', cfg.AUG_HORIZONTAL_FLIP)
    cfg.AUG_ROTATION = exp.get('aug_rotation', cfg.AUG_ROTATION)
    cfg.AUG_BRIGHTNESS = exp.get('aug_brightness', cfg.AUG_BRIGHTNESS)
    cfg.AUG_CONTRAST = exp.get('aug_contrast', cfg.AUG_CONTRAST)
    cfg.AUG_SATURATION = exp.get('aug_saturation', cfg.AUG_SATURATION)
    cfg.AUG_HUE = exp.get('aug_hue', cfg.AUG_HUE)
    cfg.NUM_WORKERS = num_workers
    cfg.DEVICE = device
    return cfg


@dataclass
class DualStreamExperimentConfig:
    """双视角 CNN 实验配置"""
    
    # 实验标识
    name: str = 'dual_stream_experiment'
    
    # 数据集
    en_dataset_json: str = 'datasets/pd_dataset_en/patients_dataset_en.json'
    unet_dataset_json: str = 'datasets/pd_dataset_unet_en/patients_dataset_en.json'
    dataset_group: str = 'pd_all'
    num_folds: int = 2
    kfold_mode: str = 'patient'
    region: str = 'both'
    
    # 模型
    backbone: str = 'resnet18'
    pretrained: object = 'Half'
    shared_backbone: bool = True
    
    # 融合
    fusion_dim: int = 256
    dropout: float = 0.3
    classifier_dropout: float = 0.3
    
    # 训练超参数
    learning_rate: float = 2e-4
    backbone_lr_ratio: float = 0.1
    weight_decay: float = 3e-3
    optimizer: str = 'adamw'
    lr_scheduler: str = 'cosine'
    warmup_epochs: int = 5
    epochs: int = 50
    batch_size: int = 8
    patience: int = 10
    
    # 损失
    label_smoothing: float = 0.0
    
    # 数据增强
    data_augmentation: bool = True
    aug_horizontal_flip: float = 0.5
    aug_vertical_flip: float = 0.0
    aug_rotation: float = 6.0
    aug_brightness: float = 0.12
    aug_contrast: float = 0.12
    
    # 其他
    num_workers: int = 0
    seed: int = 42
    reuse_checkpoints: bool = True
    tasks: Optional[List[str]] = None


class DualStreamExperimentRunner:
    """双视角 CNN 实验执行器"""
    
    def __init__(self, config: DualStreamExperimentConfig):
        self.config = config
        self.output_dir = None
        self.fold_results: Dict[str, List] = {'pd': [], 'stage': [], 'subtype': []}
        self.common_patients = []
        self.unet_patients = []
    
    def run(self, output_dir: str) -> Dict[str, Any]:
        """执行完整 K-fold 实验"""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)

        log_file = open(os.path.join(output_dir, 'output.txt'), 'w', encoding='utf-8')
        original_stdout = sys.stdout
        sys.stdout = Tee(original_stdout, log_file)

        try:
            start_time = time.time()
            print('=' * 70)
            print('双视角 CNN 实验 - pd_dataset_en + pd_dataset_unet_en')
            print('=' * 70)
            self._print_header()

            # 加载数据
            print('\n加载数据集...')
            self._load_data()

            # 执行 K-fold
            print(f'\n开始 {self.config.num_folds}-fold 交叉验证...')
            self._run_kfold()

            # 汇总与保存
            print('\n汇总结果...')
            summary = self._summarize()
            self._save_results(summary)

            # 生成报告
            print('\n生成报告...')
            self._generate_report(summary)

            duration = time.time() - start_time
            print(f'\n实验完成！总耗时 {duration/60:.1f} 分钟')
            print(f'结果保存到: {output_dir}')
            print('=' * 70)

            return {
                'status': 'success',
                'config': asdict(self.config),
                'summary': summary,
                'output_dir': output_dir,
                'duration_minutes': duration / 60.0,
            }

        except Exception as e:
            print(f'\n实验失败: {e}')
            import traceback
            print(traceback.format_exc())
            return {
                'status': 'failed',
                'error': str(e),
                'output_dir': output_dir,
            }

        finally:
            sys.stdout = original_stdout
            log_file.close()
    
    def _print_header(self):
        """打印实验头信息"""
        cfg = self.config
        print(f"架构: dual_stream | Backbone: {cfg.backbone} | 预训练: {cfg.pretrained}")
        print(f"数据集: {cfg.dataset_group} | K-fold: {cfg.num_folds} ({cfg.kfold_mode})")
        print(f"Epochs: {cfg.epochs} | LR: {cfg.learning_rate} | WD: {cfg.weight_decay}")
        print(f"Batch: {cfg.batch_size} | Early Stopping: patience={cfg.patience}")

        if cfg.data_augmentation:
            print(f"数据增强: 开启 (翻转={cfg.aug_horizontal_flip}, 旋转={cfg.aug_rotation}°, "
                  f"亮度={cfg.aug_brightness}, 对比度={cfg.aug_contrast})")
        else:
            print("数据增强: 关闭")
    
    def _load_data(self):
        """加载两个视角的数据集"""
        cfg = self.config

        # 加载原始眼底图
        en_json_path = os.path.join(CODES_ROOT, cfg.en_dataset_json)
        en_patients = sorted(
            load_and_filter_dataset(en_json_path, cfg.dataset_group),
            key=lambda p: p['patient_id'],
        )

        # 加载 UNet 处理后的眼底图
        unet_json_path = os.path.join(CODES_ROOT, cfg.unet_dataset_json)
        unet_all = load_and_filter_dataset(unet_json_path, cfg.dataset_group)
        unet_map = {p['patient_id']: p for p in unet_all}

        # 按患者 ID 匹配
        self.common_patients = [p for p in en_patients if p['patient_id'] in unet_map]
        self.unet_patients = [unet_map[p['patient_id']] for p in self.common_patients]

        print(f"  EN 数据集: {len(en_patients)} 患者")
        print(f"  UNet 数据集: {len(unet_all)} 患者")
        print(f"  共同患者: {len(self.common_patients)}")
    
    def _run_kfold(self):
        """执行 K-fold 交叉验证（按任务逐个独立训练，各自使用不同的模型）"""
        cfg = self.config

        # 准备标签
        pd_labels = [1 if p['group'] == 'pd' else 0 for p in self.common_patients]
        stage_labels = [int(p.get('stage', 0)) for p in self.common_patients]
        subtype_labels = [SUBTYPE_MAP.get(p.get('subtype') or '', -1) for p in self.common_patients]
        patient_ids = list(range(len(self.common_patients)))

        # 生成 K-fold 划分（各任务共用同一组划分）
        folds = list(stratified_kfold(
            pd_labels=pd_labels,
            stage_labels=stage_labels,
            subtype_labels=subtype_labels,
            n_splits=cfg.num_folds,
            kfold_mode=cfg.kfold_mode,
            patient_ids=patient_ids,
        ))

        base_path = os.path.join(CODES_ROOT, 'datasets')

        # 逐任务训练（与 basic_experiments 一致）
        for task in (cfg.tasks or ['pd', 'stage', 'subtype']):
            task_name = DUAL_VIEW_TASK_CONFIG[task]['name']
            print(f"\n{'='*60}")
            print(f"任务: {task_name}")
            print('='*60)
            print(f"K-fold 模式: {cfg.kfold_mode} (按了患者划分" if cfg.kfold_mode == 'patient' else f"K-fold 模式: {cfg.kfold_mode}", end=")\n")

            clf_cfg = DualViewConfig(
                task=task,
                backbone=cfg.backbone,
                pretrained=cfg.pretrained,
                shared_backbone=cfg.shared_backbone,
                fusion_dim=cfg.fusion_dim,
                dropout=cfg.dropout,
                classifier_dropout=cfg.classifier_dropout,
                lr=cfg.learning_rate,
                backbone_lr_ratio=cfg.backbone_lr_ratio,
                weight_decay=cfg.weight_decay,
                optimizer=cfg.optimizer,
                lr_scheduler=cfg.lr_scheduler,
                warmup_epochs=cfg.warmup_epochs,
                epochs=cfg.epochs,
                patience=cfg.patience,
                batch_size=cfg.batch_size,
                region=cfg.region,
                data_augmentation=cfg.data_augmentation,
                aug_horizontal_flip=cfg.aug_horizontal_flip,
                aug_rotation=cfg.aug_rotation,
                aug_brightness=cfg.aug_brightness,
                aug_contrast=cfg.aug_contrast,
                label_smoothing=cfg.label_smoothing,
                num_workers=cfg.num_workers,
                seed=cfg.seed,
            )

            for fold_idx, (train_idx, val_idx) in enumerate(folds):
                fold_num = fold_idx + 1
                print(f'\n--- Fold {fold_num}/{cfg.num_folds} ---')
                print(f'  Train: {len(train_idx)} | Val: {len(val_idx)}')

                train_en = [self.common_patients[i] for i in train_idx]
                val_en = [self.common_patients[i] for i in val_idx]
                train_unet = [self.unet_patients[i] for i in train_idx]
                val_unet = [self.unet_patients[i] for i in val_idx]

                clf = DualViewClassifier(config=clf_cfg)
                checkpoint_path = os.path.join(self.output_dir, f'{task}_fold_{fold_num}_best_model.pth')
                result = clf.train_one_fold(
                    train_en, val_en, train_unet, val_unet,
                    base_path, task=task, fold_idx=fold_num,
                    checkpoint_path=checkpoint_path,
                    reuse_checkpoint=cfg.reuse_checkpoints,
                )
                patient_result = save_fold_patient_outputs(
                    self.output_dir, task, '', fold_num,
                    result.get('patient_ids', []),
                    result.get('labels', []),
                    probs=result.get('probs'), logits=result.get('logits'),
                )
                result['patient_acc'] = patient_result['patient_acc']
                result['n_patients'] = patient_result['n_patients']
                result['patient_logits_json'] = patient_result['patient_logits_json']
                result['patient_logits_csv'] = patient_result['patient_logits_csv']

                if 'train_history' in result:
                    plot_training_curves(
                        result['train_history'],
                        task_name,
                        fold_num,
                        self.output_dir,
                    )

                self.fold_results[task].append(result)
                print(
                    f"  Acc: {result.get('acc', 0):.4f}, "
                    f"PatientAcc: {result.get('patient_acc', 0):.4f}, "
                    f"Bacc: {result.get('balanced_acc', 0):.4f}, "
                    f"F1: {result.get('f1', 0):.4f}, "
                    f"AUC: {result.get('auc', 0):.4f}"
                )
    
    def _summarize(self) -> Dict[str, Any]:
        """汇总多 fold 结果（按任务）"""
        summary = {}
        task_display = {'pd': 'PD', 'stage': 'Stage', 'subtype': 'Subtype'}

        for task, results in self.fold_results.items():
            if not results:
                continue

            train_accs  = [r.get('train_acc', 0.0) for r in results]
            val_accs    = [r.get('acc', 0.0) for r in results]
            patient_accs = [r.get('patient_acc', 0.0) for r in results]
            val_baccs   = [r.get('balanced_acc', 0.0) for r in results]
            val_f1s     = [r.get('f1', 0.0) for r in results]
            val_aucs    = [r.get('auc', 0.5) for r in results]

            train_mean = float(np.mean(train_accs))
            val_mean   = float(np.mean(val_accs))
            val_std    = float(np.std(val_accs))
            patient_mean = float(np.mean(patient_accs))
            patient_std = float(np.std(patient_accs))
            f1_mean    = float(np.mean(val_f1s))
            f1_std     = float(np.std(val_f1s))
            gap        = train_mean - val_mean

            summary[task] = {
                'train_acc_mean':  train_mean,
                'val_acc_mean':    val_mean,
                'val_acc_std':     val_std,
                'patient_acc_mean': patient_mean,
                'patient_acc_std':  patient_std,
                'val_bacc_mean':   float(np.mean(val_baccs)),
                'val_bacc_std':    float(np.std(val_baccs)),
                'val_f1_mean':     f1_mean,
                'val_f1_std':      f1_std,
                'val_auc_mean':    float(np.mean(val_aucs)),
                'acc_gap':         gap,
            }

            print(
                f"📊 {task_display[task]}: "
                f"TrainAcc={train_mean:.4f}, "
                f"ValAcc={val_mean:.4f}±{val_std:.4f}, "
                f"PatientAcc={patient_mean:.4f}±{patient_std:.4f}, "
                f"ValF1={f1_mean:.4f}±{f1_std:.4f}, "
                f"Gap={gap:.4f}",
                flush=True,
            )

        return summary
    
    def _save_results(self, summary: Dict):
        """保存结果 JSON 和 pth 模型"""
        summary_path = os.path.join(self.output_dir, 'summary.json')
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump({
                'config': asdict(self.config),
                'summary': summary,
                'fold_count': self.config.num_folds,
                'timestamp': datetime.now().isoformat(),
            }, f, indent=2, ensure_ascii=False)
        print(f"✓ 保存汇总: {summary_path}")

        # 每任务每 fold 存实一个 pth
        for task, results in self.fold_results.items():
            for i, result in enumerate(results, 1):
                if 'model_state' in result:
                    pth = os.path.join(self.output_dir, f'{task}_fold_{i}_best_model.pth')
                    torch.save(result['model_state'], pth)
                    print(f"✓ 保存 {task} Fold {i} 模型: {pth}")

        folds_path = os.path.join(self.output_dir, 'fold_results.json')
        _skip = {'model_state', 'train_history', 'preds', 'labels', 'probs', 'logits', 'patient_ids'}
        folds_data = {}
        for task, results in self.fold_results.items():
            folds_data[task] = [
                {k: (v.tolist() if hasattr(v, 'tolist') else v)
                 for k, v in r.items() if k not in _skip}
                for r in results
            ]
        with open(folds_path, 'w', encoding='utf-8') as f:
            json.dump(folds_data, f, indent=2, ensure_ascii=False)
        print(f"✓ 保存 fold 结果: {folds_path}")
    
    def _generate_report(self, summary: Dict):
        """生成 Markdown 报告"""
        cfg = self.config
        report_path = os.path.join(self.output_dir, 'experiment_report.md')

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 双视角 CNN 实验报告\n\n")
            f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**数据集**: {cfg.dataset_group} (共 {len(self.common_patients)} 患者)\n\n")

            f.write("## 实验配置\n\n")
            f.write("| 参数 | 值 |\n|---|---|\n")
            f.write("| 架构 | dual_view |\n")
            f.write(f"| Backbone | {cfg.backbone} |\n")
            f.write(f"| 预训练 | {cfg.pretrained} |\n")
            f.write(f"| 共享骨干 | {cfg.shared_backbone} |\n")
            f.write(f"| 融合维度 | {cfg.fusion_dim} |\n")
            f.write(f"| Epochs | {cfg.epochs} |\n")
            f.write(f"| 学习率 | {cfg.learning_rate} |\n")
            f.write(f"| Weight Decay | {cfg.weight_decay} |\n")
            f.write(f"| Batch Size | {cfg.batch_size} |\n")
            f.write(f"| LR Scheduler | {cfg.lr_scheduler} |\n")
            f.write(f"| Early Stopping Patience | {cfg.patience} |\n")
            f.write(f"| K-fold | {cfg.num_folds} ({cfg.kfold_mode}) |\n")

            f.write("\n## 结果摘要\n\n")
            f.write("| 任务 | TrainAcc | ValAcc | PatientAcc | ValF1 | ValAUC | Gap |\n")
            f.write("|---|---|---|---|---|---|---|\n")
            task_display = {'pd': 'PD_Classification', 'stage': 'Stage_Classification',
                            'subtype': 'Subtype_Classification'}
            for task, ts in summary.items():
                f.write(
                    f"| {task_display.get(task, task)} "
                    f"| {ts.get('train_acc_mean', 0):.4f} "
                    f"| {ts.get('val_acc_mean', 0):.4f}±{ts.get('val_acc_std', 0):.4f} "
                    f"| {ts.get('patient_acc_mean', 0):.4f}±{ts.get('patient_acc_std', 0):.4f} "
                    f"| {ts.get('val_f1_mean', 0):.4f}±{ts.get('val_f1_std', 0):.4f} "
                    f"| {ts.get('val_auc_mean', 0):.4f} "
                    f"| {ts.get('acc_gap', 0):.4f} |\n"
                )

            f.write("\n## Fold 详情\n\n")
            for task, results in self.fold_results.items():
                f.write(f"### {task_display.get(task, task)}\n\n")
                for i, result in enumerate(results, 1):
                    f.write(f"**Fold {i}**: "
                            f"Acc={result.get('acc', 0):.4f}, "
                            f"PatientAcc={result.get('patient_acc', 0):.4f}, "
                            f"Bacc={result.get('balanced_acc', 0):.4f}, "
                            f"F1={result.get('f1', 0):.4f}, "
                            f"AUC={result.get('auc', 0):.4f}\n\n")

        print(f"✓ 生成报告: {report_path}")


def dual_run_single_experiment(config: DualStreamExperimentConfig, 
                         output_dir: str) -> Dict[str, Any]:
    """
    运行单次双视角实验
    
    Args:
        config: DualStreamExperimentConfig 配置
        output_dir: 输出目录
    
    Returns:
        结果字典
    """
    runner = DualStreamExperimentRunner(config)
    return runner.run(output_dir)


def dual_build_paper_config(exp, device="cpu", num_workers=0):
    """Translate a selected paper recipe without changing its hyperparameters."""
    cfg = DualStreamExperimentConfig()
    for (key, val) in exp.items():
        if key.startswith('_'):
            continue
        if hasattr(cfg, key):
            setattr(cfg, key, val)
    cfg.num_workers = num_workers
    return cfg


def _only_subtype_resolve_dataset_json(path: str) -> str:
    if os.path.isabs(path):
        return path
    return str(CODES_ROOT / path)


def _only_subtype_build_classifier(config: UNetExperimentConfig):
    return _only_build_basic_classifier(config)


def _only_subtype_patient_records(patients: List[Dict]) -> List[Dict]:
    return [
        {'patient_id': p['patient_id'], 'subtype': p['subtype']}
        for p in patients
        if p.get('group') == 'pd' and p.get('subtype') in SUBTYPE_TO_3CLS
    ]


def _only_subtype_subset_patients(pids: List[str], pid_to_patient: Dict[str, Dict]) -> List[Dict]:
    return [pid_to_patient[pid] for pid in pids]


def _only_subtype_make_dataset(
    classifier,
    patients: List[Dict],
    config: UNetExperimentConfig,
    training: bool = False,
) -> PDDataset:
    return PDDataset(
        patients,
        config.ARCHITECTURE,
        classifier.train_transform if training else classifier.val_transform,
        base_path=str(CODES_ROOT / 'datasets'),
    )


def _only_subtype_train_one_subtask(
    config: UNetExperimentConfig,
    train_patients: List[Dict],
    val_patients: List[Dict],
    task: str,
    fold_idx: int,
):
    classifier = _only_subtype_build_classifier(config)
    train_ds = _only_subtype_make_dataset(classifier, train_patients, config, training=True)
    val_ds = _only_subtype_make_dataset(classifier, val_patients, config, training=False)
    result = classifier.train(
        train_ds,
        val_ds,
        task,
        fold_idx=fold_idx,
        suffix='_subtype_td_rest',
    )
    return classifier, result['model']


@torch.no_grad()
def _only_subtype_infer_patient_level(
    model: torch.nn.Module,
    classifier,
    patients: List[Dict],
    config: UNetExperimentConfig,
) -> Tuple[Dict[str, int], Dict[str, float]]:
    dataset = _only_subtype_make_dataset(classifier, patients, config, training=False)
    local_pid_to_real = {
        local_idx: patient['patient_id']
        for local_idx, patient in enumerate(patients)
    }
    sample_pids = [local_pid_to_real[int(pid)] for pid in dataset.patient_ids]
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=(config.DEVICE != 'cpu'),
    )

    model.eval()
    sums: Dict[str, np.ndarray] = defaultdict(lambda: np.zeros(2, dtype=np.float64))
    counts: Dict[str, int] = defaultdict(int)
    offset = 0

    for batch in loader:
        is_dual = isinstance(batch[0], (tuple, list))
        if is_dual:
            (central, whorl), _pd, _stage, subtype = batch
            outputs = model(central.to(config.DEVICE), whorl.to(config.DEVICE))
            batch_size = int(subtype.size(0))
        else:
            images, _pd, _stage, subtype = batch
            outputs = model(images.to(config.DEVICE))
            batch_size = int(subtype.size(0))

        probs = torch.softmax(outputs, dim=1).detach().cpu().numpy()
        for i, pid in enumerate(sample_pids[offset:offset + batch_size]):
            sums[pid] += probs[i]
            counts[pid] += 1
        offset += batch_size

    pred_by_pid: Dict[str, int] = {}
    prob_class1_by_pid: Dict[str, float] = {}
    for pid, total_prob in sums.items():
        mean_prob = total_prob / max(1, counts[pid])
        pred_by_pid[pid] = int(np.argmax(mean_prob))
        prob_class1_by_pid[pid] = float(mean_prob[1])
    return pred_by_pid, prob_class1_by_pid


def only_subtype_run_unet_subtype_td_rest_experiment(
    config: UNetExperimentConfig,
    output_dir: str,
    seed: int = 42,
    n_splits: int = 5,
    repeats: int = 3,
) -> Dict:
    """运行 CNN-on-UNet subtype TD-rest 二分类 5-fold x 3 重算实验。"""
    os.makedirs(output_dir, exist_ok=True)
    set_seed(seed)

    cfg = deepcopy(config)
    cfg.DATASET_MODE = 'pd_all'
    cfg.KFOLD_MODE = 'patient'
    cfg.NUM_FOLDS = n_splits
    cfg.SEED = seed

    dataset_json = _only_subtype_resolve_dataset_json(cfg.DATASET_JSON)
    all_patients = load_and_filter_dataset(dataset_json, cfg.DATASET_MODE)
    records = _only_subtype_patient_records(all_patients)
    pid_to_patient = {
        p['patient_id']: p for p in all_patients
        if p.get('group') == 'pd' and p.get('subtype') in SUBTYPE_TO_3CLS
    }
    folds = build_patient_5x3_subtype_folds(
        records, seed=seed, n_splits=n_splits, repeats=repeats
    )

    print('=' * 60)
    print(f'CNN-on-UNet Subtype TD-rest 二分类公平重算 ({n_splits}-fold x {repeats}, 仅 subtype 任务)')
    print(f"subtype 患者数: {len(records)} | backbone={cfg.BACKBONE} | "
          f"folds={len(folds)} | device={cfg.DEVICE}")
    print('=' * 60)

    fold_results: List[Dict] = []
    pred_rows: List[Dict] = []

    for global_idx, fold in enumerate(folds, start=1):
        repeat, fold_no = fold['repeat'], fold['fold']
        train_patients = _only_subtype_subset_patients(fold['train_patients_full'], pid_to_patient)
        val_patients = _only_subtype_subset_patients(fold['val_patients_full'], pid_to_patient)
        print(
            f"\n--- repeat {repeat} fold {fold_no} ({global_idx}/{len(folds)}) | "
            f"train={len(train_patients)} val={len(val_patients)} ---",
            flush=True,
        )

        tdrest_classifier, tdrest_model = _only_subtype_train_one_subtask(
            cfg, train_patients, val_patients,
            'pd_subtype_tdrest', fold_idx=global_idx,
        )
        tdrest_pred, tdrest_prob = _only_subtype_infer_patient_level(
            tdrest_model, tdrest_classifier, val_patients, cfg
        )
        del tdrest_model, tdrest_classifier
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        rows = td_rest_rows(pid_to_patient, fold['val_patients_full'],
                            tdrest_pred, tdrest_prob, repeat, fold_no)
        fold_results.append(td_rest_fold(rows))
        pred_rows.extend(rows)

    return save_td_rest_results(output_dir, fold_results, pred_rows,
                                n_splits, repeats, seed)


def _dual_subtype_load_json(path: str) -> List[Dict]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _dual_subtype_align_subtype_patients(config: DualStreamExperimentConfig) -> Tuple[List[Dict], List[Dict]]:
    en_json = CODES_ROOT / config.en_dataset_json
    unet_json = CODES_ROOT / config.unet_dataset_json
    en_all = sorted(_dual_subtype_load_json(str(en_json)), key=lambda p: p['patient_id'])
    unet_map = {p['patient_id']: p for p in _dual_subtype_load_json(str(unet_json))}

    en_patients = [
        p for p in en_all
        if p.get('group') == 'pd'
        and p.get('subtype') in SUBTYPE_TO_3CLS
        and p['patient_id'] in unet_map
    ]
    unet_patients = [unet_map[p['patient_id']] for p in en_patients]
    return en_patients, unet_patients


def _dual_subtype_binary_label(subtype: str, task: str) -> int:
    y3 = SUBTYPE_TO_3CLS[subtype]
    if task == 'pd_subtype_tdrest':
        return subtype_tdrest_label(y3)
    label = subtype_pigd_mixed_label(y3)
    return -1 if label is None else label


def _dual_subtype_sample_patient_ids(
    en_patients: List[Dict],
    unet_patients: List[Dict],
    base_path: str,
    region: str,
) -> List[str]:
    sample_pids: List[str] = []
    for en_p, unet_p in zip(en_patients, unet_patients):
        en_paths = en_p['image_paths']
        unet_paths = unet_p['image_paths']
        if region in ('both', 'central'):
            for ep, up in zip(en_paths.get('central', []), unet_paths.get('central', [])):
                if os.path.exists(os.path.join(base_path, ep)) and os.path.exists(os.path.join(base_path, up)):
                    sample_pids.append(en_p['patient_id'])
        if region in ('both', 'whorl'):
            for ep, up in zip(en_paths.get('whorl', []), unet_paths.get('whorl', [])):
                if os.path.exists(os.path.join(base_path, ep)) and os.path.exists(os.path.join(base_path, up)):
                    sample_pids.append(en_p['patient_id'])
    return sample_pids


def _dual_subtype_build_dataset_with_patient_ids(
    en_patients: List[Dict],
    unet_patients: List[Dict],
    base_path: str,
    en_transform,
    unet_transform,
    region: str,
    task: str | None = None,
) -> Tuple[DualViewDataset, List[str]]:
    ds = DualViewDataset(
        en_patients,
        unet_patients,
        base_path,
        en_transform=en_transform,
        unet_transform=unet_transform,
        region=region,
    )
    sample_pids = _dual_subtype_sample_patient_ids(en_patients, unet_patients, base_path, region)
    if len(sample_pids) != len(ds.samples):
        raise RuntimeError(
            f"sample/patient alignment mismatch: {len(ds.samples)} samples vs {len(sample_pids)} ids"
        )

    if task is not None:
        subtype_by_pid = {p['patient_id']: p['subtype'] for p in en_patients}
        relabeled = []
        for sample, pid in zip(ds.samples, sample_pids):
            en_path, unet_path, pd_l, stage_l, _sub_l = sample[:5]
            relabeled.append((
                en_path,
                unet_path,
                pd_l,
                stage_l,
                _dual_subtype_binary_label(subtype_by_pid[pid], task),
                pid,
            ))
        ds.samples = relabeled

    return ds, sample_pids


def _dual_subtype_build_model(config: DualStreamExperimentConfig, task: str, device: torch.device) -> DualViewModel:
    model = DualViewModel(
        task=task,
        backbone=config.backbone,
        pretrained=config.pretrained,
        shared_backbone=config.shared_backbone,
        fusion_dim=config.fusion_dim,
        dropout=config.dropout,
        classifier_dropout=config.classifier_dropout,
    )
    return model.to(device)


def _dual_subtype_build_optimizer(config: DualStreamExperimentConfig, model: torch.nn.Module):
    backbone_params, other_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'backbone_' in name:
            backbone_params.append(param)
        else:
            other_params.append(param)

    param_groups = [
        {'params': backbone_params, 'lr': config.learning_rate * config.backbone_lr_ratio},
        {'params': other_params, 'lr': config.learning_rate},
    ]
    if config.optimizer == 'adam':
        return torch.optim.Adam(param_groups, weight_decay=config.weight_decay)
    return torch.optim.AdamW(param_groups, weight_decay=config.weight_decay)


def _dual_subtype_build_scheduler(config: DualStreamExperimentConfig, optimizer):
    def _lr_lambda(epoch: int) -> float:
        if epoch < config.warmup_epochs:
            return (epoch + 1) / max(config.warmup_epochs, 1)
        if config.lr_scheduler == 'cosine':
            import math
            progress = (epoch - config.warmup_epochs) / max(config.epochs - config.warmup_epochs, 1)
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)


def _dual_subtype_train_one_subtask(
    config: DualStreamExperimentConfig,
    train_en: List[Dict],
    val_en: List[Dict],
    train_unet: List[Dict],
    val_unet: List[Dict],
    base_path: str,
    device: torch.device,
    task: str,
    fold_idx: int,
) -> DualViewModel:
    train_tf, val_tf = _build_transforms(_dual_subtype_to_dual_view_config(config))
    train_ds, _ = _dual_subtype_build_dataset_with_patient_ids(
        train_en, train_unet, base_path, train_tf, train_tf, config.region, task
    )
    val_ds, _ = _dual_subtype_build_dataset_with_patient_ids(
        val_en, val_unet, base_path, val_tf, val_tf, config.region, task
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    model = _dual_subtype_build_model(config, task, device)
    optimizer = _dual_subtype_build_optimizer(config, model)
    scheduler = _dual_subtype_build_scheduler(config, optimizer)

    best_metric = -1.0
    best_state = None
    patience_counter = 0

    for epoch in range(config.epochs):
        model.train()
        total_loss = 0.0
        batches = 0
        train_preds, train_labels = [], []
        for batch in train_loader:
            en_img, unet_img, _pd_l, _stage_l, sub_l = batch[:5]
            mask = sub_l >= 0
            if not mask.any():
                continue
            en_m = en_img[mask].to(device)
            unet_m = unet_img[mask].to(device)
            labels = sub_l[mask].to(device)

            logits = model(en_m, unet_m)
            loss = F.cross_entropy(logits, labels, label_smoothing=config.label_smoothing)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += float(loss.item())
            batches += 1
            train_preds.extend(logits.argmax(1).detach().cpu().numpy())
            train_labels.extend(labels.detach().cpu().numpy())

        scheduler.step()
        val_acc = _dual_subtype_eval_binary_acc(model, val_loader, device)
        improved = val_acc > best_metric
        if improved:
            best_metric = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch == 0 or epoch + 1 == config.epochs or (epoch + 1) % 10 == 0:
            train_acc = accuracy_score(train_labels, train_preds) if train_labels else 0.0
            train_loss = total_loss / max(1, batches)
            print(
                f"    [{task}] fold={fold_idx} E{epoch + 1:03d}/{config.epochs} "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"val_bin_acc={val_acc:.4f}{' *' if improved else ''}",
                flush=True,
            )
        if patience_counter >= config.patience:
            print(f"    [{task}] early stop at epoch {epoch + 1}", flush=True)
            break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return model


def _dual_subtype_to_dual_view_config(config: DualStreamExperimentConfig):
    from unet.dual_view_classifier import DualViewConfig

    return DualViewConfig(
        task='subtype',
        backbone=config.backbone,
        pretrained=config.pretrained,
        shared_backbone=config.shared_backbone,
        fusion_dim=config.fusion_dim,
        dropout=config.dropout,
        classifier_dropout=config.classifier_dropout,
        lr=config.learning_rate,
        backbone_lr_ratio=config.backbone_lr_ratio,
        weight_decay=config.weight_decay,
        optimizer=config.optimizer,
        lr_scheduler=config.lr_scheduler,
        warmup_epochs=config.warmup_epochs,
        epochs=config.epochs,
        patience=config.patience,
        batch_size=config.batch_size,
        region=config.region,
        data_augmentation=config.data_augmentation,
        aug_horizontal_flip=config.aug_horizontal_flip,
        aug_rotation=config.aug_rotation,
        aug_brightness=config.aug_brightness,
        aug_contrast=config.aug_contrast,
        label_smoothing=config.label_smoothing,
        num_workers=config.num_workers,
        seed=config.seed,
    )


@torch.no_grad()
def _dual_subtype_eval_binary_acc(model: DualViewModel, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct, total = 0, 0
    for batch in loader:
        en_img, unet_img, _pd_l, _stage_l, sub_l = batch[:5]
        mask = sub_l >= 0
        if not mask.any():
            continue
        logits = model(en_img[mask].to(device), unet_img[mask].to(device))
        preds = logits.argmax(1).cpu()
        labels = sub_l[mask]
        correct += int((preds == labels).sum().item())
        total += int(labels.numel())
    return correct / total if total else 0.0


@torch.no_grad()
def _dual_subtype_infer_patient_level(
    model: DualViewModel,
    en_patients: List[Dict],
    unet_patients: List[Dict],
    config: DualStreamExperimentConfig,
    base_path: str,
    device: torch.device,
) -> Tuple[Dict[str, int], Dict[str, float]]:
    _train_tf, val_tf = _build_transforms(_dual_subtype_to_dual_view_config(config))
    ds, sample_pids = _dual_subtype_build_dataset_with_patient_ids(
        en_patients, unet_patients, base_path, val_tf, val_tf, config.region
    )
    loader = DataLoader(
        ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    model.eval()
    sums: Dict[str, np.ndarray] = defaultdict(lambda: np.zeros(2, dtype=np.float64))
    counts: Dict[str, int] = defaultdict(int)
    offset = 0
    for batch in loader:
        en_img, unet_img, _pd_l, _stage_l, sub_l = batch[:5]
        batch_size = int(sub_l.size(0))
        logits = model(en_img.to(device), unet_img.to(device))
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        for i, pid in enumerate(sample_pids[offset:offset + batch_size]):
            sums[pid] += probs[i]
            counts[pid] += 1
        offset += batch_size

    pred_by_pid: Dict[str, int] = {}
    prob_class1_by_pid: Dict[str, float] = {}
    for pid, total_prob in sums.items():
        mean_prob = total_prob / max(1, counts[pid])
        pred_by_pid[pid] = int(np.argmax(mean_prob))
        prob_class1_by_pid[pid] = float(mean_prob[1])
    return pred_by_pid, prob_class1_by_pid


def dual_subtype_run_dual_stream_subtype_td_rest_experiment(
    config: DualStreamExperimentConfig,
    output_dir: str,
    seed: int = 42,
    n_splits: int = 5,
    repeats: int = 3,
) -> Dict:
    os.makedirs(output_dir, exist_ok=True)
    cfg = deepcopy(config)
    cfg.seed = seed
    set_seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_path = str(CODES_ROOT / 'datasets')

    en_patients, unet_patients = _dual_subtype_align_subtype_patients(cfg)
    en_by_pid = {p['patient_id']: p for p in en_patients}
    unet_by_pid = {p['patient_id']: p for p in unet_patients}
    records = [{'patient_id': p['patient_id'], 'subtype': p['subtype']} for p in en_patients]
    folds = build_patient_5x3_subtype_folds(records, seed=seed, n_splits=n_splits, repeats=repeats)

    print('=' * 60)
    print('Dual-stream subtype TD-rest revalidation (patient 5-fold x3)')
    print(f"subtype patients: {len(en_patients)} | backbone={cfg.backbone} | "
          f"shared={cfg.shared_backbone} | folds={len(folds)} | device={device}")
    print('=' * 60)

    fold_results: List[Dict] = []
    pred_rows: List[Dict] = []

    for global_idx, fold in enumerate(folds, start=1):
        repeat, fold_no = fold['repeat'], fold['fold']
        train_pids = fold['train_patients_full']
        val_pids = fold['val_patients_full']
        train_en = [en_by_pid[pid] for pid in train_pids]
        val_en = [en_by_pid[pid] for pid in val_pids]
        train_unet = [unet_by_pid[pid] for pid in train_pids]
        val_unet = [unet_by_pid[pid] for pid in val_pids]

        print(
            f"\n--- repeat {repeat} fold {fold_no} ({global_idx}/{len(folds)}) | "
            f"train={len(train_en)} val={len(val_en)} ---",
            flush=True,
        )

        tdrest_model = _dual_subtype_train_one_subtask(
            cfg, train_en, val_en, train_unet, val_unet, base_path, device,
            'pd_subtype_tdrest', fold_idx=global_idx,
        )
        tdrest_pred, tdrest_prob = _dual_subtype_infer_patient_level(
            tdrest_model, val_en, val_unet, cfg, base_path, device
        )
        del tdrest_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        rows = td_rest_rows(en_by_pid, fold['val_patients_full'],
                            tdrest_pred, tdrest_prob, repeat, fold_no)
        fold_results.append(td_rest_fold(rows))
        pred_rows.extend(rows)

    return save_td_rest_results(output_dir, fold_results, pred_rows,
                                n_splits, repeats, seed)


from utils.experiment_reporting import with_patient_report


@with_patient_report
def run_paper_experiment(exp, output_dir, device='cpu', num_workers=0):
    exp = resolve_baseline_tasks(exp)
    from unet.prerequisites import require_segmented_pd
    require_segmented_pd(CODES_ROOT / 'datasets', require_original=exp['_method'] == 'unet_dual')
    if exp['_method'] == 'unet_only':
        cfg = only_build_paper_config(exp, device, num_workers)
        if exp.get('_subtype_td_rest'):
            return only_subtype_run_unet_subtype_td_rest_experiment(
                cfg, output_dir=output_dir, seed=cfg.SEED,
                n_splits=exp.get('num_folds', cfg.NUM_FOLDS), repeats=exp.get('kfold_repeats', 3))
        return only_run_single_experiment(cfg, tasks=exp['_run_tasks'], output_dir=output_dir)
    if exp['_method'] != 'unet_dual':
        raise ValueError(f"Unknown U-Net variant: {exp['_method']}")
    cfg = dual_build_paper_config(exp, device, num_workers)
    if exp.get('_subtype_td_rest'):
        return dual_subtype_run_dual_stream_subtype_td_rest_experiment(
            cfg, output_dir=output_dir, seed=cfg.seed,
            n_splits=exp.get('num_folds', 5), repeats=exp.get('kfold_repeats', 3))
    return dual_run_single_experiment(cfg, output_dir=output_dir)
