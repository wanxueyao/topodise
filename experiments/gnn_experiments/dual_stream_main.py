"""
dual_stream_main.py - 双视角 CNN 单次实验执行器

工作流程
--------
1. 从 pd_dataset_en 加载原始眼底图像患者数据
2. 从 pd_dataset_unet_en 加载 UNet 神经纤维图像患者数据
3. 按 patient_id 匹配，执行 K-fold 划分（patient 级别）
4. 每折：联合训练双流 CNN（en 视角 + unet 视角）
5. 保存最佳模型（pth）、指标（json）和报告（md）

用法
----
    from dual_stream_main import run_single_experiment, DualStreamExperimentConfig
    
    config = DualStreamExperimentConfig(
        name='my_experiment',
        backbone='resnet18',
        pretrained='Half',
        epochs=50,
    )
    result = run_single_experiment(config, output_dir='results/dual_stream')
"""

import os
import sys
import json
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Any, Optional

import numpy as np
import torch

# ─── 路径配置 ─────────────────────────────────────────────────────────────────
_EXP_DIR = os.path.dirname(os.path.abspath(__file__))
_CODES_DIR = os.path.dirname(os.path.dirname(_EXP_DIR))
if _CODES_DIR not in sys.path:
    sys.path.insert(0, _CODES_DIR)

from core.data_loader import load_and_filter_dataset, stratified_kfold, SUBTYPE_MAP
from gnn.dual_view_classifier import DualViewConfig, DualViewClassifier
from gnn.dual_view_model import DUAL_VIEW_TASK_CONFIG
from utils.misc import Tee
from utils.patient_outputs import save_fold_patient_outputs
from utils.visualize import plot_training_curves


# ─── 配置类 ───────────────────────────────────────────────────────────────────

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


# ─── 单次实验执行 ─────────────────────────────────────────────────────────────

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
        en_json_path = os.path.join(_CODES_DIR, cfg.en_dataset_json)
        en_patients = sorted(
            load_and_filter_dataset(en_json_path, cfg.dataset_group),
            key=lambda p: p['patient_id'],
        )

        # 加载 UNet 处理后的眼底图
        unet_json_path = os.path.join(_CODES_DIR, cfg.unet_dataset_json)
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

        base_path = os.path.join(_CODES_DIR, 'datasets')

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
            f.write(f"| 架构 | dual_view |\n")
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


# ─── 公共接口 ─────────────────────────────────────────────────────────────────

def run_single_experiment(config: DualStreamExperimentConfig, 
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


# ─── 命令行入口 ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    config = DualStreamExperimentConfig(
        name='dual_stream_quick_test',
        backbone='resnet18',
        pretrained='Half',
        epochs=40,
        num_folds=2,
    )
    run_single_experiment(
        config,
        output_dir=os.path.join(_CODES_DIR, '..', 'experiment_records', 'dual_stream'),
    )
