"""simclr 实验实现：训练、评估及单组执行；参数与批量控制见 batch_experiments.py。"""
from __future__ import annotations

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from experiments._execution import single_main
    single_main("experiments.simclr_experiments.main")
    raise SystemExit(0)

from utils.td_rest import resolve_baseline_tasks, td_rest_rows, td_rest_fold, save_td_rest_results
import sys
import os
import time
import json
import torch
import numpy as np
from datetime import datetime
from collections import defaultdict
from typing import Literal, List, Dict, Any, Optional
from core.simclr_classifier import SimCLRClassifier, SimCLRClassifierConfig, TASK_CONFIG
from core.data_loader import PDDataset, load_and_filter_dataset, stratified_kfold
from utils.patient_outputs import save_fold_patient_outputs, task_labels_for_indices
from utils.misc import set_seed, Tee, save_config
from utils.visualize import plot_confusion_matrix, plot_roc_auc, plot_training_curves
import hashlib
from torch.utils.data import Subset
from core.simclr_trainer import SimCLRModel, simclr_pretrain


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)



# ==================== 实验配置 ====================

class SimCLRExperimentConfig(SimCLRClassifierConfig):
    """
    SimCLR 实验配置类（继承自 SimCLRClassifierConfig）

    配置分组
    --------
    架构配置:
        ARCHITECTURE: 'single_stream' / 'mixed_single_stream' / 'dual_stream'
        BACKBONE: 'mobilevit_xxs' / 'mobilevit_xs' / 'vanillanet_5' / 'efficientnet_b0'
        PRETRAINED: True / False / 'Half'
        DROPOUT: 特征层 Dropout 概率

    SimCLR 预训练配置:
        SIMCLR_EPOCHS: 自监督预训练轮数（推荐 50~200）
        SIMCLR_LR: 预训练学习率
        SIMCLR_BATCH_SIZE: 预训练 batch 大小
        SIMCLR_TEMPERATURE: NT-Xent 温度系数
        SIMCLR_PROJ_DIM: 投影头输出维度
        SIMCLR_PROJ_HIDDEN_DIM: 投影头隐层维度

    微调训练配置:
        NUM_EPOCHS: 微调轮数
        LEARNING_RATE: 微调学习率
        BATCH_SIZE: 微调 batch 大小
        LR_SCHEDULER: 'warmup_cosine' / 'cosine' / 'step'
        BACKBONE_LR_RATIO: backbone 差异化学习率倍率

    注意: 无 DATA_AUGMENTATION / AUG_* 参数，微调阶段不使用图像增强。
    """
    # 继承 SimCLRClassifierConfig 的所有默认值
    ARCHITECTURE: Literal['single_stream', 'mixed_single_stream', 'dual_stream'] = 'single_stream'
    BACKBONE: Literal['resnet18'] = 'resnet18'
    PRETRAINED: Literal[True, False, 'Half'] = True
    DROPOUT: float = 0.4

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

    PD_LOSS_TYPE: str = 'ce'
    STAGE_LOSS_TYPE: str = 'label_smooth'
    SUBTYPE_LOSS_TYPE: str = 'ce'
    LABEL_SMOOTH_EPS: float = 0.1

    DATASET_MODE: Literal['pd_control', 'pd_control_cataract', 'pd_all'] = 'pd_all'
    DATASET_JSON: str = 'datasets/pd_dataset_en/patients_dataset_en.json'

    BATCH_SIZE: int = 16
    NUM_EPOCHS: int = 60
    LEARNING_RATE: float = 5e-4
    WEIGHT_DECAY: float = 1e-3
    OPTIMIZER: str = 'adamw'

    LR_SCHEDULER: str = 'warmup_cosine'
    LR_STEP_SIZE: int = 15
    LR_GAMMA: float = 0.5
    BACKBONE_LR_RATIO: float = 0.1

    EARLY_STOPPING: bool = True
    PATIENCE: int = 15


    NUM_FOLDS: int = 5
    KFOLD_REPEATS: int = 1
    SEED: int = 42
    KFOLD_MODE: Literal['patient', 'image'] = 'patient'

    NUM_WORKERS: int = 4
    DEVICE: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    OUTPUT_DIR: str = 'results'


# ==================== 实验运行器 ====================

class SimCLRExperimentRunner:
    """
    SimCLR 实验运行器

    执行一组配置下的完整 K-fold 实验（SimCLR 预训练 + 有监督微调）。
    每个 fold 独立执行 SimCLR 预训练，确保预训练数据与验证数据无泄露。

    使用方法
    --------
    >>> config = SimCLRExperimentConfig()
    >>> config.BACKBONE = 'mobilevit_xxs'
    >>> config.SIMCLR_EPOCHS = 100
    >>> runner = SimCLRExperimentRunner(config)
    >>> results = runner.run(tasks=['pd', 'stage'])
    """

    def __init__(self, config: SimCLRExperimentConfig):
        self.config = config
        self.classifier = SimCLRClassifier(config)
        self.output_dir: Optional[str] = None

    def _resolve_dataset_json_path(self) -> str:
        """解析数据集 JSON 路径，兼容从 ARIS 根目录或 codes 目录启动。"""
        json_path = self.config.DATASET_JSON
        if os.path.isabs(json_path):
            return json_path

        candidates = [
            os.path.join(PROJECT_ROOT, json_path),
            os.path.join(os.path.dirname(PROJECT_ROOT), json_path),
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
        return candidates[0]

    def run(self, tasks: Optional[List[str]] = None,
            output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        运行完整 K-fold 实验

        Args:
            tasks: 任务列表，默认 ['pd', 'stage', 'subtype']
            output_dir: 自定义输出目录，默认自动生成

        Returns:
            实验结果字典
        """
        set_seed(self.config.SEED)
        start_time = time.time()

        if tasks is None:
            tasks = ['pd', 'stage', 'subtype']

        self.config.DATASET_JSON = self._resolve_dataset_json_path()

        self.output_dir = output_dir or _create_simclr_output_dir('results', self.config)
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

        print(f"\n实验完成！用时 {duration/60:.2f} 分钟")
        print(f"结果保存到: {self.output_dir}")
        return results

    def _execute_experiment(self, tasks: List[str]) -> Dict[str, Any]:
        """执行实验主流程"""
        config = self.config
        config.DATASET_JSON = self._resolve_dataset_json_path()
        dataset_base_path = os.path.dirname(os.path.dirname(config.DATASET_JSON))
        self._print_header()

        patients = load_and_filter_dataset(config.DATASET_JSON, config.DATASET_MODE)
        print(f"\n加载 {len(patients)} 个患者")

        results = {}

        if config.ARCHITECTURE == 'single_stream':
            for task in tasks:
                results[task] = {'regions': {}}
                for region in ['central', 'whorl']:
                    print(f"\n{'='*60}")
                    print(f"任务: {TASK_CONFIG[task]['name']} | 区域: {region}")
                    print('='*60)

                    dataset = PDDataset(
                        patients, 'single_stream',
                        self.classifier.train_transform,
                        region=region, base_path=dataset_base_path,
                    )
                    if len(dataset) == 0:
                        print(f"⚠ 区域 {region} 无数据，跳过")
                        continue

                    print(f"样本数: {len(dataset)}")
                    if task == 'pd_subtype_tdrest':
                        region_result = self._run_subtype_td_rest(dataset, f"_{region}")
                    else:
                        region_result = self._run_kfold(dataset, task, f"_{region}")
                    results[task]['regions'][region] = region_result
        else:
            dataset = PDDataset(
                patients, config.ARCHITECTURE,
                self.classifier.train_transform,
                base_path=dataset_base_path,
            )

            if len(dataset) == 0:
                print("⚠ 数据集为空")
                return results

            print(f"样本数: {len(dataset)}")

            for task in tasks:
                print(f"\n{'='*60}")
                print(f"任务: {TASK_CONFIG[task]['name']}")
                print('='*60)
                if task == 'pd_subtype_tdrest':
                    results[task] = self._run_subtype_td_rest(dataset)
                else:
                    results[task] = self._run_kfold(dataset, task)

        self._generate_report(results)
        return results

    def _run_kfold(self, dataset, task: str, suffix: str = "") -> Dict[str, Any]:
        """执行单任务 K-fold 交叉验证"""
        config = self.config
        task_cfg = TASK_CONFIG[task]

        splits = list(stratified_kfold(
            dataset.PD_labels, dataset.stage_labels, dataset.subtype_labels,
            n_splits=config.NUM_FOLDS, random_state=config.SEED,
            kfold_mode=config.KFOLD_MODE, patient_ids=dataset.patient_ids,
        ))

        print(f"K-fold 模式: {config.KFOLD_MODE} "
              f"(按{'患者' if config.KFOLD_MODE == 'patient' else '图像'}划分)")

        fold_results = []
        all_preds = {'labels': [], 'preds': [], 'probs': []}
        best_models = []

        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            print(f"\n--- Fold {fold_idx+1}/{config.NUM_FOLDS} ---")

            train_subset = torch.utils.data.Subset(dataset, train_idx)
            val_subset = torch.utils.data.Subset(dataset, val_idx)

            result = self.classifier.train(
                train_subset, val_subset, task,
                fold_idx=fold_idx+1, suffix=suffix,
            )

            plot_training_curves(
                result['train_history'],
                task_cfg['name'] + suffix,
                fold_idx+1, self.output_dir,
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
                    'train_acc': result['train_history']['train_acc'][-1]
                        if 'train_acc' in result['train_history'] else 0.0,
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
                'fold': fold_idx+1,
                'state': result['model_state'],
                'acc': val_result['acc'],
            })

        if not fold_results:
            return {}

        summary = self._summarize_results(fold_results, all_preds, task_cfg, suffix)

        best = max(best_models, key=lambda x: x['acc'])
        model_path = os.path.join(self.output_dir, f"best_model_{task}{suffix}.pth")
        torch.save(best['state'], model_path)
        print(f"✓ 保存最佳模型: {model_path}")

        return summary

    def _run_subtype_td_rest(self, dataset, suffix: str = "") -> Dict[str, Any]:
        """执行 SimCLR subtype hierarchy：共享预训练 backbone，TD-vs-rest 二分类 head。"""
        config = self.config
        full_indices = [i for i, y in enumerate(dataset.subtype_labels) if int(y) >= 0]
        if not full_indices:
            print("⚠ 无 subtype 标签样本，跳过 pd_subtype_tdrest")
            return {}

        full_pd = [1 for _ in full_indices]
        full_stage = [-1 for _ in full_indices]
        full_subtype = [int(dataset.subtype_labels[i]) for i in full_indices]
        full_patient_ids = [int(dataset.patient_ids[i]) for i in full_indices]

        fold_results: List[Dict[str, Any]] = []
        prediction_rows: List[Dict[str, Any]] = []
        total_folds = config.NUM_FOLDS * config.KFOLD_REPEATS
        print(
            f"TD-rest subtype 协议: patient {config.NUM_FOLDS}-fold x "
            f"{config.KFOLD_REPEATS} repeats，共 {total_folds} folds"
        )

        original_precomputed = self.classifier.precomputed_simclr
        try:
            for repeat in range(config.KFOLD_REPEATS):
                splits = list(stratified_kfold(
                    full_pd, full_stage, full_subtype,
                    n_splits=config.NUM_FOLDS,
                    random_state=config.SEED + repeat * 1000,
                    kfold_mode='patient',
                    patient_ids=full_patient_ids,
                ))

                for fold_idx, (train_local_idx, val_local_idx) in enumerate(splits, start=1):
                    global_fold = repeat * config.NUM_FOLDS + fold_idx
                    print(f"\n--- TD-rest Repeat {repeat} Fold {fold_idx}/{config.NUM_FOLDS} ---")
                    train_idx = [full_indices[int(i)] for i in train_local_idx]
                    val_idx = [full_indices[int(i)] for i in val_local_idx]

                    td_train = torch.utils.data.Subset(dataset, train_idx)
                    full_val = torch.utils.data.Subset(dataset, val_idx)

                    self.classifier.precomputed_simclr = original_precomputed
                    shared_simclr = self.classifier.pretrain_backbone(td_train, fold_idx=global_fold)
                    self.classifier.precomputed_simclr = shared_simclr
                    td_result = self.classifier.train(
                        td_train, full_val, 'pd_subtype_tdrest',
                        fold_idx=global_fold, suffix=f'{suffix}_td_rest',
                    )

                    td_pred = self.classifier.predict_dataset(
                        td_result['model'], full_val, 'pd_subtype_tdrest'
                    )

                    pids = [int(dataset.patient_ids[i]) for i in val_idx]
                    by_pid = self._aggregate_patient_predictions(pids, td_pred, positive_class=1)
                    patient_map = {int(dataset.patient_ids[i]): {'subtype': ('TD', 'PIGD', 'mixed')[int(dataset.subtype_labels[i])]} for i in val_idx}
                    rows = td_rest_rows(patient_map, sorted(set(pids)),
                                        {pid: v['pred'] for pid, v in by_pid.items()},
                                        {pid: v['prob_positive'] for pid, v in by_pid.items()}, repeat, fold_idx)
                    fold_results.append(td_rest_fold(rows))
                    prediction_rows.extend(rows)

        finally:
            self.classifier.precomputed_simclr = original_precomputed

        return save_td_rest_results(self.output_dir, fold_results, prediction_rows,
                                    config.NUM_FOLDS, config.KFOLD_REPEATS, config.SEED,
                                    prefix='pd_subtype' + suffix)


    @staticmethod
    def _aggregate_patient_predictions(pids: List[int], pred: Dict[str, Any], positive_class: int) -> Dict[int, Dict[str, Any]]:
        class_ids = [int(c) for c in pred['class_ids']]
        pos_idx = class_ids.index(positive_class) if positive_class in class_ids else len(class_ids) - 1
        grouped: Dict[int, List[List[float]]] = defaultdict(list)
        for pid, probs in zip(pids, pred['probs']):
            grouped[int(pid)].append([float(p) for p in probs])

        out = {}
        for pid, rows in grouped.items():
            mean_probs = np.mean(np.array(rows), axis=0)
            pred_pos = int(np.argmax(mean_probs))
            out[pid] = {
                'pred': class_ids[pred_pos],
                'prob_positive': float(mean_probs[pos_idx]),
            }
        return out





    def _summarize_results(self, fold_results: List[Dict], all_preds: Dict,
                           task_cfg: Dict, suffix: str) -> Dict[str, Any]:
        """汇总 K-fold 结果并生成可视化"""
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
            actual_names = [class_names[l] if l < len(class_names) else str(l)
                            for l in unique_labels]

            plot_confusion_matrix(
                all_preds['labels'], all_preds['preds'],
                actual_names, safe_name, 0, self.output_dir, is_test=True,
            )

            if all_preds['probs']:
                auc = plot_roc_auc(
                    all_preds['labels'],
                    np.array(all_preds['probs']),
                    actual_names, safe_name, self.output_dir,
                )
                if isinstance(auc, dict):
                    auc_scores = {actual_names[i]: auc[i]
                                  for i in auc if isinstance(i, int)}
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
        """打印实验信息头"""
        cfg = self.config
        print("="*60)
        print("帕金森病图像分类实验 [SimCLR 自监督预训练]")
        print("="*60)
        print(f"架构: {cfg.ARCHITECTURE} | Backbone: {cfg.BACKBONE}")
        print(f"预训练权重: {cfg.PRETRAINED} | 微调 Dropout: {cfg.DROPOUT}")
        print(f"数据集: {cfg.DATASET_MODE} | K-fold: {cfg.NUM_FOLDS} ({cfg.KFOLD_MODE})")
        print(f"[SimCLR] 预训练数据源: {cfg.SIMCLR_PRETRAIN_DATASET}")
        print(f"[SimCLR] 预训练 Epochs: {cfg.SIMCLR_EPOCHS} | LR: {cfg.SIMCLR_LR} | "
              f"BatchSize: {cfg.SIMCLR_BATCH_SIZE} | T: {cfg.SIMCLR_TEMPERATURE}")
        print(f"[SimCLR] ProjDim: {cfg.SIMCLR_PROJ_DIM} | "
              f"ProjHiddenDim: {cfg.SIMCLR_PROJ_HIDDEN_DIM}")
        print(f"[微调] Epochs: {cfg.NUM_EPOCHS} | LR: {cfg.LEARNING_RATE} | "
              f"Optimizer: {cfg.OPTIMIZER} | WeightDecay: {cfg.WEIGHT_DECAY}")
        print(f"[微调] Loss — PD: {cfg.PD_LOSS_TYPE} | Stage: {cfg.STAGE_LOSS_TYPE} | "
              f"Subtype: {cfg.SUBTYPE_LOSS_TYPE}")
        print("图像增强: 无（SimCLR 预训练阶段内部处理，微调阶段使用纯净数据）")
        print("="*60)

    def _generate_report(self, results: Dict):
        """生成 Markdown 实验报告"""
        report_path = os.path.join(self.output_dir, 'experiment_report.md')
        cfg = self.config

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# SimCLR 增强实验报告\n\n")
            f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("## 实验配置\n\n")
            f.write("### 架构与 backbone\n\n")
            f.write("| 参数 | 值 |\n|---|---|\n")
            f.write(f"| 架构 | `{cfg.ARCHITECTURE}` |\n")
            f.write(f"| Backbone | `{cfg.BACKBONE}` |\n")
            f.write(f"| ImageNet 预训练 | `{cfg.PRETRAINED}` |\n")
            f.write(f"| 微调 Dropout | {cfg.DROPOUT} |\n")
            f.write(f"| 数据集 | `{cfg.DATASET_MODE}` |\n")
            f.write(f"| K-fold | {cfg.NUM_FOLDS} ({cfg.KFOLD_MODE}) |\n")
            f.write("\n")

            f.write("### SimCLR 预训练配置\n\n")
            f.write("| 参数 | 值 |\n|---|---|\n")
            f.write(f"| 预训练 Epochs | {cfg.SIMCLR_EPOCHS} |\n")
            f.write(f"| 预训练 LR | {cfg.SIMCLR_LR} |\n")
            f.write(f"| 预训练 Batch Size | {cfg.SIMCLR_BATCH_SIZE} |\n")
            f.write(f"| 预训练数据源 | `{cfg.SIMCLR_PRETRAIN_DATASET}` |\n")
            f.write(f"| CORN-1 路径 | `{cfg.CORN1_ROOT}` |\n")
            f.write(f"| CORN-1 splits | `{cfg.CORN1_PRETRAIN_SPLITS}` |\n")
            f.write(f"| 温度系数 (τ) | {cfg.SIMCLR_TEMPERATURE} |\n")
            f.write(f"| 投影维度 | {cfg.SIMCLR_PROJ_DIM} |\n")
            f.write(f"| 投影隐层维度 | {cfg.SIMCLR_PROJ_HIDDEN_DIM} |\n")
            f.write(f"| 预训练权重衰减 | {cfg.SIMCLR_WEIGHT_DECAY} |\n")
            f.write("\n")

            f.write("### 微调配置\n\n")
            f.write("| 参数 | 值 |\n|---|---|\n")
            f.write(f"| 微调 Epochs | {cfg.NUM_EPOCHS} |\n")
            f.write(f"| 微调 LR | {cfg.LEARNING_RATE} |\n")
            f.write(f"| 微调 Batch Size | {cfg.BATCH_SIZE} |\n")
            f.write(f"| 优化器 | `{cfg.OPTIMIZER}` (WeightDecay={cfg.WEIGHT_DECAY}) |\n")
            f.write(f"| 调度器 | `{cfg.LR_SCHEDULER}` |\n")
            f.write(f"| Backbone LR 倍率 | {cfg.BACKBONE_LR_RATIO} |\n")
            f.write(f"| PD 损失 | `{cfg.PD_LOSS_TYPE}` |\n")
            f.write(f"| Stage 损失 | `{cfg.STAGE_LOSS_TYPE}` |\n")
            f.write(f"| Subtype 损失 | `{cfg.SUBTYPE_LOSS_TYPE}` |\n")
            f.write("| 图像增强 | 无（SimCLR 内部处理） |\n")
            f.write("\n")

            f.write("## 结果摘要\n\n")
            f.write("| 任务 | 区域 | Patient Accuracy | Image Accuracy | F1 Score |\n")
            f.write("|---|---|---|---|---|\n")

            for task, result in results.items():
                if task == 'meta':
                    continue
                task_name = TASK_CONFIG[task]['name']

                if 'regions' in result:
                    for region, rr in result['regions'].items():
                        if 'final_results' in rr:
                            fr = rr['final_results']
                            f.write(f"| {task_name} | {region} | "
                                    f"{fr.get('avg_patient_acc', 0.0):.4f}±{fr.get('std_patient_acc', 0.0):.4f} | "
                                    f"{fr['avg_acc']:.4f}±{fr['std_acc']:.4f} | "
                                    f"{fr['avg_f1']:.4f}±{fr['std_f1']:.4f} |\n")
                elif 'final_results' in result:
                    fr = result['final_results']
                    f.write(f"| {task_name} | - | "
                            f"{fr.get('avg_patient_acc', 0.0):.4f}±{fr.get('std_patient_acc', 0.0):.4f} | "
                            f"{fr['avg_acc']:.4f}±{fr['std_acc']:.4f} | "
                            f"{fr['avg_f1']:.4f}±{fr['std_f1']:.4f} |\n")

            f.write("\n---\n\n## 详细结果\n\n")

            for task, result in results.items():
                if task == 'meta':
                    continue
                task_name = TASK_CONFIG[task]['name']

                if 'regions' in result:
                    for region, rr in result['regions'].items():
                        _write_task_section(f, task_name, region, rr, self.output_dir)
                elif 'final_results' in result:
                    _write_task_section(f, task_name, None, result, self.output_dir)

            f.write("\n---\n\n## 附件说明\n\n")
            f.write("| 文件 | 说明 |\n|---|---|\n")
            f.write("| `config.txt` | 完整配置参数 |\n")
            f.write("| `output.txt` | 完整训练日志 |\n")
            f.write("| `best_model_*.pth` | 最佳微调模型权重 |\n")

        print(f"✓ 生成报告: {report_path}")

    def _config_to_dict(self) -> Dict[str, Any]:
        return {
            attr: getattr(self.config, attr)
            for attr in dir(self.config)
            if attr.isupper() and not attr.startswith('_')
        }


# ==================== 辅助函数 ====================

def _write_task_section(f, task_name: str, region: Optional[str],
                        result: Dict, output_dir: str):
    """写入单个任务/区域的详细结果段落"""
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

    task_safe = task_name.replace(' ', '_').replace('/', '_')
    suffix_safe = f"_{region}" if region else ""
    safe_name = f"{task_safe}{suffix_safe}"

    cm_file = f"{safe_name}_Test_cm.png"
    roc_file = f"{safe_name}_Test_roc.png"

    if os.path.exists(os.path.join(output_dir, cm_file)):
        f.write(f"**混淆矩阵**:\n\n![混淆矩阵]({cm_file})\n\n")
    if os.path.exists(os.path.join(output_dir, roc_file)):
        f.write(f"**ROC 曲线**:\n\n![ROC曲线]({roc_file})\n\n")

    curve_files = [
        fn for fn in os.listdir(output_dir)
        if fn.startswith(f"{task_name}{suffix}_fold") and fn.endswith("_curves.png")
    ]
    if curve_files:
        f.write("**训练曲线**:\n\n")
        for curve_file in sorted(curve_files):
            fold_num = curve_file.split('_fold')[1].split('_')[0]
            f.write(f"Fold {fold_num}:\n\n![Fold{fold_num}训练曲线]({curve_file})\n\n")


def _create_simclr_output_dir(base_dir: str, config: SimCLRExperimentConfig) -> str:
    """
    创建 SimCLR 实验输出目录

    格式: simclr_<backbone>_<arch>_pre<simclr_epochs>_ft<ft_epochs>_<timestamp>
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    arch_short = {
        'single_stream': 'single',
        'mixed_single_stream': 'mixed',
        'dual_stream': 'dual',
    }.get(config.ARCHITECTURE, config.ARCHITECTURE)
    dir_name = (f"simclr_{config.BACKBONE}_{arch_short}"
                f"_pre{config.SIMCLR_EPOCHS}_ft{config.NUM_EPOCHS}_{timestamp}")
    output_dir = os.path.join(base_dir, dir_name)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


# ==================== 公共接口 ====================

def run_single_experiment(
    config: Optional[SimCLRExperimentConfig] = None,
    tasks: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    运行单次 SimCLR 实验（批量实验入口）

    Args:
        config: 实验配置，默认使用 SimCLRExperimentConfig()
        tasks: 任务列表，默认 ['pd', 'stage', 'subtype']
        output_dir: 输出目录，默认自动生成

    Returns:
        实验结果字典（含 meta.duration / meta.output_dir）

    使用示例
    --------
    默认配置:
        >>> result = run_single_experiment()

    自定义 backbone 和 SimCLR 参数:
        >>> cfg = SimCLRExperimentConfig()
        >>> cfg.BACKBONE = 'mobilevit_xxs'
        >>> cfg.SIMCLR_EPOCHS = 150
        >>> cfg.SIMCLR_TEMPERATURE = 0.3
        >>> result = run_single_experiment(cfg)
    """
    if config is None:
        config = SimCLRExperimentConfig()

    runner = SimCLRExperimentRunner(config)
    return runner.run(tasks=tasks, output_dir=output_dir)


# ==================== 命令行入口 ====================


def build_paper_config(exp, device="cpu", num_workers=0):
    """Translate a selected paper recipe without changing its hyperparameters."""
    cfg = SimCLRExperimentConfig()
    cfg.BACKBONE = exp.get('backbone', 'efficientnet_b0')
    cfg.ARCHITECTURE = exp.get('architecture', 'single_stream')
    cfg.PRETRAINED = exp.get('pretrained', True)
    cfg.DROPOUT = exp.get('dropout', 0.3)
    cfg.SIMCLR_EPOCHS = exp.get('simclr_epochs', 100)
    cfg.SIMCLR_LR = exp.get('simclr_lr', 0.001)
    cfg.SIMCLR_BATCH_SIZE = exp.get('simclr_batch', 32)
    cfg.SIMCLR_TEMPERATURE = exp.get('temperature', 0.5)
    cfg.SIMCLR_PROJ_DIM = exp.get('simclr_proj_dim', 128)
    cfg.SIMCLR_PROJ_HIDDEN_DIM = exp.get('simclr_proj_hidden_dim', 256)
    cfg.SIMCLR_WEIGHT_DECAY = exp.get('simclr_weight_decay', 0.0001)
    cfg.SIMCLR_PRETRAIN_DATASET = exp.get('simclr_pretrain_dataset', 'fold')
    cfg.CORN1_ROOT = exp.get('corn1_root', cfg.CORN1_ROOT)
    cfg.CORN1_PRETRAIN_SPLITS = exp.get('corn1_pretrain_splits', cfg.CORN1_PRETRAIN_SPLITS)
    cfg.NUM_EPOCHS = exp.get('ft_epochs', 60)
    cfg.LEARNING_RATE = exp.get('ft_lr', 0.0005)
    cfg.WEIGHT_DECAY = exp.get('weight_decay', 0.001)
    cfg.OPTIMIZER = exp.get('optimizer', 'adamw')
    cfg.LR_SCHEDULER = exp.get('scheduler', 'warmup_cosine')
    cfg.PD_LOSS_TYPE = exp.get('pd_loss_type', 'ce')
    cfg.STAGE_LOSS_TYPE = exp.get('stage_loss', 'label_smooth')
    cfg.SUBTYPE_LOSS_TYPE = exp.get('subtype_loss_type', 'ce')
    cfg.LABEL_SMOOTH_EPS = exp.get('label_smooth_eps', 0.1)
    cfg.KFOLD_MODE = exp.get('kfold_mode', 'patient')
    cfg.DATASET_MODE = exp.get('dataset_mode', 'pd_all')
    cfg.EARLY_STOPPING = exp.get('early_stopping', cfg.EARLY_STOPPING)
    cfg.PATIENCE = exp.get('patience', cfg.PATIENCE)
    cfg.BACKBONE_LR_RATIO = exp.get('backbone_lr_ratio', 0.1)
    cfg.NUM_FOLDS = exp.get('num_folds', cfg.NUM_FOLDS)
    cfg.KFOLD_REPEATS = exp.get('kfold_repeats', cfg.KFOLD_REPEATS)
    cfg.SEED = exp.get('seed', cfg.SEED)
    cfg.NUM_WORKERS = num_workers
    cfg.DEVICE = device
    return cfg


from utils.experiment_reporting import with_patient_report


@with_patient_report
def run_paper_experiment(exp, output_dir, device="cpu", num_workers=0):
    exp = resolve_baseline_tasks(exp)
    cfg = build_paper_config(exp, device, num_workers)
    if exp.get('_transfer') == 'dm':
        return run_single_dm_transfer_experiment(cfg, tasks=exp['_run_tasks'], output_dir=output_dir)
    return run_single_experiment(cfg, tasks=exp['_run_tasks'], output_dir=output_dir)

# ==================== DM workflow ====================



_dm_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _dm_THIS_DIR not in sys.path:
    sys.path.insert(0, _dm_THIS_DIR)
_dm_PROJECT_ROOT = os.path.dirname(os.path.dirname(_dm_THIS_DIR))
if _dm_PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _dm_PROJECT_ROOT)




# ==================== 路径与任务常量 ====================

DM_JSON = 'datasets/dm_dataset_en/dm_patients_dataset_en.json'
PD_PRETRAIN_JSON = 'datasets/pd_dataset_en/patients_dataset_en.json'
PD_PRETRAIN_MODE = 'pd_all'
CORN1_PRETRAIN_CACHE_TAG = 'corn1'

DM_TASKS: List[str] = ['dm_binary', 'dm_subtype']

# DM 原始 group → DM 二分类标签
_dm_DM_SUBTYPE_MAP = {'NDR': 0, 'NPDR': 1}

# 所有 DM 任务都映射到二分类 'pd' 头（num_classes=2, label_key='pd'）
_dm_DM_HEAD_TASK = 'pd'

# pd 自监督预训练使用固定 seed，使所有 DM 实验共享同一份 pd 表征
PRETRAIN_SEED = 42

# backbone 缓存目录（相对 PROJECT_ROOT）
PRETRAIN_CACHE_DIR = 'results/simclr_dm_transfer/pretrain_cache'

# 进程内缓存：签名 -> SimCLRModel，避免同一进程多任务重复加载
_dm_INPROC_PRETRAIN_CACHE: Dict[str, SimCLRModel] = {}


def _dm_json_default(o: Any):
    """让 json.dump 能序列化 numpy 标量/数组（如 int64/float64/ndarray）。"""
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")


# ==================== 路径解析 ====================

def _dm_resolve_project_path(path: str) -> str:
    """把相对路径解析为存在的绝对路径，兼容从 ARIS 根或 codes 启动。"""
    if os.path.isabs(path):
        return path
    candidates = [
        os.path.join(_dm_PROJECT_ROOT, path),
        os.path.join(os.path.dirname(_dm_PROJECT_ROOT), path),
    ]
    for cand in candidates:
        if os.path.exists(cand):
            return cand
    return candidates[0]


def _dm_datasets_base_path(json_abs_path: str) -> str:
    """由数据集 JSON 绝对路径推断 datasets 根目录（image_paths 的拼接基准）。"""
    return os.path.dirname(os.path.dirname(json_abs_path))


# ==================== DM 数据构建 ====================

def build_dm_patients(task: str) -> List[Dict[str, Any]]:
    """构建符合 PD 数据 schema 的 DM 患者列表，使其可直接喂给 ``PDDataset``。

    DM 原始 JSON 的 ``image_paths`` 是扁平列表（无 central/whorl 区分），这里把
    同一张图同时作为 central 与 whorl（duplicate 配对），以适配 dual_stream
    编码器；并通过 ``group`` 字段编码二分类标签（control→0, pd→1），``stage`` 与
    ``subtype`` 置空使其编码为 -1（不参与）。

    参数：
        task: 'dm_binary' 或 'dm_subtype'。

    返回：
        PD-schema 患者列表（每个含 group/stage/subtype/image_paths）。
    """
    if task not in DM_TASKS:
        raise ValueError(f"未知 DM 任务: {task}（应为 {DM_TASKS}）")

    json_abs = _dm_resolve_project_path(DM_JSON)
    with open(json_abs, 'r', encoding='utf-8') as f:
        raw_patients = json.load(f)

    out: List[Dict[str, Any]] = []
    for p in raw_patients:
        group = p.get('group', '')
        if task == 'dm_binary':
            label = 0 if group == 'control' else 1
        else:  # dm_subtype：排除 control，NDR vs NPDR
            if group == 'control':
                continue
            label = _dm_DM_SUBTYPE_MAP.get(group)
            if label is None:
                continue

        imgs = list(p.get('image_paths', []))
        if not imgs:
            continue

        out.append({
            'patient_id': p.get('patient_id', ''),
            'name': p.get('name', ''),
            # group 仅用于 encode_labels 得到二分类 pd_label：1→'pd'，0→'control'
            'group': 'pd' if label == 1 else 'control',
            'stage': 0,        # → stage_label = -1（不参与）
            'subtype': '',     # → subtype_label = -1（不参与）
            'image_paths': {'central': imgs, 'whorl': imgs},
        })
    return out


# ==================== pd 自监督 backbone 复用 ====================

def _dm_pretrain_signature(cfg: SimCLRExperimentConfig, input_size: int) -> str:
    """根据 SimCLR 超参与 backbone 生成缓存签名（不含微调 seed）。"""
    key = {
        'backbone': cfg.BACKBONE,
        'pretrained': str(cfg.PRETRAINED),
        'simclr_epochs': cfg.SIMCLR_EPOCHS,
        'simclr_lr': cfg.SIMCLR_LR,
        'simclr_batch': cfg.SIMCLR_BATCH_SIZE,
        'temperature': cfg.SIMCLR_TEMPERATURE,
        'proj_dim': cfg.SIMCLR_PROJ_DIM,
        'proj_hidden_dim': cfg.SIMCLR_PROJ_HIDDEN_DIM,
        'simclr_wd': cfg.SIMCLR_WEIGHT_DECAY,
        'input_size': input_size,
        'pretrain_source': getattr(cfg, 'SIMCLR_PRETRAIN_DATASET', 'fold'),
        'corn1_root': getattr(cfg, 'CORN1_ROOT', ''),
        'corn1_splits': getattr(cfg, 'CORN1_PRETRAIN_SPLITS', ''),
        'pretrain_json': PD_PRETRAIN_JSON,
        'pretrain_mode': PD_PRETRAIN_MODE,
        'pretrain_seed': PRETRAIN_SEED,
    }
    raw = json.dumps(key, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(raw.encode('utf-8')).hexdigest()[:12]


def _dm_build_simclr_model_from_state(cfg: SimCLRExperimentConfig,
                                   backbone_state: Dict[str, Any]) -> SimCLRModel:
    """重建 SimCLRModel 并载入 backbone 权重（投影头随机初始化，不影响下游）。"""
    model = SimCLRModel(
        cfg.BACKBONE,
        pretrained=False,
        proj_hidden_dim=cfg.SIMCLR_PROJ_HIDDEN_DIM,
        proj_out_dim=cfg.SIMCLR_PROJ_DIM,
    )
    model.get_backbone().load_state_dict(backbone_state)
    return model.to(cfg.DEVICE)


def get_or_build_pd_simclr(cfg: SimCLRExperimentConfig,
                           classifier: SimCLRClassifier,
                           log) -> SimCLRModel:
    """获取在 pd_dataset_en 上预训练好的 SimCLR backbone（复用优先，必要时训练一次）。

    复用优先级：
    1. 环境变量 ``RUN_SIMCLR_PRETRAIN_CKPT`` 指定的 backbone ckpt；
    2. 进程内缓存；
    3. 磁盘缓存（按签名）；
    4. 都没有则在完整 pd_dataset_en 上预训练一次并原子化写盘。
    """
    input_size = classifier._get_input_size()
    sig = _dm_pretrain_signature(cfg, input_size)

    # 1) 显式 ckpt 覆盖
    env_ckpt = os.environ.get('RUN_SIMCLR_PRETRAIN_CKPT', '').strip()
    if env_ckpt:
        ckpt_abs = _dm_resolve_project_path(env_ckpt)
        if os.path.exists(ckpt_abs):
            log(f"[dm-transfer] 复用显式 SimCLR backbone ckpt: {ckpt_abs}")
            state = torch.load(ckpt_abs, map_location=cfg.DEVICE)
            backbone_state = state.get('backbone_state', state) if isinstance(state, dict) else state
            return _dm_build_simclr_model_from_state(cfg, backbone_state)
        log(f"[dm-transfer][warn] RUN_SIMCLR_PRETRAIN_CKPT 指定的文件不存在: {ckpt_abs}")

    # 2) 进程内缓存
    if sig in _dm_INPROC_PRETRAIN_CACHE:
        return _dm_INPROC_PRETRAIN_CACHE[sig]

    # 3) 磁盘缓存
    cache_dir = _dm_resolve_project_path(PRETRAIN_CACHE_DIR)
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{cfg.BACKBONE}_{sig}.pt")
    if os.path.exists(cache_path) and os.path.getsize(cache_path) > 0:
        log(f"[dm-transfer] 复用 pd_dataset_en SimCLR backbone 缓存（跳过预训练）:\n"
            f"             {cache_path}")
        state = torch.load(cache_path, map_location=cfg.DEVICE)
        model = _dm_build_simclr_model_from_state(cfg, state['backbone_state'])
        _dm_INPROC_PRETRAIN_CACHE[sig] = model
        return model

    # 4) 训练一次：默认完整 pd_dataset_en；若配置为 corn1，则使用 CORN-1 图像。
    pretrain_source = str(getattr(cfg, 'SIMCLR_PRETRAIN_DATASET', 'fold')).lower()
    if pretrain_source in ('corn1', 'corn-1'):
        log(f"[dm-transfer] 未找到缓存，开始在 CORN-1 上做一次 SimCLR 预训练"
            f"（backbone={cfg.BACKBONE}, epochs={cfg.SIMCLR_EPOCHS}）...")
        pd_dataset = classifier._resolve_pretrain_dataset(None)
    else:
        log(f"[dm-transfer] 未找到缓存，开始在 pd_dataset_en（{PD_PRETRAIN_MODE}）上做一次 "
            f"SimCLR 预训练（backbone={cfg.BACKBONE}, epochs={cfg.SIMCLR_EPOCHS}）...")
        pd_json_abs = _dm_resolve_project_path(PD_PRETRAIN_JSON)
        pd_base_path = _dm_datasets_base_path(pd_json_abs)
        pd_patients = load_and_filter_dataset(pd_json_abs, PD_PRETRAIN_MODE)
        pd_dataset = PDDataset(
            pd_patients, 'dual_stream',
            classifier.train_transform, base_path=pd_base_path,
        )
        if len(pd_dataset) == 0:
            raise RuntimeError(f"pd 预训练数据集为空: {pd_json_abs}")

    set_seed(PRETRAIN_SEED)
    model = simclr_pretrain(
        backbone_name=cfg.BACKBONE,
        pretrained=cfg.PRETRAINED,
        train_dataset=pd_dataset,
        num_epochs=cfg.SIMCLR_EPOCHS,
        batch_size=cfg.SIMCLR_BATCH_SIZE,
        lr=cfg.SIMCLR_LR,
        temperature=cfg.SIMCLR_TEMPERATURE,
        device=cfg.DEVICE,
        proj_hidden_dim=cfg.SIMCLR_PROJ_HIDDEN_DIM,
        proj_out_dim=cfg.SIMCLR_PROJ_DIM,
        weight_decay=cfg.SIMCLR_WEIGHT_DECAY,
        num_workers=cfg.NUM_WORKERS,
        input_size=input_size,
        verbose=True,
    )

    # 原子化写盘（容忍并发：tmp + replace）
    backbone_state = {k: v.detach().cpu() for k, v in model.get_backbone().state_dict().items()}
    payload = {
        'backbone_state': backbone_state,
        'meta': {
            'backbone': cfg.BACKBONE,
            'signature': sig,
            'pretrain_source': getattr(cfg, 'SIMCLR_PRETRAIN_DATASET', 'fold'),
            'pretrain_json': PD_PRETRAIN_JSON,
            'pretrain_mode': PD_PRETRAIN_MODE,
            'pretrain_seed': PRETRAIN_SEED,
            'simclr_epochs': cfg.SIMCLR_EPOCHS,
            'temperature': cfg.SIMCLR_TEMPERATURE,
            'proj_dim': cfg.SIMCLR_PROJ_DIM,
            'created': datetime.now().isoformat(timespec='seconds'),
        },
    }
    tmp_path = f"{cache_path}.tmp.{os.getpid()}"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, cache_path)
    log(f"[dm-transfer] 已缓存 pd SimCLR backbone: {cache_path}")

    _dm_INPROC_PRETRAIN_CACHE[sig] = model
    return model


# ==================== 单实验运行 ====================

def _dm_aggregate_fold_metrics(fold_metrics: List[Dict[str, float]]) -> Dict[str, float]:
    if not fold_metrics:
        return {}
    accs = [m['acc'] for m in fold_metrics]
    patient_accs = [m.get('patient_acc', 0.0) for m in fold_metrics]
    f1s = [m['f1'] for m in fold_metrics]
    train_accs = [m.get('train_acc', 0.0) for m in fold_metrics]
    return {
        'avg_train_acc': float(np.mean(train_accs)),
        'avg_acc': float(np.mean(accs)),
        'std_acc': float(np.std(accs)),
        'avg_patient_acc': float(np.mean(patient_accs)),
        'std_patient_acc': float(np.std(patient_accs)),
        'avg_f1': float(np.mean(f1s)),
        'std_f1': float(np.std(f1s)),
        'num_folds': len(fold_metrics),
    }


def _dm_run_dm_task_kfold(classifier: SimCLRClassifier,
                       cfg: SimCLRExperimentConfig,
                       task: str) -> Dict[str, Any]:
    """对单个 DM 任务执行 patient-level K-fold（复用二分类 'pd' 头）。"""
    dm_json_abs = _dm_resolve_project_path(DM_JSON)
    dm_base_path = _dm_datasets_base_path(dm_json_abs)

    patients = build_dm_patients(task)
    dataset = PDDataset(
        patients, cfg.ARCHITECTURE,
        classifier.train_transform, base_path=dm_base_path,
    )
    print(f"\n{'='*60}\nDM 任务: {task} | 样本数: {len(dataset)} | "
          f"患者数: {len(set(dataset.patient_ids))}\n{'='*60}")
    if len(dataset) == 0:
        print(f"⚠ DM 任务 {task} 无可用样本，跳过")
        return {}

    splits = list(stratified_kfold(
        dataset.PD_labels, dataset.stage_labels, dataset.subtype_labels,
        n_splits=cfg.NUM_FOLDS, random_state=cfg.SEED,
        kfold_mode=cfg.KFOLD_MODE, patient_ids=dataset.patient_ids,
    ))

    fold_metrics: List[Dict[str, float]] = []
    all_labels: List[int] = []
    all_preds: List[int] = []

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        print(f"\n--- DM {task} Fold {fold_idx + 1}/{cfg.NUM_FOLDS} ---")
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        result = classifier.train(
            train_subset, val_subset, _dm_DM_HEAD_TASK,
            fold_idx=fold_idx + 1, suffix=f"_{task}",
        )
        vr = result['val_result']
        if vr.get('labels'):
            th = result.get('train_history', {})
            patient_labels, patient_ids = task_labels_for_indices(dataset, val_idx, 'pd')
            patient_result = save_fold_patient_outputs(
                cfg.OUTPUT_DIR, _dm_DM_HEAD_TASK, f"_{task}", fold_idx + 1,
                patient_ids, patient_labels,
                probs=vr.get('probs'), logits=vr.get('logits'),
            )
            fold_metrics.append({
                'fold': fold_idx + 1,
                'train_acc': th['train_acc'][-1] if th.get('train_acc') else 0.0,
                'acc': vr['acc'],
                'patient_acc': patient_result['patient_acc'],
                'n_patients': patient_result['n_patients'],
                'patient_logits_json': patient_result['patient_logits_json'],
                'patient_logits_csv': patient_result['patient_logits_csv'],
                'f1': vr['f1'],
            })
            all_labels.extend(vr['labels'])
            all_preds.extend(vr['preds'])
            print(f"  Acc: {vr['acc']:.4f}, PatientAcc: {patient_result['patient_acc']:.4f}, F1: {vr['f1']:.4f}")

    summary = _dm_aggregate_fold_metrics(fold_metrics)
    if summary:
        print(f"\n📊 DM {task}: TrainAcc={summary['avg_train_acc']:.4f}, "
              f"ValAcc={summary['avg_acc']:.4f}±{summary['std_acc']:.4f}, "
              f"PatientAcc={summary.get('avg_patient_acc', 0.0):.4f}±{summary.get('std_patient_acc', 0.0):.4f}, "
              f"ValF1={summary['avg_f1']:.4f}±{summary['std_f1']:.4f}")
    return {
        'fold_results': fold_metrics,
        'final_results': summary,
        'all_predictions': {'labels': all_labels, 'preds': all_preds},
    }


def run_single_dm_transfer_experiment(cfg: SimCLRExperimentConfig,
                                      tasks: Optional[List[str]] = None,
                                      output_dir: Optional[str] = None) -> Dict[str, Any]:
    """运行单组 DM 迁移实验：复用 pd SimCLR backbone，仅微调下游头。

    参数：
        cfg: SimCLR 实验配置（由批量 runner 注入超参）。
        tasks: DM 任务列表，默认 ``DM_TASKS``。
        output_dir: 输出目录（由批量 runner 提供）。

    返回：
        结果字典（每个 DM 任务的 K-fold 汇总）。
    """
    tasks = [t for t in (tasks or DM_TASKS) if t in DM_TASKS] or list(DM_TASKS)

    output_dir = output_dir or os.path.join(
        _dm_resolve_project_path('results/simclr_dm_transfer'),
        f"dm_transfer_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    os.makedirs(output_dir, exist_ok=True)
    cfg.OUTPUT_DIR = output_dir
    save_config(cfg, output_dir)

    log_file = open(os.path.join(output_dir, 'output.txt'), 'w', encoding='utf-8')
    original_stdout = sys.stdout
    sys.stdout = Tee(original_stdout, log_file)

    start_time = time.time()
    results: Dict[str, Any] = {}
    try:
        print("=" * 60)
        print("SimCLR → DM 迁移实验（复用 pd_dataset_en 自监督表征，仅微调下游头）")
        print("=" * 60)
        print(f"Backbone={cfg.BACKBONE} | Arch={cfg.ARCHITECTURE} | "
              f"Pretrained={cfg.PRETRAINED}")
        print(f"微调: epochs={cfg.NUM_EPOCHS}, lr={cfg.LEARNING_RATE}, "
              f"wd={cfg.WEIGHT_DECAY}, backbone_lr_ratio={cfg.BACKBONE_LR_RATIO}, "
              f"pd_loss={cfg.PD_LOSS_TYPE}")
        print(f"K-fold: {cfg.NUM_FOLDS} ({cfg.KFOLD_MODE}) | seed={cfg.SEED} | "
              f"tasks={tasks}")

        classifier = SimCLRClassifier(cfg)

        # 复用 pd 自监督 backbone（不重新训练）
        simclr_model = get_or_build_pd_simclr(cfg, classifier, log=print)
        classifier.precomputed_simclr = simclr_model

        # 复原微调阶段随机性（pd 预训练可能改过全局 seed）
        set_seed(cfg.SEED)

        for task in tasks:
            results[task] = _dm_run_dm_task_kfold(classifier, cfg, task)

        _dm_write_dm_report(output_dir, cfg, results, tasks)
    finally:
        sys.stdout = original_stdout
        log_file.close()

    duration = time.time() - start_time
    results['meta'] = {
        'output_dir': output_dir,
        'duration': duration,
        'tasks': tasks,
        'backbone': cfg.BACKBONE,
        'seed': cfg.SEED,
    }
    with open(os.path.join(output_dir, 'results.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=_dm_json_default)

    print(f"DM 迁移实验完成（{cfg.BACKBONE}, seed={cfg.SEED}），用时 "
          f"{duration / 60:.1f}min，结果: {output_dir}")
    return results


def _dm_write_dm_report(output_dir: str, cfg: SimCLRExperimentConfig,
                     results: Dict[str, Any], tasks: List[str]) -> None:
    path = os.path.join(output_dir, 'experiment_report.md')
    with open(path, 'w', encoding='utf-8') as f:
        f.write("# SimCLR → DM 迁移实验报告\n\n")
        f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("复用 `pd_dataset_en` 上的 SimCLR 自监督 backbone，仅在 "
                "`dm_dataset_en` 上微调下游二分类头。\n\n")
        f.write("## 配置\n\n| 参数 | 值 |\n|---|---|\n")
        f.write(f"| Backbone | `{cfg.BACKBONE}` |\n")
        f.write(f"| 架构 | `{cfg.ARCHITECTURE}` |\n")
        f.write(f"| ImageNet 预训练 | `{cfg.PRETRAINED}` |\n")
        f.write(f"| SimCLR 预训练源 | `{getattr(cfg, 'SIMCLR_PRETRAIN_DATASET', 'fold')}` |\n")
        f.write(f"| CORN-1 路径 | `{getattr(cfg, 'CORN1_ROOT', '')}` |\n")
        f.write(f"| CORN-1 splits | `{getattr(cfg, 'CORN1_PRETRAIN_SPLITS', '')}` |\n")
        f.write(f"| SimCLR Epochs/T/dim | {cfg.SIMCLR_EPOCHS} / "
                f"{cfg.SIMCLR_TEMPERATURE} / {cfg.SIMCLR_PROJ_DIM} |\n")
        f.write(f"| 微调 Epochs/LR/WD | {cfg.NUM_EPOCHS} / {cfg.LEARNING_RATE} / "
                f"{cfg.WEIGHT_DECAY} |\n")
        f.write(f"| Backbone LR 倍率 | {cfg.BACKBONE_LR_RATIO} |\n")
        f.write(f"| 微调损失 | `{cfg.PD_LOSS_TYPE}` (eps={cfg.LABEL_SMOOTH_EPS}) |\n")
        f.write(f"| K-fold | {cfg.NUM_FOLDS} ({cfg.KFOLD_MODE}) |\n")
        f.write(f"| Seed | {cfg.SEED} |\n\n")

        f.write("## 结果摘要\n\n| DM 任务 | Patient Accuracy | Image Accuracy | F1 |\n|---|---|---|---|\n")
        for task in tasks:
            fr = results.get(task, {}).get('final_results', {})
            if fr:
                f.write(
                    f"| {task} | {fr.get('avg_patient_acc', 0.0):.4f}±{fr.get('std_patient_acc', 0.0):.4f} | "
                    f"{fr['avg_acc']:.4f}±{fr['std_acc']:.4f} | "
                    f"{fr['avg_f1']:.4f}±{fr['std_f1']:.4f} |\n"
                )
            else:
                f.write(f"| {task} | - | - | - |\n")
        f.write("\n")
