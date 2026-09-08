"""
main.py - 原型学习实验主程序

模块职责
--------
执行单组配置下的完整 K-fold 交叉验证实验（小样本原型学习 Episodic Training）:
1. 加载数据并生成 K-fold 划分
2. 每个 fold 执行 Episodic 训练（支持集/查询集采样）
3. 汇总指标 (Acc, F1, AUC)
4. 生成可视化 (训练曲线、混淆矩阵、ROC)
5. 输出实验报告 (Markdown 格式)

支持 Backbone
-------------
- densenet121: 1024 维，来自 basic_models（推荐首选）
- resnet18: 512 维，来自 basic_models
- efficientnet_b0: 1280 维，来自 light_models
- vanillanet_5: 4096 维，来自 light_models

接口设计
--------
输入:
    ProtoExperimentConfig: 继承自 ProtoClassifierConfig，添加实验管理字段

输出:
    - results/<proto_backbone_arch_timestamp>/
        - config.txt: 配置记录
        - output.txt: 训练日志
        - experiment_report.md: 实验报告
        - *.png: 可视化图表
        - best_model_*.pth: 最佳模型权重

批量调用示例
------------
    >>> from main import run_single_experiment, ProtoExperimentConfig
    >>>
    >>> cfg = ProtoExperimentConfig()
    >>> cfg.BACKBONE = 'densenet121'
    >>> cfg.N_SHOT = 5
    >>> result = run_single_experiment(cfg)
"""
import sys
import os
import time
import json
import torch
import numpy as np
from datetime import datetime
from collections import defaultdict
from typing import Literal, List, Dict, Any, Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from core.prototypical_classifier import (
    FewShotPrototypicalClassifier,
    ProtoClassifierConfig,
    TASK_CONFIG,
)
from core.data_loader import PDDataset, load_and_filter_dataset, stratified_kfold
from utils.patient_outputs import save_fold_patient_outputs, task_labels_for_indices
from utils.misc import set_seed, Tee, create_output_dir, save_config
from utils.visualize import plot_confusion_matrix, plot_roc_auc, plot_training_curves


# ==================== 实验配置 ====================

class ProtoExperimentConfig(ProtoClassifierConfig):
    """
    原型学习实验配置类（继承自 ProtoClassifierConfig）

    新增字段（实验管理）:
        NUM_FOLDS: K-fold 折数
        SEED: 随机种子
        KFOLD_MODE: K-fold 划分模式 ('patient' / 'image')
        DATASET_MODE: 数据集模式
        DATASET_JSON: 数据集 JSON 路径
        OUTPUT_DIR: 输出目录
        VANILLANET_INIT_ACT_NUM: VanillaNet 激活初始分支数
    """
    ARCHITECTURE: Literal['single_stream', 'mixed_single_stream', 'dual_stream'] = 'single_stream'
    BACKBONE: Literal[
        'densenet121', 'resnet18', 'efficientnet_b0', 'vanillanet_5',
        'mobilevit_xxs', 'mobilevit_xs',
    ] = 'densenet121'
    PRETRAINED: Literal[True, False, 'Half'] = True

    EMBEDDING_DIM: Literal[128, 256] = 128
    FEATURE_DROPOUT: float = 0.2

    N_WAY: int = 3
    N_SHOT: int = 3
    N_QUERY: int = 4
    TRAIN_EPISODES_PER_EPOCH: int = 80
    VAL_EPISODES: int = 80

    NUM_EPOCHS: int = 40
    LEARNING_RATE: float = 5e-4
    WEIGHT_DECAY: float = 1e-4
    LR_SCHEDULER: str = 'cosine'

    EARLY_STOPPING: bool = True
    PATIENCE: int = 8

    DATA_AUGMENTATION: bool = True
    AUG_HORIZONTAL_FLIP: float = 0.5
    AUG_ROTATION: float = 10.0
    AUG_BRIGHTNESS: float = 0.2
    AUG_CONTRAST: float = 0.2
    AUG_SATURATION: float = 0.0
    AUG_HUE: float = 0.0

    BATCH_SIZE_INFERENCE: int = 64
    NUM_WORKERS: int = 4

    DATASET_MODE: Literal['pd_control', 'pd_control_cataract', 'pd_all'] = 'pd_all'
    DATASET_JSON: str = 'datasets/pd_dataset_en/patients_dataset_en.json'

    NUM_FOLDS: int = 5
    KFOLD_REPEATS: int = 1
    SEED: int = 42
    KFOLD_MODE: Literal['patient', 'image'] = 'patient'

    VANILLANET_INIT_ACT_NUM: int = 3

    DEVICE: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    OUTPUT_DIR: str = 'results'


# ==================== 实验运行器 ====================

class ProtoExperimentRunner:
    """
    原型学习实验运行器

    执行一组配置下的完整 K-fold 实验（Episodic 训练）。

    使用方法
    --------
    >>> config = ProtoExperimentConfig()
    >>> config.BACKBONE = 'densenet121'
    >>> config.N_SHOT = 5
    >>> runner = ProtoExperimentRunner(config)
    >>> results = runner.run(tasks=['pd', 'stage'])
    """

    def __init__(self, config: ProtoExperimentConfig):
        self.config = config
        self.classifier = FewShotPrototypicalClassifier(config)
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
        """运行完整 K-fold 实验"""
        set_seed(self.config.SEED)
        start_time = time.time()

        if tasks is None:
            tasks = ['pd', 'stage', 'subtype']

        self.output_dir = output_dir or _create_proto_output_dir('results', self.config)
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
                if task == 'pd_subtype_hierarchy':
                    results[task] = self._run_subtype_hierarchy(dataset)
                else:
                    results[task] = self._run_kfold(dataset, task)

        self._generate_report(results)
        return results

    def _run_kfold(self, dataset, task: str, suffix: str = '') -> Dict[str, Any]:
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
                fold_idx=fold_idx + 1, suffix=suffix,
            )

            plot_training_curves(
                result['train_history'],
                task_cfg['name'] + suffix,
                fold_idx + 1, self.output_dir,
            )

            val_result = result['val_result']

            if val_result.get('labels'):
                patient_pred = self.classifier.predict_with_prototypes(
                    result['model'], train_subset, val_subset, task
                )
                patient_labels, patient_ids = task_labels_for_indices(dataset, val_idx, task)
                patient_result = save_fold_patient_outputs(
                    self.output_dir, task, suffix, fold_idx + 1,
                    patient_ids, patient_labels,
                    probs=patient_pred.get('probs'), logits=patient_pred.get('logits')
                )
                fold_results.append({
                    'fold': fold_idx + 1,
                    'train_acc': result['train_history']['train_acc'][-1]
                        if result['train_history']['train_acc'] else 0.0,
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
                'acc': val_result.get('acc', 0.0),
            })

        if not fold_results:
            return {}

        summary = self._summarize_results(fold_results, all_preds, task_cfg, suffix)

        best = max(best_models, key=lambda x: x['acc'])
        model_path = os.path.join(self.output_dir, f"best_model_{task}{suffix}.pth")
        torch.save(best['state'], model_path)
        print(f"✓ 保存最佳模型: {model_path}")

        return summary

    def _run_subtype_hierarchy(self, dataset) -> Dict[str, Any]:
        """按 TD-vs-rest -> PIGD-vs-Mixed 层级协议计算 patient-level 三分类准确率。"""
        config = self.config
        full_indices = [i for i, y in enumerate(dataset.subtype_labels) if int(y) >= 0]
        if not full_indices:
            print("⚠ 无 subtype 标签样本，跳过 pd_subtype_hierarchy")
            return {}

        full_pd = [1 for _ in full_indices]
        full_stage = [-1 for _ in full_indices]
        full_subtype = [int(dataset.subtype_labels[i]) for i in full_indices]
        full_patient_ids = [int(dataset.patient_ids[i]) for i in full_indices]

        fold_results: List[Dict[str, Any]] = []
        prediction_rows: List[Dict[str, Any]] = []
        total_folds = config.NUM_FOLDS * config.KFOLD_REPEATS

        print(
            f"层级 subtype 协议: patient {config.NUM_FOLDS}-fold x "
            f"{config.KFOLD_REPEATS} repeats，共 {total_folds} folds"
        )

        for repeat in range(config.KFOLD_REPEATS):
            splits = list(stratified_kfold(
                full_pd, full_stage, full_subtype,
                n_splits=config.NUM_FOLDS,
                random_state=config.SEED + repeat * 1000,
                kfold_mode='patient',
                patient_ids=full_patient_ids,
            ))

            for fold_idx, (train_local_idx, val_local_idx) in enumerate(splits, start=1):
                print(f"\n--- Hierarchy Repeat {repeat} Fold {fold_idx}/{config.NUM_FOLDS} ---")
                train_idx = [full_indices[int(i)] for i in train_local_idx]
                val_idx = [full_indices[int(i)] for i in val_local_idx]
                train_pm_idx = [i for i in train_idx if int(dataset.subtype_labels[i]) in (1, 2)]
                val_pm_idx = [i for i in val_idx if int(dataset.subtype_labels[i]) in (1, 2)]

                td_train = torch.utils.data.Subset(dataset, train_idx)
                full_val = torch.utils.data.Subset(dataset, val_idx)
                pm_train = torch.utils.data.Subset(dataset, train_pm_idx)
                pm_val = torch.utils.data.Subset(dataset, val_pm_idx)

                td_result = self.classifier.train(
                    td_train, full_val, 'pd_subtype_tdrest',
                    fold_idx=repeat * config.NUM_FOLDS + fold_idx,
                    suffix='_hier_tdrest',
                )
                pm_result = self.classifier.train(
                    pm_train, pm_val, 'pd_subtype_pigd_mixed',
                    fold_idx=repeat * config.NUM_FOLDS + fold_idx,
                    suffix='_hier_pigd_mixed',
                )

                td_pred = self.classifier.predict_with_prototypes(
                    td_result['model'], td_train, full_val, 'pd_subtype_tdrest'
                )
                pm_pred = self.classifier.predict_with_prototypes(
                    pm_result['model'], pm_train, full_val, 'pd_subtype_pigd_mixed'
                )

                fold_metrics, fold_rows = self._compute_subtype_hierarchy_fold(
                    dataset, val_idx, td_pred, pm_pred, repeat, fold_idx
                )
                fold_results.append(fold_metrics)
                prediction_rows.extend(fold_rows)

                print(
                    f"  HierAcc: {fold_metrics['subtype_hierarchy_acc']:.4f}, "
                    f"BalAcc: {fold_metrics['subtype_hierarchy_balanced_acc']:.4f}, "
                    f"TDRest: {fold_metrics['tdrest_acc']:.4f}, "
                    f"PIGD/Mixed: {fold_metrics['pigd_mixed_acc']:.4f}"
                )

        summary = self._summarize_subtype_hierarchy(fold_results)
        pred_path = os.path.join(self.output_dir, 'pd_subtype_hierarchy_predictions.csv')
        json_path = os.path.join(self.output_dir, 'pd_subtype_hierarchy_summary.json')
        self._write_hierarchy_predictions_csv(pred_path, prediction_rows)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"✓ 保存 hierarchy 预测: {pred_path}")
        print(f"✓ 保存 hierarchy 汇总: {json_path}")

        return {
            'fold_results': fold_results,
            'final_results': {
                'avg_acc': summary['subtype_hierarchy_acc_mean'],
                'std_acc': summary['subtype_hierarchy_acc_std'],
                'avg_f1': summary['subtype_hierarchy_balanced_acc_mean'],
                'std_f1': summary['subtype_hierarchy_balanced_acc_std'],
                'tdrest_acc_mean': summary['tdrest_acc_mean'],
                'tdrest_acc_std': summary['tdrest_acc_std'],
                'pigd_mixed_acc_mean': summary['pigd_mixed_acc_mean'],
                'pigd_mixed_acc_std': summary['pigd_mixed_acc_std'],
                'auc_scores': {},
            },
            'hierarchy_summary': summary,
            'prediction_csv': pred_path,
        }

    def _compute_subtype_hierarchy_fold(
        self,
        dataset,
        val_idx: List[int],
        td_pred: Dict[str, Any],
        pm_pred: Dict[str, Any],
        repeat: int,
        fold: int,
    ) -> tuple:
        pids = [int(dataset.patient_ids[i]) for i in val_idx]
        true_subtypes = [int(dataset.subtype_labels[i]) for i in val_idx]
        td_by_pid = self._aggregate_patient_predictions(pids, td_pred, positive_class=1)
        pm_by_pid = self._aggregate_patient_predictions(pids, pm_pred, positive_class=1)

        rows: List[Dict[str, Any]] = []
        true_3cls, pred_3cls = [], []
        td_true, td_hat = [], []
        pm_true, pm_hat = [], []

        for pid in sorted(set(pids)):
            idx = pids.index(pid)
            y = int(true_subtypes[idx])
            gate = int(td_by_pid[pid]['pred'])
            pm = int(pm_by_pid[pid]['pred'])
            final_pred = 0 if gate == 0 else (1 if pm == 0 else 2)

            true_3cls.append(y)
            pred_3cls.append(final_pred)
            td_true.append(0 if y == 0 else 1)
            td_hat.append(gate)
            if y in (1, 2):
                pm_true.append(0 if y == 1 else 1)
                pm_hat.append(pm)

            rows.append({
                'repeat': repeat,
                'fold': fold,
                'patient_id': pid,
                'true_subtype': y,
                'tdrest_pred': gate,
                'tdrest_prob_rest': td_by_pid[pid]['prob_positive'],
                'pigd_mixed_pred': pm,
                'pigd_mixed_prob_mixed': pm_by_pid[pid]['prob_positive'],
                'final_pred': final_pred,
            })

        per_class_3 = self._per_class_recall(true_3cls, pred_3cls, [0, 1, 2])
        td_per_class = self._per_class_recall(td_true, td_hat, [0, 1])
        pm_per_class = self._per_class_recall(pm_true, pm_hat, [0, 1])
        cm_3 = self._confusion_matrix(true_3cls, pred_3cls, [0, 1, 2])

        metrics = {
            'repeat': repeat,
            'fold': fold,
            'n_val_full': len(set(pids)),
            'n_val_td': sum(1 for y in true_3cls if y == 0),
            'n_val_rest': sum(1 for y in true_3cls if y in (1, 2)),
            'n_val_pigd_mixed': len(pm_true),
            'tdrest_acc': float(np.mean(np.array(td_true) == np.array(td_hat))) if td_true else 0.0,
            'tdrest_per_class_acc': td_per_class,
            'pigd_mixed_acc': float(np.mean(np.array(pm_true) == np.array(pm_hat))) if pm_true else 0.0,
            'pigd_mixed_per_class_acc': pm_per_class,
            'subtype_hierarchy_acc': float(np.mean(np.array(true_3cls) == np.array(pred_3cls))) if true_3cls else 0.0,
            'subtype_hierarchy_balanced_acc': float(np.mean(list(per_class_3.values()))) if per_class_3 else 0.0,
            'subtype_hierarchy_per_class_acc': per_class_3,
            'confusion_matrix_3cls': cm_3,
        }
        return metrics, rows

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

    @staticmethod
    def _per_class_recall(labels: List[int], preds: List[int], classes: List[int]) -> Dict[str, float]:
        out = {}
        for cls in classes:
            total = sum(1 for y in labels if y == cls)
            correct = sum(1 for y, p in zip(labels, preds) if y == cls and p == cls)
            out[str(cls)] = float(correct / total) if total else 0.0
        return out

    @staticmethod
    def _confusion_matrix(labels: List[int], preds: List[int], classes: List[int]) -> List[List[int]]:
        pos = {cls: i for i, cls in enumerate(classes)}
        cm = [[0 for _ in classes] for _ in classes]
        for y, p in zip(labels, preds):
            if y in pos and p in pos:
                cm[pos[y]][pos[p]] += 1
        return cm

    def _summarize_subtype_hierarchy(self, fold_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        def mean_std(key: str) -> tuple:
            vals = [float(r[key]) for r in fold_results]
            return float(np.mean(vals) if vals else 0.0), float(np.std(vals) if vals else 0.0)

        td_mean, td_std = mean_std('tdrest_acc')
        pm_mean, pm_std = mean_std('pigd_mixed_acc')
        h_mean, h_std = mean_std('subtype_hierarchy_acc')
        hb_mean, hb_std = mean_std('subtype_hierarchy_balanced_acc')

        return {
            'protocol': 'patient_5fold_x3_subtype_hierarchy',
            'num_folds': self.config.NUM_FOLDS,
            'kfold_repeats': self.config.KFOLD_REPEATS,
            'evaluated_folds': len(fold_results),
            'tdrest_acc_mean': td_mean,
            'tdrest_acc_std': td_std,
            'pigd_mixed_acc_mean': pm_mean,
            'pigd_mixed_acc_std': pm_std,
            'subtype_hierarchy_acc_mean': h_mean,
            'subtype_hierarchy_acc_std': h_std,
            'subtype_hierarchy_balanced_acc_mean': hb_mean,
            'subtype_hierarchy_balanced_acc_std': hb_std,
            'fold_results': fold_results,
        }

    @staticmethod
    def _write_hierarchy_predictions_csv(path: str, rows: List[Dict[str, Any]]):
        fields = [
            'repeat', 'fold', 'patient_id', 'true_subtype', 'tdrest_pred',
            'tdrest_prob_rest', 'pigd_mixed_pred', 'pigd_mixed_prob_mixed', 'final_pred',
        ]
        with open(path, 'w', encoding='utf-8') as f:
            f.write(','.join(fields) + '\n')
            for row in rows:
                f.write(','.join(str(row.get(field, '')) for field in fields) + '\n')

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
        print("帕金森病图像分类实验 [小样本原型学习]")
        print("="*60)
        print(f"架构: {cfg.ARCHITECTURE} | Backbone: {cfg.BACKBONE}")
        print(f"预训练权重: {cfg.PRETRAINED} | Embedding Dim: {cfg.EMBEDDING_DIM}")
        print(f"Episode: {cfg.N_WAY}-way {cfg.N_SHOT}-shot {cfg.N_QUERY}-query")
        print(f"训练轮次/epoch: {cfg.TRAIN_EPISODES_PER_EPOCH} | "
              f"验证轮次: {cfg.VAL_EPISODES}")
        print(f"数据集: {cfg.DATASET_MODE} | K-fold: {cfg.NUM_FOLDS} ({cfg.KFOLD_MODE})")
        print(f"Epochs: {cfg.NUM_EPOCHS} | LR: {cfg.LEARNING_RATE} | "
              f"WeightDecay: {cfg.WEIGHT_DECAY}")
        print("="*60)

    def _generate_report(self, results: Dict):
        """生成 Markdown 实验报告"""
        report_path = os.path.join(self.output_dir, 'experiment_report.md')
        cfg = self.config

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 原型学习实验报告\n\n")
            f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("## 实验配置\n\n")
            f.write("| 参数 | 值 |\n|---|---|\n")
            f.write(f"| 架构 | `{cfg.ARCHITECTURE}` |\n")
            f.write(f"| Backbone | `{cfg.BACKBONE}` |\n")
            f.write(f"| ImageNet 预训练 | `{cfg.PRETRAINED}` |\n")
            f.write(f"| Embedding 维度 | {cfg.EMBEDDING_DIM} |\n")
            f.write(f"| Feature Dropout | {cfg.FEATURE_DROPOUT} |\n")
            f.write(f"| Episode | {cfg.N_WAY}-way {cfg.N_SHOT}-shot {cfg.N_QUERY}-query |\n")
            f.write(f"| 训练 Episodes/epoch | {cfg.TRAIN_EPISODES_PER_EPOCH} |\n")
            f.write(f"| 验证 Episodes | {cfg.VAL_EPISODES} |\n")
            f.write(f"| 数据集 | `{cfg.DATASET_MODE}` |\n")
            f.write(f"| K-fold | {cfg.NUM_FOLDS} ({cfg.KFOLD_MODE}) |\n")
            f.write(f"| Epochs | {cfg.NUM_EPOCHS} |\n")
            f.write(f"| LR | {cfg.LEARNING_RATE} |\n")
            f.write(f"| WeightDecay | {cfg.WEIGHT_DECAY} |\n")
            f.write(f"| LR Scheduler | `{cfg.LR_SCHEDULER}` |\n\n")

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
                        _write_task_section(f, task_name, region, rr, self.output_dir)
                elif 'final_results' in result:
                    _write_task_section(f, task_name, None, result, self.output_dir)

            f.write("\n---\n\n## 附件说明\n\n")
            f.write("| 文件 | 说明 |\n|---|---|\n")
            f.write("| `config.txt` | 完整配置参数 |\n")
            f.write("| `output.txt` | 完整训练日志 |\n")
            f.write("| `best_model_*.pth` | 最佳模型权重 |\n")

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


def _create_proto_output_dir(base_dir: str, config: ProtoExperimentConfig) -> str:
    """
    创建原型学习实验输出目录

    格式: proto_<backbone>_<arch>_<nway>way<nshot>shot_ep<epochs>_<timestamp>
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    arch_short = {
        'single_stream': 'single',
        'mixed_single_stream': 'mixed',
        'dual_stream': 'dual',
    }.get(config.ARCHITECTURE, config.ARCHITECTURE)
    dir_name = (f"proto_{config.BACKBONE}_{arch_short}"
                f"_{config.N_WAY}way{config.N_SHOT}shot"
                f"_ep{config.NUM_EPOCHS}_{timestamp}")
    output_dir = os.path.join(base_dir, dir_name)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


# ==================== 公共接口 ====================

def run_single_experiment(
    config: Optional[ProtoExperimentConfig] = None,
    tasks: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    运行单次原型学习实验

    Args:
        config: 实验配置，默认使用 ProtoExperimentConfig()
        tasks: 任务列表，默认 ['pd', 'stage', 'subtype']
        output_dir: 输出目录，默认自动生成

    Returns:
        实验结果字典（含 meta.duration / meta.output_dir）

    使用示例
    --------
    默认配置:
        >>> result = run_single_experiment()

    自定义 backbone 和 episode 参数:
        >>> cfg = ProtoExperimentConfig()
        >>> cfg.BACKBONE = 'resnet18'
        >>> cfg.N_SHOT = 5
        >>> result = run_single_experiment(cfg)
    """
    if config is None:
        config = ProtoExperimentConfig()

    runner = ProtoExperimentRunner(config)
    return runner.run(tasks=tasks, output_dir=output_dir)


# ==================== 命令行入口 ====================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='原型学习实验')
    parser.add_argument('--backbone', default='resnet18',
                        choices=['resnet18', 'efficientnet_b0', 'densenet121', 'vanillanet_5'])
    parser.add_argument('--arch', default='dual_stream',
                        choices=['single_stream', 'mixed_single_stream', 'dual_stream'])
    parser.add_argument('--pretrained', default='Half',
                        choices=['True', 'False', 'Half'])
    parser.add_argument('--n-way', type=int, default=3)
    parser.add_argument('--n-shot', type=int, default=3)
    parser.add_argument('--n-query', type=int, default=4)
    parser.add_argument('--embedding-dim', type=int, default=128, choices=[128, 256])
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--dataset-mode', default='pd_all',
                        choices=['pd_control', 'pd_control_cataract', 'pd_all'])
    parser.add_argument('--tasks', nargs='+', default=['pd', 'stage', 'subtype'])
    parser.add_argument('--output-dir', default=None)
    args = parser.parse_args()

    _pretrained_map = {'True': True, 'False': False, 'Half': 'Half'}

    cfg = ProtoExperimentConfig()
    cfg.BACKBONE = args.backbone
    cfg.ARCHITECTURE = args.arch
    cfg.PRETRAINED = _pretrained_map[args.pretrained]
    cfg.N_WAY = args.n_way
    cfg.N_SHOT = args.n_shot
    cfg.N_QUERY = args.n_query
    cfg.EMBEDDING_DIM = args.embedding_dim
    cfg.NUM_EPOCHS = args.epochs
    cfg.LEARNING_RATE = args.lr
    cfg.DATASET_MODE = args.dataset_mode

    result = run_single_experiment(cfg, tasks=args.tasks, output_dir=args.output_dir)
    print(f"\n完成。输出目录: {result['meta']['output_dir']}")
