"""prototype 实验实现：训练、评估及单组执行；参数与批量控制见 batch_experiments.py。"""

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from experiments._execution import single_main
    single_main("experiments.prototype_experiments.main")
    raise SystemExit(0)

from utils.td_rest import resolve_baseline_tasks, td_rest_rows, td_rest_fold, save_td_rest_results
import sys
import os
import time
import torch
import numpy as np
from datetime import datetime
from collections import defaultdict
from typing import Literal, List, Dict, Any, Optional
from core.prototypical_classifier import FewShotPrototypicalClassifier, ProtoClassifierConfig, TASK_CONFIG
from core.data_loader import PDDataset, load_and_filter_dataset, stratified_kfold
from utils.patient_outputs import save_fold_patient_outputs, task_labels_for_indices
from utils.misc import set_seed, Tee, save_config
from utils.visualize import plot_confusion_matrix, plot_roc_auc, plot_training_curves


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)



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
    """
    ARCHITECTURE: Literal['single_stream', 'mixed_single_stream', 'dual_stream'] = 'single_stream'
    BACKBONE: Literal['resnet18'] = 'resnet18'
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
                if task == 'pd_subtype_tdrest':
                    results[task] = self._run_subtype_td_rest(dataset)
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

    def _run_subtype_td_rest(self, dataset) -> Dict[str, Any]:
        """按 TD-vs-rest 二分类协议计算 patient-level 准确率。"""
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

        for repeat in range(config.KFOLD_REPEATS):
            splits = list(stratified_kfold(
                full_pd, full_stage, full_subtype,
                n_splits=config.NUM_FOLDS,
                random_state=config.SEED + repeat * 1000,
                kfold_mode='patient',
                patient_ids=full_patient_ids,
            ))

            for fold_idx, (train_local_idx, val_local_idx) in enumerate(splits, start=1):
                print(f"\n--- TD-rest Repeat {repeat} Fold {fold_idx}/{config.NUM_FOLDS} ---")
                train_idx = [full_indices[int(i)] for i in train_local_idx]
                val_idx = [full_indices[int(i)] for i in val_local_idx]

                td_train = torch.utils.data.Subset(dataset, train_idx)
                full_val = torch.utils.data.Subset(dataset, val_idx)

                td_result = self.classifier.train(
                    td_train, full_val, 'pd_subtype_tdrest',
                    fold_idx=repeat * config.NUM_FOLDS + fold_idx,
                    suffix='_td_rest',
                )

                td_pred = self.classifier.predict_with_prototypes(
                    td_result['model'], td_train, full_val, 'pd_subtype_tdrest'
                )

                pids = [int(dataset.patient_ids[i]) for i in val_idx]
                by_pid = self._aggregate_patient_predictions(pids, td_pred, positive_class=1)
                patient_map = {int(dataset.patient_ids[i]): {'subtype': ('TD', 'PIGD', 'mixed')[int(dataset.subtype_labels[i])]} for i in val_idx}
                rows = td_rest_rows(patient_map, sorted(set(pids)),
                                    {pid: v['pred'] for pid, v in by_pid.items()},
                                    {pid: v['prob_positive'] for pid, v in by_pid.items()}, repeat, fold_idx)
                fold_results.append(td_rest_fold(rows))
                prediction_rows.extend(rows)

        return save_td_rest_results(self.output_dir, fold_results, prediction_rows,
                                    config.NUM_FOLDS, config.KFOLD_REPEATS, config.SEED,
                                    prefix='pd_subtype' + '')


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


def build_paper_config(exp, device="cpu", num_workers=0):
    """Translate a selected paper recipe without changing its hyperparameters."""
    cfg = ProtoExperimentConfig()
    cfg.BACKBONE = exp.get('backbone', 'densenet121')
    cfg.ARCHITECTURE = exp.get('architecture', 'single_stream')
    cfg.PRETRAINED = exp.get('pretrained', True)
    cfg.EMBEDDING_DIM = exp.get('embedding_dim', 128)
    cfg.FEATURE_DROPOUT = exp.get('feature_dropout', 0.2)
    cfg.N_WAY = exp.get('n_way', 3)
    cfg.N_SHOT = exp.get('n_shot', 3)
    cfg.N_QUERY = exp.get('n_query', 4)
    cfg.TRAIN_EPISODES_PER_EPOCH = exp.get('train_episodes_per_epoch', 80)
    cfg.VAL_EPISODES = exp.get('val_episodes', 80)
    cfg.NUM_EPOCHS = exp.get('num_epochs', 40)
    cfg.LEARNING_RATE = exp.get('learning_rate', 0.0005)
    cfg.WEIGHT_DECAY = exp.get('weight_decay', 0.0001)
    cfg.LR_SCHEDULER = exp.get('lr_scheduler', 'cosine')
    cfg.KFOLD_MODE = exp.get('kfold_mode', 'patient')
    cfg.PATIENCE = exp.get('patience', cfg.PATIENCE)
    cfg.NUM_FOLDS = exp.get('num_folds', cfg.NUM_FOLDS)
    cfg.KFOLD_REPEATS = exp.get('kfold_repeats', cfg.KFOLD_REPEATS)
    cfg.SEED = exp.get('seed', cfg.SEED)
    cfg.DATASET_MODE = exp.get('dataset_mode', cfg.DATASET_MODE)
    cfg.DATASET_JSON = exp.get('dataset_json', cfg.DATASET_JSON)
    cfg.NUM_WORKERS = num_workers
    cfg.DEVICE = device
    return cfg


def run_paper_experiment(exp, output_dir, device="cpu", num_workers=0):
    exp = resolve_baseline_tasks(exp)
    cfg = build_paper_config(exp, device, num_workers)
    return run_single_experiment(cfg, tasks=exp['_run_tasks'], output_dir=output_dir)
