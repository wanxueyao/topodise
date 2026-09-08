"""
DM 数据集 ResNet baseline 运行器

模块职责
--------
为 ``codes/datasets/dm_dataset_en`` 提供一组「正常重新训练」的通用 baseline，
与 pd 数据集采用相同的训练框架与参数设计（dual_stream + K-fold + 数据增强 +
完整可视化/报告），但完全独立训练，不复用任何预训练表征（非迁移实验）。

与 ``main.py`` 的关系
--------------------
- 复用 ``ExperimentRunner`` 的全部训练 / 可视化 / 汇总 / 报告能力。
- 仅重写 ``_execute_experiment``：把 DM JSON 映射成 PD-schema 患者列表，
  按 DM 任务（dm_binary / dm_subtype）分别构建 ``PDDataset`` 并跑 K-fold。
- 两个 DM 任务都是二分类，统一复用二分类头（TASK_CONFIG['pd']）：
    * dm_binary : control(0) vs DR(NDR/NPDR, 1)，80 患者
    * dm_subtype: NDR(0)   vs NPDR(1)（排除 control），60 患者

入口接口
--------
    >>> from experiments.basic_experiments.dm_runner import run_single_dm_baseline_experiment
    >>> cfg = ExperimentConfig()
    >>> cfg.BACKBONE = 'resnet18'
    >>> run_single_dm_baseline_experiment(cfg, tasks=['dm_binary', 'dm_subtype'])
"""
import os
import json
from datetime import datetime
from typing import List, Dict, Any, Optional

from experiments.basic_experiments.main import ExperimentConfig, ExperimentRunner
from core.basic_classifier import TASK_CONFIG
from core.data_loader import PDDataset


# ==================== DM 数据常量 ====================

# DM 数据集 JSON（相对 codes/ 项目根；运行时 cwd 已切到项目根）
DM_JSON = 'datasets/dm_dataset_en/dm_patients_dataset_en.json'

# DM 支持的任务
DM_TASKS = ['dm_binary', 'dm_subtype']

# dm_subtype 子型标签映射（排除 control）
_DM_SUBTYPE_MAP = {'NDR': 0, 'NPDR': 1}

# DM 任务的中文展示名（用于报告，避免索引 TASK_CONFIG[dm_task]）
_DM_TASK_NAME = {
    'dm_binary': 'DM二分类 (control vs DR)',
    'dm_subtype': 'DM亚型 (NDR vs NPDR)',
}

# codes/ 项目根
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ==================== 路径解析 ====================

def _resolve_project_path(path: str) -> str:
    """把相对路径解析为存在的绝对路径，兼容从 ARIS 根或 codes 启动。"""
    if os.path.isabs(path):
        return path
    candidates = [
        os.path.join(_PROJECT_ROOT, path),
        os.path.join(os.path.dirname(_PROJECT_ROOT), path),
        path,
    ]
    for cand in candidates:
        if os.path.exists(cand):
            return cand
    return candidates[0]


def _datasets_base_path(json_abs_path: str) -> str:
    """由数据集 JSON 绝对路径推断 datasets 根目录（image_paths 拼接基准）。"""
    return os.path.dirname(os.path.dirname(json_abs_path))


# ==================== DM 数据构建 ====================

def build_dm_patients(task: str) -> List[Dict[str, Any]]:
    """构建符合 PD 数据 schema 的 DM 患者列表，使其可直接喂给 ``PDDataset``。

    DM 原始 JSON 的 ``image_paths`` 是扁平列表（无 central/whorl 区分），这里把
    同一张图同时作为 central 与 whorl（duplicate 配对），以适配 dual_stream
    编码器；并通过 ``group`` 字段编码二分类标签（control→0, 阳性→1），``stage`` 与
    ``subtype`` 置空使其编码为不参与。

    参数：
        task: 'dm_binary' 或 'dm_subtype'。

    返回：
        PD-schema 患者列表（每个含 group/stage/subtype/image_paths）。
    """
    if task not in DM_TASKS:
        raise ValueError(f"未知 DM 任务: {task}（应为 {DM_TASKS}）")

    json_abs = _resolve_project_path(DM_JSON)
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
            label = _DM_SUBTYPE_MAP.get(group)
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
            'stage': 0,        # → stage_label 不参与
            'subtype': '',     # → subtype_label 不参与
            'image_paths': {'central': imgs, 'whorl': imgs},
        })
    return out


# ==================== DM 基线运行器 ====================

class DMBaselineRunner(ExperimentRunner):
    """DM 数据集通用 baseline 运行器（正常重新训练，非迁移）。

    复用 ``ExperimentRunner`` 的训练 / K-fold / 可视化 / 汇总能力，仅重写
    ``_execute_experiment``：按 DM 任务分别构建数据集并跑二分类 K-fold。
    """

    def _execute_experiment(self, tasks: List[str]) -> Dict[str, Any]:
        config = self.config
        self._print_header()

        json_abs = _resolve_project_path(DM_JSON)
        base_path = _datasets_base_path(json_abs)

        # DM baseline 强制使用双流（与 pd 同框架）；single_stream 不适用此处映射
        architecture = config.ARCHITECTURE
        if architecture == 'single_stream':
            print("⚠ DM baseline 不支持 single_stream，自动切换为 dual_stream")
            architecture = 'dual_stream'

        results: Dict[str, Any] = {}

        for dm_task in tasks:
            if dm_task not in DM_TASKS:
                print(f"⚠ 跳过未知 DM 任务: {dm_task}")
                continue

            print(f"\n{'='*60}")
            print(f"任务: {_DM_TASK_NAME.get(dm_task, dm_task)}")
            print('='*60)

            patients = build_dm_patients(dm_task)
            print(f"加载 {len(patients)} 个 DM 患者 (任务={dm_task})")
            if not patients:
                print(f"⚠ 任务 {dm_task} 无患者，跳过")
                continue

            dataset = PDDataset(
                patients, architecture,
                self.classifier.train_transform,
                base_path=base_path,
            )
            if len(dataset) == 0:
                print(f"⚠ 任务 {dm_task} 数据集为空，跳过")
                continue

            print(f"样本数: {len(dataset)}")
            # 两个 DM 任务都是二分类，统一复用 'pd' 二分类头；suffix 区分任务
            results[dm_task] = self._run_kfold(dataset, 'pd', suffix=f"_{dm_task}")

        self._generate_dm_report(results)
        return results

    def _generate_dm_report(self, results: Dict[str, Any]) -> None:
        """生成 DM baseline 的 Markdown 报告（不依赖 TASK_CONFIG[dm_task]）。"""
        report_path = os.path.join(self.output_dir, 'experiment_report.md')
        cfg = self.config
        pd_name = TASK_CONFIG['pd']['name']

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# DM Baseline 实验报告\n\n")
            f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("## 实验配置\n\n")
            f.write("| 参数 | 值 |\n|---|---|\n")
            f.write("| 数据集 | `dm_dataset_en` |\n")
            f.write("| 训练方式 | `正常重新训练 (非迁移)` |\n")
            f.write(f"| 架构 | `{cfg.ARCHITECTURE}` |\n")
            f.write(f"| Backbone | `{cfg.BACKBONE}` |\n")
            f.write(f"| 预训练 | `{cfg.PRETRAINED}` |\n")
            f.write(f"| K-fold | {cfg.NUM_FOLDS} |\n")
            f.write(f"| K-fold模式 | `{cfg.KFOLD_MODE}` |\n")
            f.write(f"| Epochs | {cfg.NUM_EPOCHS} |\n")
            f.write(f"| 学习率 | {cfg.LEARNING_RATE} |\n")
            f.write(f"| Batch Size | {cfg.BATCH_SIZE} |\n")
            f.write(f"| 数据增强 | `{'开启' if cfg.DATA_AUGMENTATION else '关闭'}` |\n")
            f.write("\n")

            f.write("## 结果摘要\n\n")
            f.write("| DM 任务 | Accuracy | F1 Score |\n|---|---|---|\n")
            for dm_task, result in results.items():
                if dm_task == 'meta' or 'final_results' not in result:
                    continue
                fr = result['final_results']
                f.write(
                    f"| {_DM_TASK_NAME.get(dm_task, dm_task)} | "
                    f"{fr['avg_acc']:.4f}±{fr['std_acc']:.4f} | "
                    f"{fr['avg_f1']:.4f}±{fr['std_f1']:.4f} |\n"
                )

            f.write("\n---\n\n## 详细结果\n\n")
            for dm_task, result in results.items():
                if dm_task == 'meta':
                    continue
                # 复用基类的 task section 写法（图像文件名前缀为 pd_name+suffix）
                display = _DM_TASK_NAME.get(dm_task, dm_task)
                f.write(f"### {display}\n\n")
                if 'final_results' not in result:
                    f.write("*无有效结果*\n\n")
                    continue
                fr = result['final_results']
                f.write("| 指标 | 值 |\n|---|---|\n")
                f.write(f"| Accuracy | {fr['avg_acc']:.4f} ± {fr['std_acc']:.4f} |\n")
                f.write(f"| F1 Score | {fr['avg_f1']:.4f} ± {fr['std_f1']:.4f} |\n")
                if 'avg_precision' in fr:
                    f.write(f"| Precision | {fr['avg_precision']:.4f} ± {fr['std_precision']:.4f} |\n")
                if 'avg_recall' in fr:
                    f.write(f"| Recall | {fr['avg_recall']:.4f} ± {fr['std_recall']:.4f} |\n")
                f.write("\n")

                suffix = f"_{dm_task}"
                cm_file = f"{pd_name}{suffix}_Test_cm.png"
                roc_file = f"{pd_name}{suffix}_Test_roc.png"
                if os.path.exists(os.path.join(self.output_dir, cm_file)):
                    f.write(f"**混淆矩阵**:\n\n![混淆矩阵]({cm_file})\n\n")
                if os.path.exists(os.path.join(self.output_dir, roc_file)):
                    f.write(f"**ROC 曲线**:\n\n![ROC曲线]({roc_file})\n\n")

            f.write("\n---\n\n## 附件说明\n\n")
            f.write("| 文件 | 说明 |\n|---|---|\n")
            f.write("| `config.txt` | 完整配置 |\n")
            f.write("| `output.txt` | 训练日志 |\n")
            f.write("| `best_model_*.pth` | 最佳模型权重 |\n")

        print(f"✓ 生成报告: {report_path}")


# ==================== 公共接口 ====================

def run_single_dm_baseline_experiment(config: Optional[ExperimentConfig] = None,
                                      tasks: Optional[List[str]] = None,
                                      output_dir: Optional[str] = None) -> Dict[str, Any]:
    """运行单次 DM baseline 实验（批量实验入口）。

    参数：
        config: 实验配置，默认 ``ExperimentConfig()``。
        tasks: DM 任务列表，默认 ``['dm_binary', 'dm_subtype']``。
        output_dir: 输出目录，默认自动生成。

    返回：
        实验结果字典。
    """
    if config is None:
        config = ExperimentConfig()
    if tasks is None:
        tasks = list(DM_TASKS)

    runner = DMBaselineRunner(config)
    return runner.run(tasks=tasks, output_dir=output_dir)
