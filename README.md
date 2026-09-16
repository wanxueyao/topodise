# TopoDiSE 与分类实验代码

当前正式 TopoDiSE 为 **Stage0 GT + Stage1 HGrad 双阈值**，可运行 PD binary、PD stage、TD/non-TD subtype 和 DM transfer 四任务。所有 experiments 分类入口及新消融任务自动输出 Accuracy、BAcc、Macro-F1、AUC 和患者 bootstrap 95% CI。

## 安装与数据

Python >=3.10，先安装适合设备的 PyTorch，再运行 `pip install -r requirements.txt`。
授权数据默认放在 `datasets/`，包含 `CORN-1`（images 和 labels）、`pd_dataset_en/patients_dataset_en.json`、`dm_dataset_en/dm_patients_dataset_en.json`、`unlabeled_dataset_en/unlabeled_dataset_en.json`。图像路径相对于 datasets；主方法支持 `--dataset-root` 指定外部路径。数据及权重不随源码提供。

## 新主方法与消融入口

```text
codes/
  experiments/nerve_metrics_lr_experiments/  11 神经指标 + LR
  experiments/topodise_experiments/    GT/HGrad-dual 主方法与完整四任务 Stage2
    recipes.py                        固定下游候选配置
  topodise/training/                   三阶段公共实现、中性注入接口与运行设施
  ablation_experiments/                与 experiments 同级
    G01_G05_encoder_comparison/       编码器比较及冻结探针
    B01_B06_topology_targets/         GT、匹配双阈值、通道与空间置乱
    A01_A07_... 到 I01_I09_...         全部 68 个消融设计编号
  utils/confidence_intervals.py        患者 cluster bootstrap 统计
codes2/                               同级独立研究副本，原 HGrad/HGrad 和双阈值 HGrad/HGrad
```

```bash
# 主方法从零运行四任务
python -m experiments.topodise_experiments.batch_experiments --devices cuda:0
# 查看结构目标组的消融计划
python -m ablation_experiments.B01_B06_topology_targets.batch_experiments --dry-run
# 编码器对比（先按说明放置 DINOv2 本地权重）
python -m ablation_experiments.G01_G05_encoder_comparison.run --devices cuda:0 cuda:1
```

详细运行、复用旧权重、续跑及指标定义：
- [TopoDiSE 主方法](experiments/topodise_experiments/README.md)
- [消融目录](ablation_experiments/README.md)
- [编码器对比](ablation_experiments/G01_G05_encoder_comparison/README.md)
- [结构目标消融](ablation_experiments/B01_B06_topology_targets/README.md)

原始 HGrad/HGrad 与双阈值 HGrad/HGrad 两套基础代码独立保存于同级 codes2；不再作为本仓库主方法默认配置。旧消融结果不覆盖、不并入新报告。其余已有 baseline 实验入口与参数保留如下。

## U-Net 和 GNN 的共同预备条件

**先使用 CORN-1 训练 nnU-Net，再用训练权重生成 PD 分割数据，最后运行分类实验。** U-Net-only、U-Net-dual 和 GNN 都依赖这一步。

在 `codes/` 下按顺序执行：

```bash
# 1. CORN-1 转为 nnU-Net 数据格式
python -m unet.prepare_dataset

# 2. 预处理并训练全部 5 折
python -m unet.train --profile baseline

# 3. 使用训练好的分割模型处理 PD 原图，保留患者元数据及区域对应关系
python -m unet.build_pd_unet_dataset

# 4. 才开始 U-Net / GNN 分类实验
python -m experiments.unet_experiments.batch_experiments
python -m experiments.gnn_experiments.batch_experiments
```

生成结果为 `datasets/pd_dataset_unet_en/patients_dataset_en.json` 和其中引用的分割图。U-Net-dual 还需要对齐的 `pd_dataset_en` 原图。两类入口会检查清单及图像是否存在。已按上述流程生成的完整分割数据可直接迁移使用；分类时无需再次加载分割模型。训练恢复和推理参数见 [unet/README.md](unet/README.md)。

## 其他 baseline 的 main 与 batch_experiments 分工

按 `original_codes` 的方式使用：**编辑 `batch_experiments.py`，然后运行它**。即使只跑一组，也在批量文件中选择该组。

- `main.py` 实现训练、验证、迁移、TD-rest 二分类和结果输出，接收批量文件传入的配置。
- `batch_experiments.py` 定义最佳参数、选择哪些实验、使用哪些 GPU、每卡并发数和输出目录。默认参数就是保留的最佳配置；TopoDiSE 消融另以编号目录 JSON 登记。

每个批量文件顶部可直接修改：

```python
AVAILABLE_GPUS = [0]       # 一张卡；[0, 1] 两张卡；[] 使用 CPU
CONCURRENCY_PER_GPU = 1    # 每张卡同时运行几组实验；每组实验使用一张卡
NUM_WORKERS = 0            # DataLoader worker 数；TopoDiSE 默认 4
OUTPUT_DIR = 'results/basic'
RUN_GROUPS = []            # 留空运行全部，或填入 EXPERIMENTS 中的 _group 名称
```

下方 `EXPERIMENTS` 字典列表包含学习率、训练轮数、batch size、随机种子、任务及交叉验证等参数。参数修改会原样传给执行进程，结果目录同时保存这份配置。TopoDiSE 使用独立的新主入口，见上文。

## Baseline 任务约定

七种 baseline（basic、SimCLR、MTL、Prototype、GNN、U-Net-only、U-Net-dual）均包含：

| 任务 | 实际执行 |
|---|---|
| `pd_binary` | PD 二分类 |
| `pd_stage` | PD 分期分类 |
| `pd_subtype` | **TD-rest 二分类**：TD=0，Rest（PIGD+mixed）=1 |

`pd_subtype` 配置明确设置 `_subtype_protocol='td_rest'`，只训练 TD-rest head，不再执行 PIGD-vs-Mixed 或合成三分类。保留原 TD-rest 分支的训练参数和患者级 5-fold × 3 划分；输出二分类 accuracy、balanced accuracy、macro F1、AUC、混淆矩阵和患者概率。新实验组名使用 `subtype_td_rest`，避免与旧层级三分类结果混淆。

三种方法各有 3 个 seed 的 `dm_binary` 配置，来源在批量文件 `_transfer_source` 中明确：

| 方法 | DM 初始化/迁移来源 |
|---|---|
| basic | `imagenet`：ImageNet 初始化后在 DM 上训练；**不是 PD 检查点→DM** |
| SimCLR | `pd_simclr`：PD 自监督预训练 backbone→DM |
| MTL | `pd_corn1_mtl`：PD+CORN-1 多任务预训练 encoder→DM |

TopoDiSE 和新的编码器对照使用完整 Stage2，见对应目录的说明。

## Baseline 运行

```bash
# 列出实验或预览参数
python -m experiments.basic_experiments.batch_experiments --list
python -m experiments.basic_experiments.batch_experiments --dry-run
# 只运行指定组，使用 GPU 0
python -m experiments.basic_experiments.batch_experiments \
  --group P11_resnet18_patient_acc_seed42 --gpus 0
# 使用两张卡，每卡同时运行一组
python -m experiments.basic_experiments.batch_experiments --gpus 0,1 --concurrency-per-gpu 1
```

其他家族用相同方式运行：

```bash
python -m experiments.simclr_experiments.batch_experiments
python -m experiments.mtl_experiments.batch_experiments
python -m experiments.prototype_experiments.batch_experiments
python -m experiments.gnn_experiments.batch_experiments
python -m experiments.unet_experiments.batch_experiments --variant unet_only
python -m experiments.unet_experiments.batch_experiments --variant unet_dual
```

U-Net 的 `VARIANT` 默认 `all`，两种分类方法各保留 4 组。命令行可覆盖 `--group`（逗号分隔多组）、`--gpus`、`--concurrency-per-gpu`、`--num-workers` 和 `--output-dir`；`--gpus cpu` 使用 CPU。`--list` / `--dry-run` 不启动训练。

公共 `_execution.py` 只提供进程执行、GPU 槽位和日志机制，不保存实验选择或模型参数。它将批量文件提供的完整配置写入结果目录，调用同目录 `main.py --config ...` 执行，避免参数在子进程中被旧默认值覆盖。训练日志与 `batch_summary.json` 保存在输出目录，失败返回非零状态。

## 所有分类实验的直接结果输出

Basic、SimCLR、Prototype、MTL、GNN、U-Net-only、U-Net-dual、11 metrics + LR 的正式 main/batch 入口均在训练完成后自动生成：

- 每组 `metrics_ci95.md`：可直接阅读和复制的 accuracy、bacc、macro-F1、AUC 及 95% CI。
- `metrics_ci95.csv/json`：同一结果的结构化版本。
- `oof_predictions.json`：所选模型的患者级折外预测；`report_status.json` 标明汇总是否完整。
- batch 根目录 `metrics_ci95.*`：各组结果；`combined_metrics_ci95.*`：只合并超参数完全相同、仅训练 seed 不同且全部成功的组。不同方法/配置不合并，失败组不会产生完整联合结果。

TopoDiSE 与新消融沿用自身 `task_summary.json` 和 `metrics_ci95.csv/md`，总表在 `analysis/`。无需再运行计算脚本。

新表统一按患者计算：多图预测先取患者平均概率；每次完整 OOF 计算指标，再对重复求均值。95% CI 按真实类别分层重采样患者，重复间保持同一抽样权重，不把重复患者当作新样本。AUC 使用原始概率，二分类取 class 1，三分类取 macro OvR；未定义值明确为 NA。CI 条件于已训练模型，不纠正使用验证折选择 checkpoint 的偏差。

MTL 的历史 stage 头保留 5 个输出，新评估明确采用真实任务标签 0/1/2；预测到额外输出仍计为错误，概率不裁剪、不重新归一化。最佳轮次的预测现在单独保存，避免被最后一轮覆盖。旧图像级指标和折均值文件仍保留用于追溯，应以新患者表作为统一口径。不同方法的数据队列和划分未因此自动变成一致，请同时核对 n_patients 和数据审计。

11 指标 LR 入口和真实可用数据边界见 [说明](experiments/nerve_metrics_lr_experiments/README.md)：

```bash
python -m experiments.nerve_metrics_lr_experiments.batch_experiments
```

消融按指南 68 个编号统一登记在 [消融入口](ablation_experiments/README.md)，共享三阶段训练位于 `topodise/training/`。正式方法仅保留固定 M0 候选；完整消融计划使用 `python -m ablation_experiments.run --dry-run`。
