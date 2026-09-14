# TopoDiSE 实验代码

代码保留论文采用的完整配置：7 种 baseline 的 43 组实验，以及 TopoDiSE 的 3 个主任务、40 组消融实验和 1 组 DM 迁移实验。配置中的随机种子、交叉验证、baseline TD-rest 二分类和 TopoDiSE 集成候选均保留。

## 环境与数据

在 `codes/` 下安装依赖，并使用与设备匹配的 PyTorch：

```bash
python -m pip install -r requirements.txt
```

将授权数据放在下面的目录中，JSON 图像路径相对于 `datasets/`：

```text
datasets/
├── CORN-1/
│   ├── images/{train_orig,val_orig,test_orig}/
│   └── labels/{train_label,val_label,test_label}/
├── pd_dataset_en/patients_dataset_en.json
├── pd_dataset_unet_en/patients_dataset_en.json
├── dm_dataset_en/dm_patients_dataset_en.json
└── unlabeled_dataset_en/unlabeled_dataset_en.json
```

数据和训练权重不随仓库提供。需要 ImageNet 初始化的配置会读取本地缓存或下载权重；下载失败时实验报错，不会自动改成随机初始化。

## 目录与统一入口

每个实验家族及 TopoDiSE 阶段目录只保留两个 Python 文件：`main.py` 包含单组实验、DM 实验及 TD-rest 二分类逻辑，`batch_experiments.py` 控制批量运行。实验参数直接写在各目录的 `batch_experiments.py` 中；不再从外部 JSON 读取默认配置。公共进程执行工具和阶段配置类型放在 `experiments/` 根目录，不再保留独立的 `dm_runner.py`、`dm_transfer.py` 或 `subtype_hierarchy_runner.py`。

| 实验家族 | 目录 | 保留内容 |
|---|---|---|
| ResNet18 | `experiments/basic_experiments/` | binary、stage、subtype、DM |
| SimCLR | `experiments/simclr_experiments/` | binary、stage、subtype、DM；保留预训练 |
| MTL-MAE | `experiments/mtl_experiments/` | binary、stage、subtype、DM；保留多任务预训练 |
| Prototype | `experiments/prototype_experiments/` | binary、stage、subtype |
| U-Net-only / U-Net-dual | `experiments/unet_experiments/` | 分割图单独分类 / 原图与分割图融合分类 |
| GNN | `experiments/gnn_experiments/` | 分割图骨架转图后的 GAT 分类 |
| TopoDiSE | `experiments/topodise_experiments/` | Stage0/1/2、主任务、消融、DM |

`unet/` 包含 CORN-1 nnU-Net 训练、推理、评估、PD 分割数据生成，以及分类所需的模型和数据组件。U-Net-only / dual 的实验流程在 `experiments/unet_experiments/main.py`。`gnn/` 只包含骨架转图、图数据集、GNN 模型与训练。`utils/` 保留正式实验依赖的日志、绘图、患者级结果导出和亚型评估工具。

TopoDiSE 阶段的配置类型定义在 `experiments/_stage_configs.py`；实际运行参数在各 stage 的 `batch_experiments.py` 中编辑。旧配置导入路径由公共包兼容，供未改动的统一 Stage2 使用。

`experiments/unified_stage2_encoder_comparison/` 保持原样，包括其配置、测试与现有入口，详见该目录的 [README](experiments/unified_stage2_encoder_comparison/README.md)。

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

## main 与 batch_experiments 的分工

按 `original_codes` 的方式使用：**编辑 `batch_experiments.py`，然后运行它**。即使只跑一组，也在批量文件中选择该组。

- `main.py` 实现训练、验证、迁移、TD-rest 二分类和结果输出，接收批量文件传入的配置。
- `batch_experiments.py` 定义最佳参数、选择哪些实验、使用哪些 GPU、每卡并发数和输出目录。默认参数就是保留的最佳配置；消融实验保留对应的变量设置。

每个批量文件顶部可直接修改：

```python
AVAILABLE_GPUS = [0]       # 一张卡；[0, 1] 两张卡；[] 使用 CPU
CONCURRENCY_PER_GPU = 1    # 每张卡同时运行几组实验；每组实验使用一张卡
NUM_WORKERS = 0            # DataLoader worker 数；TopoDiSE 默认 4
OUTPUT_DIR = 'results/basic'
RUN_GROUPS = []            # 留空运行全部，或填入 EXPERIMENTS 中的 _group 名称
```

下方 `EXPERIMENTS` 字典列表包含学习率、训练轮数、batch size、随机种子、任务及交叉验证等参数。参数修改会原样传给执行进程，结果目录同时保存这份配置。TopoDiSE 总入口使用 `RUN_SECTION` 选择主任务/消融/DM，并将各分区列表汇总到 `EXPERIMENTS`。

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

TopoDiSE 的任务、集成和消融配置，以及受保护的统一 Stage2 项目保持不变。

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

## TopoDiSE

总入口的 `batch_experiments.py` 直接包含 3 个主任务、40 组消融和 1 组 DM 迁移的完整配置。`main.py` 实现阶段衔接、训练恢复、分类和集成；批量文件控制共享预训练去重、任务顺序和多卡调度。

```bash
python -m experiments.topodise_experiments.batch_experiments --section main_tasks
python -m experiments.topodise_experiments.batch_experiments --section ablation --gpus 0,1
python -m experiments.topodise_experiments.batch_experiments --section dm_transfer
python -m experiments.topodise_experiments.batch_experiments --section all --dry-run
```

### TopoDiSE 三态 reproducibility

总入口的 `reproducibility` 参数有三个明确状态：

| 状态 | Stage0/1 | 用途 |
|---|---|---|
| `strict` | 从零训练 | 固定 c8 配方、seed 42、4 workers、全部 RNG 和确定性后端，用于稳定的近似复刻 |
| `portable` | 从零训练 | 只保持方法和配方一致，不强制 RNG/确定性 CUDA 约束，适配不同 GPU/CPU/MPS |
| `checkpoint` | 跳过 | 直接使用给定 c8 Stage1 checkpoint 运行 Stage2，用于复现论文最佳权重结果 |

```bash
# 1. 从零严格近似复刻
python -m experiments.topodise_experiments.batch_experiments \
  --section main_tasks --reproducibility strict

# 2. 从零设备自适应训练（默认）
python -m experiments.topodise_experiments.batch_experiments \
  --section main_tasks --reproducibility portable

# 3. 直接调用 c8 Stage1 权重
python -m experiments.topodise_experiments.batch_experiments \
  --section main_tasks --reproducibility checkpoint \
  --checkpoint-dir /path/to/c8_checkpoint_bundle
```

不带值的 `--reproducibility` 保留为 `strict` 的兼容写法；
`--no-reproducibility` 等价于 `portable`。也可使用
`TOPODISE_REPRODUCIBILITY=strict|portable|checkpoint`，checkpoint 目录可用
`TOPODISE_CHECKPOINT_DIR` 指定。

`strict` 和 `portable` 都会禁止跨批次复用旧预训练权重，确保本次预训练起步于随机初始化；
同一输出目录的中断任务仍可继续恢复。`strict` 在不支持确定性算子的设备上会明确报错，
此时应改用 `portable`。

checkpoint 目录推荐保留服务器现有的预训练包结构：

```text
c8_checkpoint_bundle/
├── _experiment_def.json                 # 可选，实验溯源
├── summary.json                         # 可选，训练摘要
├── <name>_stage0/                       # 可选
│   └── ckpt/ema_best.pt
└── <name>_stage1/                       # 必需
    └── ckpt/ema_best.pt
```

对精简分发包，也可只保留 `stage1/ema_best.pt`。程序会优先寻找
`ema_best.pt`，其次是 `last.pt`；同一阶段存在多个候选时会报错，避免静默选错权重。

已验证的 `strict` 从零基线为 Stage0 epoch 100、loss `0.1106367812`；
GPU0/GPU1 学习状态完全一致。固定下游规则下，PD Stage 为
**65.61%**，PD Subtype TD-rest 为 **70.90%**。历史 c8 Stage1 checkpoint 对应
**69.31% / 74.07%**。Stage 使用单患者 U0/U1/U2 最低熵选择，Subtype 使用
TDREST0/1/2/5/7 多数投票，都不读取患者标签。详细数值记录见
[最佳预训练配置说明](experiments/topodise_experiments/REPRODUCE_C8BF6B548656.md)。

三个 stage 也有各自独立的批量配置：

```bash
python -m experiments.topodise_experiments.stage0_corn1_pretrain.batch_experiments
# 将 Stage0 实际输出的 ema_best.pt 填入 Stage1 批量文件 config.parent_ckpt
python -m experiments.topodise_experiments.stage1_rscp_finetune.batch_experiments
# 将 Stage1 实际输出的 ema_best.pt 填入 Stage2 批量文件 PARENT_CKPT
python -m experiments.topodise_experiments.stage2_racf_downstream.batch_experiments
```

Stage0 默认使用最佳配方的 200 epochs，Stage1 为 80 epochs；各训练参数均在对应批量文件中。独立 Stage2 只运行已有 Stage1 检查点上的下游任务，不自动重训前序阶段；完整消融从总入口选择。复现参数和参考记录见 [最佳预训练配置说明](experiments/topodise_experiments/REPRODUCE_C8BF6B548656.md)。

完整实验仍需授权数据与相应训练设备。
