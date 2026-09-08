# ARIS / TopoDiSE 实验代码

本目录用于运行当前版本的实验实现。实验分为两部分：

1. `experiments/*_experiments/`：论文批量实验，包含 7 个 baseline 家族、TopoDiSE 三个主任务、TopoDiSE ablation 和 DM 迁移。每个 baseline 只保留论文采用的最佳配置。
2. `experiments/unified_stage2_encoder_comparison/`：统一 Stage2 冻结编码器比较。该部分只使用本机进程，不包含双 DGX Spark、SSH、rsync 或节点编号逻辑。

历史调参网格和未进入论文的 light-model 实验已从默认开源入口移除。

## 目录结构

```text
codes/
├── core/                         # 共享数据、模型与训练工具
├── datasets/                     # 数据目录与 manifest
├── experiments/
│   ├── basic_experiments/          # ResNet18 baseline
│   ├── simclr_experiments/         # SimCLR baseline
│   ├── gnn_experiments/            # UNet-only / UNet-dual / GNN
│   ├── mtl_experiments/            # MTL-MAE baseline
│   ├── prototype_experiments/      # Prototype baseline
│   ├── topodise_experiments/       # TopoDiSE main / ablation / DM
│   ├── paper_baseline_experiments.json
│   └── unified_stage2_encoder_comparison/
├── gnn/                          # GNN 与双视角模型
├── multitasklearning/            # MTL-MAE 模型
├── topodise/                     # TopoDiSE 核心实现
├── unet/                         # CORN-1 nnU-Net 分割流程
└── requirements.txt              # 唯一依赖文件
```

## 1. 环境

建议 Python 3.10–3.12。只需使用根目录的一个依赖文件：

```bash
cd codes
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

CUDA 机器可先按 PyTorch 官方说明安装与驱动匹配的 `torch` / `torchvision`，再执行上述 requirements。历史最佳 parent 的参考环境是 Python 3.10.19、PyTorch 2.9.1+cu128、cuDNN 91002 和 RTX A6000；其他设备可复现配置和损失轨迹，但不承诺 checkpoint 逐字节相同。

## 2. 数据

将已授权数据放置为：

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

JSON 内图像路径应相对于 `datasets/` 有效。CORN-1 需按原数据许可单独获取。

UNet-only、UNet-dual 和 GNN 依赖 `pd_dataset_unet_en`。如果未颈先生成，可按 [unet/README.md](unet/README.md) 先训练 CORN-1 nnU-Net，再运行：

```bash
cd unet
python prepare_dataset.py
python train.py --profile baseline
python build_pd_unet_dataset.py
cd ..
```

## 3. 复现 TopoDiSE Stage0 / Stage1 最佳 parent

论文 main tasks 和 DM 迁移统一使用 stage/subtype 最佳组对应的 parent recipe：`c8bf6b548656`。它在 Stage0 使用 CORN-1 `hgrad` 监督，Stage1 使用 PD control/control-cataract region pairs，并从 Stage0 的原始 encoder 权重初始化。

从 Stage0 开始运行：

```bash
python -m experiments.topodise_experiments.reproduce_c8bf6b548656 \
  --data-root datasets \
  --output-dir results/reproduce_c8bf6b548656 \
  --device auto
```

默认 Stage0 200 epochs、Stage1 80 epochs。中断后使用 `--resume`；只运行某一阶段可使用 `--stage 0` 或 `--stage 1 --parent-ckpt PATH`。专用入口会记录解析后配置、数据指纹、RNG 状态和 checkpoint SHA256。完整参考信息见 [REPRODUCE_C8BF6B548656.md](experiments/topodise_experiments/REPRODUCE_C8BF6B548656.md)。

### 可复现开关

Stage0/Stage1 只保留 `reproducibility` 开关，并且该开关只控制一个行为：
开启时在每个阶段初始化处执行 `torch.manual_seed(42)`；关闭时不设置任何
seed，直接使用环境的 RNG 状态。两种模式都不设置 Python/NumPy seed，也不
修改任何 PyTorch/CUDA 后端开关。检查点始终保存 PyTorch CPU/CUDA RNG 状态，
断点续训时始终优先恢复该状态，不受此开关影响。

| 配置 | Stage0/Stage1 初始化 | 适用场景 |
|---|---|---|
| `reproducibility: true` | 执行 `torch.manual_seed(42)` | 复刻与固定起始随机流 |
| `reproducibility: false` | 不主动设置 seed | 普通训练与迁移测试 |

该开关不表示完全确定性训练，也不会启用 deterministic algorithms；它只决定
是否执行上述一次 PyTorch seed 初始化。

专用复刻入口默认开启。普通训练如不需要固定 seed，可显式关闭：

```bash
python -m experiments.topodise_experiments.reproduce_c8bf6b548656 \
  --no-reproducibility
```

## 4. TopoDiSE 论文实验

精确配置位于 `experiments/topodise_experiments/paper_experiments.json`，分为：

- `main_tasks`：binary、stage、subtype 三个主任务；
- `ablation`：论文表格使用的 40 个 ablation；
- `dm_transfer`：`DMMIL3F1_weak_strict_fdiff`。

当前主任务采用唯一的正式患者级汇总策略：

- Stage 使用 U0/U1/U2 的 `fold_calibrated_entropy_routing`。对每个候选计算
  `当前患者熵 - 该候选在当前折的平均熵`，选择校准熵最低者；不再生成原始
  最低熵、多数投票、均值融合或标签 oracle 路由。
- TD_REST subtype 使用 TDREST0/1/2/5/7 的
  `majority_vote_tie_avg_prob`，不使用校准熵路由。

校准熵不读取标签，但需要一个患者群体估计候选基准熵。部署到独立单患者时，
应固定使用训练集或独立校准集得到的基准熵，不能用单个测试患者重新估计。

先检查将要运行的配置：

```bash
RUN_PAPER_SECTION=main_tasks \
python -m experiments.topodise_experiments.batch_experiments --dry-run
```

分别运行：

```bash
RUN_PAPER_SECTION=main_tasks python -m experiments.topodise_experiments.batch_experiments
RUN_PAPER_SECTION=ablation python -m experiments.topodise_experiments.batch_experiments
RUN_PAPER_SECTION=dm_transfer python -m experiments.topodise_experiments.batch_experiments
```

当本地不存在 parent 时，Stage2 定义会创建并复用同一个 `c8bf6b548656` Stage0/1 parent。机器参数可通过 `RUN_OUTPUT_DIR`、`RUN_NUM_WORKERS`、`RUN_AVAILABLE_GPUS`、`RUN_PARALLEL_EXPERIMENTS` 设置。

## 5. 论文 baseline

所有 baseline 的公开配置只来自 `experiments/paper_baseline_experiments.json`；批处理脚本不再构造历史扫参网格。

| 方法 | 入口 | 保留的论文实验 |
|---|---|---|
| ResNet18 | `basic_experiments.batch_experiments` | binary/stage 3 seeds，subtype 5-fold x3，DM 3 seeds |
| SimCLR | `simclr_experiments.batch_experiments` | binary 3 seeds，CORN-1 stage 3 seeds，subtype 5-fold x3，DM 3 seeds |
| UNet-only | `gnn_experiments.batch_experiments` | binary/stage 3 seeds，subtype 5-fold x3 |
| UNet-dual | `gnn_experiments.dual_stream_batch` | binary/stage 3 seeds，subtype 5-fold x3 |
| GNN | `gnn_experiments.gnn_batch` | binary/stage 3 seeds，subtype 5-fold x3 |
| MTL-MAE | `mtl_experiments.batch_experiments` | binary/stage 3 seeds，subtype 5-fold x3，DM 3 seeds |
| Prototype | `prototype_experiments.batch_experiments` | binary 3 seeds，stage 3 seeds，subtype 5-fold x3 |

在一台普通 CUDA 机器上，可用 `RUN_SITE=auto` 自动检测 GPU：

```bash
RUN_SITE=auto RUN_CONCURRENCY_PER_GPU=1 python -m experiments.basic_experiments.batch_experiments
RUN_SITE=auto RUN_CONCURRENCY_PER_GPU=1 python -m experiments.simclr_experiments.batch_experiments
RUN_SITE=auto RUN_CONCURRENCY_PER_GPU=1 python -m experiments.gnn_experiments.batch_experiments
RUN_SITE=auto RUN_CONCURRENCY_PER_GPU=1 python -m experiments.gnn_experiments.dual_stream_batch
RUN_SITE=auto RUN_CONCURRENCY_PER_GPU=1 python -m experiments.gnn_experiments.gnn_batch
RUN_SITE=auto RUN_CONCURRENCY_PER_GPU=1 python -m experiments.mtl_experiments.batch_experiments
RUN_SITE=auto RUN_CONCURRENCY_PER_GPU=1 python -m experiments.prototype_experiments.batch_experiments
```

不设 `RUN_INCLUDE_GROUPS` 时运行该方法全部论文配置。如只运行一组，使用 manifest 中的精确 `_group`：

```bash
RUN_SITE=auto \
RUN_INCLUDE_GROUPS=P11_c304_pd_patient_acc_seed42 \
python -m experiments.simclr_experiments.batch_experiments
```

CPU 也可使用 `RUN_SITE=local`，但完整 5-fold x3 训练会很慢。默认结果位于 `results/`，可用 `RUN_OUTPUT_DIR` 覆盖。

## 6. 统一 Stage2 编码器比较

该协议在同一 patient-level Stage2 流程中比较冻结的 TopoDiSE、SimCLR ResNet-18 和 DINOv2。TopoDiSE 仍使用统一 `c8bf6b548656` parent。审计、smoke test、单机运行、断点续跑和分析命令见 [unified_stage2_encoder_comparison/README.md](experiments/unified_stage2_encoder_comparison/README.md)。

## 7. 快速校验

```bash
python -m compileall -q core gnn multitasklearning topodise unet experiments
python -m unittest experiments.unified_stage2_encoder_comparison.tests.test_protocol
python -m experiments.unified_stage2_encoder_comparison.smoke_test \
  --config experiments/unified_stage2_encoder_comparison/configs/protocol_v1.yaml
```

报告结果时应同时保留：数据/配置 hash、checkpoint SHA256、patient-level fold predictions 和软件环境。
