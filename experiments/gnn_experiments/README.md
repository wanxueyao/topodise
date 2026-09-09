# GNN 实验

本目录只运行论文 GAT 最佳配置：binary/stage 各 3 seeds，以及 subtype TD-rest 二分类 5-fold × 3。最佳参数直接写在本目录 `batch_experiments.py` 的 `EXPERIMENTS` 中。

## 必须先完成 CORN-1 U-Net 训练

GNN 的输入是 **U-Net 分割图骨架化后得到的图结构**。在 `codes/` 下先执行：

```bash
python -m unet.prepare_dataset
python -m unet.train --profile baseline
python -m unet.build_pd_unet_dataset
```

准备 CORN-1 原始图像和标签，以及 `datasets/pd_dataset_en` 原始 PD 数据。生成的 `datasets/pd_dataset_unet_en/patients_dataset_en.json` 和全部分割图必须存在。若已有按此流程生成的完整数据，可直接使用。详细步骤见 [U-Net 说明](../../unet/README.md)。

## 分类实验

```bash
python -m experiments.gnn_experiments.batch_experiments --list
python -m experiments.gnn_experiments.batch_experiments --group P01_a01_patient_acc_seed42
python -m experiments.gnn_experiments.batch_experiments --dry-run
python -m experiments.gnn_experiments.batch_experiments --gpus 0
```

`main.py` 实现完整实验，`batch_experiments.py` 配置最佳参数、选择实验并分配 GPU。TD-rest 二分类实验逻辑已合并进 `main.py`，本目录只保留这两个 Python 文件。模型、骨架化、构图和训练实现位于 `gnn/`。U-Net-only / U-Net-dual 的分类入口位于独立的 `experiments/unet_experiments/`。

任务统一为 `pd_binary`、`pd_stage`、`pd_subtype`；其中 `pd_subtype` 只执行 TD=0 / Rest（PIGD+mixed）=1 二分类。
