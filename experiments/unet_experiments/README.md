# U-Net 分类实验

两种方法的实验实现统一位于 `main.py`，`batch_experiments.py` 配置参数、选择方法和分配 GPU；模型与数据组件位于 `unet/`：

| `--variant` | 方法 | 输入 |
|---|---|---|
| `unet_only` | U-Net-only | U-Net 分割图 |
| `unet_dual` | U-Net-dual | 原图与对应的 U-Net 分割图 |
| `all` | 两种方法 | 仅批量运行时使用 |

## 必须先完成 CORN-1 U-Net 训练

在 `codes/` 下按顺序执行：

```bash
python -m unet.prepare_dataset
python -m unet.train --profile baseline
python -m unet.build_pd_unet_dataset
```

先准备 CORN-1 图像/标签和 `datasets/pd_dataset_en` 原始 PD 数据。训练后的模型用于生成 `datasets/pd_dataset_unet_en` 分割图及患者清单；U-Net-dual 还要求原图与分割图的患者和区域对应关系一致。已有按此流程生成的完整分割数据可直接使用，详见 [U-Net 说明](../../unet/README.md)。

## 分类实验

```bash
python -m experiments.unet_experiments.batch_experiments --variant unet_only --list
python -m experiments.unet_experiments.batch_experiments --variant unet_dual \
  --group P01_b04_patient_acc_seed42
python -m experiments.unet_experiments.batch_experiments --variant all --dry-run
python -m experiments.unet_experiments.batch_experiments --variant all --gpus 0
```

每种方法保留 4 组正式配置：binary/stage 三个 seed，以及 subtype TD-rest 二分类 5-fold × 3。批量文件的 `VARIANT` 默认 `all`。最佳参数直接写在同文件的 `EXPERIMENTS` 中；使用 `RUN_GROUPS` 和 `AVAILABLE_GPUS` 选择实验和显卡。

任务统一为 `pd_binary`、`pd_stage`、`pd_subtype`；其中 `pd_subtype` 只执行 TD=0 / Rest（PIGD+mixed）=1 二分类。
