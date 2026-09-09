# CORN-1 nnU-Net 分割流程

本目录是基于 nnU-Net v2 的 CORN-1 角膜内皮细胞边界分割流程，包含数据转换、训练、推理、评估和 PD 图像分割版数据集生成脚本，以及 U-Net-only / U-Net-dual 分类实现。所有路径和 nnU-Net 运行目录默认都收敛在 `codes/unet/` 下，便于迁移和复现实验。

## 当前文件结构

```
codes/unet/
├── README.md                  # 本说明文档
├── config.py                  # 数据路径、nnU-Net路径、dataset id、fold配置
├── prepare_dataset.py         # 将 CORN-1 TIF 图像/标签转换为 nnU-Net NIfTI 格式
├── train.py                   # 规划、预处理并训练 nnU-Net v2
├── predict.py                 # 加载训练权重，对 TIF 图像输出分割掩模
├── evaluate.py                # 计算 Dice、IoU、像素准确率
└── build_pd_unet_dataset.py   # 用 nnU-Net 掩模替换 PD 原图，生成 pd_dataset_unet_en
```

运行后会自动生成：

```
codes/unet/nnunet_data/
├── nnUNet_raw/Dataset001_Cornea/
├── nnUNet_preprocessed/Dataset001_Cornea/
└── nnUNet_results/Dataset001_Cornea/
```

## 数据集布局

`config.py` 默认从 `codes/datasets/CORN-1` 读取 CORN-1：

```
codes/datasets/CORN-1/
├── images/
│   ├── train_orig/
│   ├── val_orig/
│   └── test_orig/
└── labels/
    ├── train_label/
    ├── val_label/
    └── test_label/
```

图像为灰度 TIF；标签为二值 TIF，背景为 `0`，细胞边界为 `255`。`prepare_dataset.py` 默认会把原始 train 和 val 合并为 nnU-Net 训练集，因为 nnU-Net 会自行管理交叉验证划分。

## 依赖安装

先安装适合当前设备的 PyTorch，再安装本流程依赖：

```bash
pip install nnunetv2 nibabel pillow tqdm numpy
```

推荐使用 CUDA GPU 训练。`train.py` 支持 `cuda`、`cpu`、`mps` 参数，但 CPU/MPS 训练速度和兼容性取决于具体环境。

## 完整流程

建议从本目录运行：

```bash
cd codes/unet
```

### 1. 准备 nnU-Net 数据

```bash
python prepare_dataset.py
```

输出目录：

```
nnunet_data/nnUNet_raw/Dataset001_Cornea/
├── imagesTr/
├── labelsTr/
├── imagesTs/
└── dataset.json
```

如果不想把原始 val 合并进训练集：

```bash
python prepare_dataset.py --no_val_merge
```

### 2. 训练 nnU-Net

默认 `--profile baseline` 使用原始 nnU-Net 训练行为。可选 `--profile dgx-spark` 会复用已有预处理、使用 250 epoch trainer、默认不导出 `.npz` 并跳过部分校验；此选项会改变训练长度，论文准备流程使用 baseline。

```bash
# 默认训练
python train.py

# 只训练一个 fold
python train.py --fold 0

# 已经预处理过时跳过预处理
python train.py --skip_preprocess --fold 0

# 使用指定 epoch 数的内置 trainer
python train.py --skip_preprocess --fold 0 --fast_epochs 250 --no_npz

# 从最近 checkpoint 继续训练
python train.py --skip_preprocess --fold 0 --continue_training

# 只打印将要执行的 nnU-Net 命令
python train.py --dry_run
```

常用参数：

| 参数 | 说明 |
| --- | --- |
| `--profile {dgx-spark,baseline}` | 运行 profile；`baseline` 保留原始 nnU-Net 行为。 |
| `--fold 0` / `--fold all` | 训练单个 fold 或所有 fold。 |
| `--device {cuda,cpu,mps}` | 训练设备。 |
| `--cuda_devices 0,1` | 设置 `CUDA_VISIBLE_DEVICES`。 |
| `--num_gpus 2` | 传给 nnU-Net 的 GPU 数量。 |
| `--fast_epochs 5...8000` | 使用 nnU-Net 内置的不同训练长度 trainer。 |
| `--trainer nnUNetTrainer_250epochs` | 手动指定 trainer 类名。 |
| `--no_npz` / `--save_npz` | 禁用或强制导出验证 softmax。 |
| `--disable_checkpointing` | 仅用于速度 benchmark；不会保存可复用最终权重。 |

训练结果位于：

```
nnunet_data/nnUNet_results/Dataset001_Cornea/
└── nnUNetTrainer...__nnUNetPlans__2d/
    ├── fold_0/checkpoint_final.pth
    ├── fold_1/checkpoint_final.pth
    └── ...
```

### 3. 推理

```bash
# 对文件夹中所有 .tif 图像预测，输出 PNG 掩模
python predict.py \
    --input  /path/to/tif_images/ \
    --output /path/to/output_masks/

# 预测单张图像
python predict.py \
    --input  image.tif \
    --output mask.png

# 指定迁移后的模型目录
python predict.py \
    --input  /path/to/tif_images/ \
    --output /path/to/output_masks/ \
    --model  /absolute/path/to/nnUNet_results/Dataset001_Cornea

# 使用指定 fold，并输出 TIF 掩模
python predict.py --input /path/to/tif_images/ --output /path/to/masks/ --fold 0 --format tif
```

Python API：

```python
from unet.predict import load_predictor, predict_single, predict_folder

predictor = load_predictor(
    model_dir="codes/unet/nnunet_data/nnUNet_results/Dataset001_Cornea",
    fold="all",
)

mask = predict_single("path/to/image.tif", predictor)  # uint8，取值 0/1
predict_folder("path/to/images", "path/to/masks", predictor, save_as="png")
```

### 4. 评估

```bash
python evaluate.py \
    --pred /path/to/predicted_masks/

python evaluate.py \
    --pred /path/to/predicted_masks/ \
    --gt   codes/datasets/CORN-1/labels/test_label/ \
    --save results/evaluation.csv
```

`evaluate.py` 接受 PNG/TIF 掩模，输出 Dice、IoU 和像素准确率。预测文件名的 stem 需要和对应 GT TIF 文件一致。

## 生成 PD 分割版数据集

`build_pd_unet_dataset.py` 会用训练好的 nnU-Net 模型处理 `codes/datasets/pd_dataset_en`，并把掩模图像写入 `codes/datasets/pd_dataset_unet_en`。脚本会保留原始目录层级，并将 JSON 中的图像路径改写为 PNG 掩模路径。

```bash
python build_pd_unet_dataset.py

python build_pd_unet_dataset.py \
    --source_root ../datasets/pd_dataset_en \
    --output_root ../datasets/pd_dataset_unet_en \
    --model_dir nnunet_data/nnUNet_results/Dataset001_Cornea \
    --overwrite
```

脚本会在存在时复制并改写 `patients_dataset_en.json`、`patients_dataset_test.json` 等元数据文件。

## 关键配置

主要默认值集中在 `config.py`：

| 配置 | 默认值 | 含义 |
| --- | --- | --- |
| `DATASET_ROOT` | `codes/datasets/CORN-1` | CORN-1 源数据路径。 |
| `DATASET_ID` | `1` | nnU-Net 数据集 ID。 |
| `DATASET_NAME` | `Dataset001_Cornea` | nnU-Net 数据集目录名。 |
| `NNUNET_CONFIG` | `2d` | 使用 nnU-Net 2D 配置。 |
| `FOLD` | `all` | 默认训练 fold。 |
| `IMAGE_SIZE` | `(384, 384)` | 预期 CORN-1 图像尺寸。 |
| `PIXEL_SPACING` | `(1.0, 1.0)` | NIfTI 转换时使用的显微图像像素间距。 |

迁移模型到新机器时，复制 `nnUNet_results/Dataset001_Cornea`，然后通过 `predict.py --model` 指定路径即可。`predict.py` 会自动设置 `nnUNet_raw`、`nnUNet_preprocessed` 和 `nnUNet_results` 环境变量。

## U-Net / GNN 分类实验的准备顺序

U-Net-only、U-Net-dual 和 GNN 都必须先用 CORN-1 训练 nnU-Net，再运行 `build_pd_unet_dataset.py` 生成 PD 分割图。训练默认执行全部 5 折，生成数据时使用现有已完成的折集成推理。分类数据需要完整的患者清单和图像，U-Net-dual 还需要对应原图。已有按此流程生成的完整数据可以直接使用。

也可全程从 `codes/` 运行，避免切换目录：

```bash
python -m unet.prepare_dataset
python -m unet.train --profile baseline
python -m unet.build_pd_unet_dataset
python -m experiments.unet_experiments.batch_experiments --variant all
python -m experiments.gnn_experiments.batch_experiments
```

U-Net-only / dual 实现位于本目录的 `only.py`、`dual.py`、`dual_view_*.py` 和亚型辅助模块中，通过 `experiments/unet_experiments/main.py` 与 `batch_experiments.py` 调用。
