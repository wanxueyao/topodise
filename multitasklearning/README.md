# 多任务学习系统

本目录实现了用于帕金森病图像分析的 U-Net 双分支多任务学习系统。模型共享一个编码器，一条分支用于角膜神经/边界分割与自监督重建，另一条分支用于 PD 相关分类任务。

本 README 已整合原 `FILE_MANIFEST.md`、`QUICKSTART.md` 和 `PROJECT_SUMMARY.md` 的核心内容，作为当前目录的主要入口文档。

## 系统能力

核心功能：

- 共享 ResNet/EfficientNet 编码器，输出多尺度特征。
- 分支 A：分割、MAE 风格重建、Frangi/Hessian 伪标签支持。
- 分支 B：PD 分类，支持 `single_stream`、`dual_stream`、`mixed_single_stream` 三种分类架构。
- PD 数据集支持患者级 K-fold 划分，避免同一患者图像跨训练/验证泄漏。
- CORN-1 分割数据集支持图像和掩模同步增强。
- 提供训练、推理、评估、示例和系统测试脚本。

分类任务标签：

| 任务 | 标签 |
| --- | --- |
| PD 二分类 | `0=control`，`1=PD` |
| HY 分级 | `0-4`，对应 HY 1-5；控制组为 `-1` |
| 亚型分类 | `0=TD`，`1=PIGD`，`2=mixed`；控制组为 `-1` |

## 当前目录结构

```
codes/multitasklearning/
├── README.md                    # 当前整合文档
├── __init__.py                  # 模块公开导出
├── config.py                    # MTLConfig 和预定义配置
├── encoder.py                   # ResNet/EfficientNet U-Net 编码器
├── segmentation_branch.py       # 分割解码器和 MAE 重建器
├── classification_branch.py     # 单流、双流、混合单流分类器
├── unet_multitask.py            # 完整多任务模型和构建器
├── multitask_data_loader.py     # CORN-1、PD、K-fold、多任务数据加载
├── multitask_trainer.py         # 训练器、早停、checkpoint、混合精度
├── multitask_utils.py           # Frangi/Hessian、指标、损失函数
├── inference.py                 # 推理、评估、模型选择
├── helpers.py                   # 随机种子等辅助函数
├── train_multitask.py           # 命令行训练脚本
├── examples.py                  # 使用示例
└── test_system.py               # 功能测试/冒烟测试
```

## 快速开始

使用包式导入时建议从 `codes/` 运行；直接训练脚本时也可以进入 `codes/multitasklearning/`。

安装核心依赖：

```bash
pip install torch torchvision numpy scipy scikit-image opencv-python pillow scikit-learn
```

运行测试和示例：

```bash
cd codes
python -m multitasklearning.test_system
python -m multitasklearning.examples
```

训练命令：

```bash
cd codes/multitasklearning

# 默认：ResNet50 + dual_stream + standard 配置
python train_multitask.py

# 轻量级快速原型
python train_multitask.py --config lightweight --mode single_stream --epochs 20 --batch_size 16

# dual_stream 的第 0 折训练
python train_multitask.py \
    --mode dual_stream \
    --encoder resnet50 \
    --epochs 100 \
    --batch_size 32 \
    --lr 1e-4 \
    --kfold 0 \
    --kfold_splits 5

# EfficientNet 大模型配置
python train_multitask.py --config large --encoder efficientnet_b4 --mode dual_stream
```

运行 5 折：

```bash
for fold in 0 1 2 3 4; do
    python train_multitask.py --kfold "$fold" --kfold_splits 5 --mode dual_stream
done
```

常用 CLI 参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--mode` | `dual_stream` | 分类架构：`single_stream`、`dual_stream`、`mixed_single_stream`。 |
| `--encoder` | `resnet50` | CLI 支持 `resnet18`、`resnet50`、`efficientnet_b4`。 |
| `--config` | `standard` | 预定义配置：`lightweight`、`standard`、`large`。 |
| `--corn1_root` | `datasets/CORN-1` | CORN-1 分割数据路径。 |
| `--pd_root` | `datasets/pd_dataset_en` | PD 数据集路径。 |
| `--kfold` | `0` | 患者级验证 fold 编号。 |
| `--output_dir` | `checkpoints` | checkpoint 输出目录。 |

## 数据格式

### CORN-1

当前加载器使用本仓库的 CORN-1 split 结构：

```
datasets/CORN-1/
├── images/
│   ├── train_orig/
│   ├── val_orig/
│   └── test_orig/
└── labels/
    ├── train_label/
    ├── val_label/
    └── test_label/
```

图像和标签文件名应一一对应。标签支持 TIF/PNG，加载时会二值化。

### PD 数据集

`load_pd_dataset()` 会优先寻找 `patients_dataset_en.json`，找不到时回退到 `metadata.json`。当前患者级加载器使用如下 schema：

```json
[
  {
    "patient_id": "patient_001",
    "group": "pd",
    "stage": 2,
    "subtype": "TD",
    "image_paths": {
      "central": ["pd_dataset_en/patient_001/central_1.jpg"],
      "whorl": ["pd_dataset_en/patient_001/whorl_1.jpg"]
    }
  }
]
```

支持的分组过滤模式为 `pd_control`、`pd_control_cataract`、`pd_all`。兼容类 `PDDataset` 仍支持旧版扁平 `image_path` metadata 格式。

## Python API 速查

创建模型：

```python
from multitasklearning import UNetMultiTask

model = UNetMultiTask(
    encoder_type="resnet50",
    classification_mode="dual_stream",
)
```

前向传播：

```python
import torch

images = torch.randn(4, 3, 256, 256)
outputs = model(images, training=True, use_mae=True)

segmentation = outputs["segmentation"]
pd_logits = outputs["pd_logits"]
stage_logits = outputs["stage_logits"]
subtype_logits = outputs["subtype_logits"]
```

加载数据：

```python
from multitasklearning.multitask_data_loader import load_corn1_dataset, load_pd_dataset

corn_train, corn_val = load_corn1_dataset(
    "datasets/CORN-1",
    batch_size=32,
    augment=True,
)

pd_train = load_pd_dataset(
    "datasets/pd_dataset_en",
    split="train",
    kfold_idx=0,
    kfold_splits=5,
    batch_size=32,
    datasets_base="datasets",
)

pd_val = load_pd_dataset(
    "datasets/pd_dataset_en",
    split="val",
    kfold_idx=0,
    kfold_splits=5,
    batch_size=32,
    datasets_base="datasets",
)
```

通过 API 训练：

```python
from multitasklearning import MTLConfig, UNetMultiTask
from multitasklearning.multitask_trainer import MultiTaskTrainer

config = MTLConfig(num_epochs=100, batch_size=32, learning_rate=1e-4)
model = UNetMultiTask(
    encoder_type=config.encoder_type,
    classification_mode=config.classification_mode,
)

trainer = MultiTaskTrainer(config, model, pd_train, pd_val)
trainer.train()
```

推理和评估：

```python
from multitasklearning.inference import MTLInference, MTLEvaluator

inferencer = MTLInference(model)
seg = inferencer.infer_segmentation(images)
cls = inferencer.infer_classification(images)
joint = inferencer.infer_joint(images)

evaluator = MTLEvaluator(model)
seg_metrics = evaluator.evaluate_segmentation(corn_val)
cls_metrics = evaluator.evaluate_classification(pd_val)
```

伪标签工具：

```python
from multitasklearning.multitask_utils import FrangiFilter, HessianPseudoLabeler, batch_pseudo_labels

frangi = FrangiFilter(sigma_range=(1, 10))
enhanced = frangi(image_np)

hessian = HessianPseudoLabeler(sigma=2.0)
pseudo = hessian(image_np)

batch_masks = batch_pseudo_labels(images_np, method="frangi")
```

## 架构概览

```
输入图像 [B, 3, H, W]
        |
        v
共享编码器：ResNet / EfficientNet
        |
        +-----------------------------+
        |                             |
        v                             v
分支 A                           分支 B
分割 / MAE 重建                  分类
        |                             |
骨架/边界掩模                    PD / HY 分级 / 亚型 logits
```

分类架构对比：

| 模式 | 参数量 | 速度 | 适用场景 |
| --- | --- | --- | --- |
| `single_stream` | 最少 | 最快 | 快速原型或资源受限环境。 |
| `dual_stream` | 最多 | 较慢 | 最大化利用分割特征和解码器特征。 |
| `mixed_single_stream` | 中等 | 中等 | 平衡速度和性能。 |

默认多任务损失：

```
L = w_seg L_seg + w_mae L_mae + w_pseudo L_pseudo
  + w_pd L_pd + w_stage L_stage + w_subtype L_subtype
```

默认任务权重位于 `MTLConfig`：

| 权重 | 默认值 |
| --- | --- |
| `loss_weight_segmentation` | `0.3` |
| `loss_weight_mae` | `0.2` |
| `loss_weight_pseudo` | `0.1` |
| `loss_weight_pd_classification` | `0.2` |
| `loss_weight_stage` | `0.1` |
| `loss_weight_subtype` | `0.1` |

## 配置预设

```python
from multitasklearning.config import get_config_lightweight, get_config_standard, get_config_large

light = get_config_lightweight()  # ResNet18, batch_size=16, 50 epochs
std = get_config_standard()       # ResNet50, batch_size=32, 100 epochs
large = get_config_large()        # EfficientNet-B4, batch_size=64, 150 epochs
```

自定义配置：

```python
from multitasklearning import MTLConfig

config = MTLConfig(
    encoder_type="resnet50",
    classification_mode="dual_stream",
    batch_size=32,
    num_epochs=100,
    learning_rate=1e-4,
    mixed_precision=True,
    gradient_accumulation_steps=1,
)
```

## 模块速查

| 文件 | 主要对象 | 作用 |
| --- | --- | --- |
| `config.py` | `MTLConfig`，预设配置函数 | 模型、数据、训练和任务权重配置。 |
| `encoder.py` | `UNetEncoder`，`ResNetEncoder`，`EfficientNetEncoder` | Backbone 特征提取。 |
| `segmentation_branch.py` | `SegmentationBranch`，`MAEReconstructor` | 分割解码与重建分支。 |
| `classification_branch.py` | `ClassificationBranch`，各类 stream classifier | PD、HY、亚型分类头。 |
| `unet_multitask.py` | `UNetMultiTask`，`MTLBuilder` | 集成模型与模型工厂。 |
| `multitask_data_loader.py` | `CORNDataset`，`PDImageDataset`，加载器函数 | 数据集构建和患者级 K-fold。 |
| `multitask_trainer.py` | `MultiTaskTrainer`，`EarlyStopping` | 训练、验证、checkpoint、混合精度。 |
| `multitask_utils.py` | `FrangiFilter`，`HessianPseudoLabeler`，指标函数 | 伪标签、指标和多任务损失。 |
| `inference.py` | `MTLInference`，`MTLEvaluator`，`ModelSelector` | 推理、评估、模型比较。 |
| `helpers.py` | `set_seed` | 可复现性辅助函数。 |
| `train_multitask.py` | CLI 入口 | 端到端训练脚本。 |
| `examples.py` | 示例函数 | 常见使用方式演示。 |
| `test_system.py` | 测试函数 | 导入、建模、前向传播、工具函数冒烟测试。 |

## 常见任务

只训练分类任务：

```python
model = UNetMultiTask(classification_mode="dual_stream")
trainer = MultiTaskTrainer(config, model, pd_train, pd_val)
trainer.train()
```

只获取分割输出：

```python
with torch.no_grad():
    seg = model.get_segmentation(images)
```

只获取分类输出：

```python
with torch.no_grad():
    cls = model.get_classification(images)
```

获取分类概率：

```python
with torch.no_grad():
    outputs = model(images)
    pd_probs = torch.softmax(outputs["pd_logits"], dim=1)
    stage_probs = torch.softmax(outputs["stage_logits"], dim=1)
```

K-fold 循环：

```python
results = []
for fold_idx in range(5):
    train_loader = load_pd_dataset(
        "datasets/pd_dataset_en",
        split="train",
        kfold_idx=fold_idx,
        kfold_splits=5,
    )
    val_loader = load_pd_dataset(
        "datasets/pd_dataset_en",
        split="val",
        kfold_idx=fold_idx,
        kfold_splits=5,
    )

    model = UNetMultiTask()
    trainer = MultiTaskTrainer(config, model, train_loader, val_loader)
    trainer.train()
    results.append(trainer.history)
```

## 性能与调试建议

CUDA 显存不足：

```python
config.batch_size = 16
config.gradient_accumulation_steps = 2
config.encoder_type = "resnet18"
```

快速原型：

```python
from multitasklearning.config import get_config_lightweight
config = get_config_lightweight()
```

收敛慢：

```python
config.learning_rate = 5e-4
config.warmup_epochs = 10
config.loss_weight_segmentation = 0.5
```

固定随机种子：

```python
from multitasklearning.helpers import set_seed
set_seed(42)
```

验证输出形状：

```python
x = torch.randn(2, 3, 256, 256)
outputs = model(x)
print(outputs["segmentation"].shape)
print(outputs["pd_logits"].shape)
print(outputs["stage_logits"].shape)
print(outputs["subtype_logits"].shape)
```

## 测试

从 `codes/` 运行：

```bash
python -m multitasklearning.test_system
```

也可以在 `codes/multitasklearning/` 下直接运行：

```bash
python test_system.py
```

## 开发注意事项

- 新增主要模块后，同步更新 `__init__.py` 的公开导出。
- PD metadata 优先使用 `patients_dataset_en.json`；`metadata.json` 仅作为兼容回退。
- PD 验证默认使用患者级 K-fold，避免图像级数据泄漏。
- CORN-1 分割标签来自显式的 `labels/<split>_label` 目录。
- 原文件清单、快速开始和项目总结的内容已经整合进本 README。
