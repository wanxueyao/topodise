# c8bf6b548656 复现说明

TopoDiSE 总入口使用一个三态 `reproducibility` 参数。三个状态使用相同的
Stage0、Stage1 和 Stage2 方法，只改变预训练的来源和确定性约束。

## 三种运行状态

### 1. strict：从零近似复刻

```bash
python -m experiments.topodise_experiments.batch_experiments \
  --section main_tasks \
  --reproducibility strict \
  --gpus 0 \
  --num-workers 4
```

`strict` 固定如下内容：

- Stage0：1,516 张 CORN-1 图像、224 px、200 epochs、batch 16、seed 42、4 workers。
- Stage1：182 组 control/control-cataract region pairs、80 epochs、batch 16、seed 42。
- Stage1 固定读取 Stage0 `ema_best.pt` 容器中的 raw `encoder`。
- 固定 Python、NumPy、PyTorch CPU/CUDA、DataLoader worker 和 shuffle 的随机状态。
- 关闭 cuDNN benchmark 和 matmul TF32，设置确定性 cuBLAS workspace，并要求
  PyTorch 使用确定性算子。
- 检查点保存和恢复全部 RNG 状态，DataLoader persistent workers 关闭。
- 禁止跨批次复用旧的 Stage0/1 checkpoint；只允许继续同一输出目录内由本次
  从零训练产生的中断检查点。

该模式在两张 RTX A6000 上从零得到了完全一致的 Stage0/1 学习状态和下游预测。
Stage0 最佳 epoch 为 100，loss 为 `0.11063678121749074`。原始的单患者熵路由/多数投票规则下：

| 任务 | Accuracy | Macro-F1 | Accuracy 95% CI | Macro-F1 95% CI |
|---|---:|---:|---:|---:|
| PD Stage | 65.61% | 62.95% | 56.08–75.13% | 50.70–72.90% |
| PD Subtype TD-rest | 70.90% | 69.23% | 63.49–77.78% | 61.16–76.37% |

置信区间使用 63 位患者聚类、50,000 次 percentile bootstrap（seed `20260911`）。

### 2. portable：从零设备自适应训练

```bash
python -m experiments.topodise_experiments.batch_experiments \
  --section main_tasks \
  --reproducibility portable
```

`portable` 保持相同的模型、数据、损失和训练参数，但 Stage0/1 不强制设置 RNG
或确定性后端；Stage2 保留原方法已有的 Torch seed。它适用于不同 GPU、CPU、MPS
和不同 PyTorch/CUDA 组合；多次运行可能有数值和准确率波动。该模式同样禁止跨批次
复用旧预训练权重。

### 3. checkpoint：使用历史最佳 c8 权重

```bash
python -m experiments.topodise_experiments.batch_experiments \
  --section main_tasks \
  --reproducibility checkpoint \
  --checkpoint-dir /path/to/c8_checkpoint_bundle
```

`checkpoint` 跳过 Stage0 和 Stage1，直接将目录中的 Stage1 `.pt` 作为全部 Stage2 候选的父权重。
Stage2 仍使用固定的论文方法和确定性执行路径。这是复现历史 c8 权重结果
PD Stage 69.31% 和 PD Subtype 74.07% 的推荐入口。

完整预训练包建议保持现有服务器结构：

```text
c8_checkpoint_bundle/
├── _experiment_def.json
├── summary.json
├── <name>_stage0/
│   └── ckpt/ema_best.pt
└── <name>_stage1/
    └── ckpt/ema_best.pt
```

程序的唯一强制要求是 Stage1 `.pt` 存在。Stage0 目录、`_experiment_def.json` 和
`summary.json` 用于溯源，可以省略。最小目录可以是：

```text
c8_checkpoint_bundle/
└── stage1/
    └── ema_best.pt
```

可用 `TOPODISE_CHECKPOINT_DIR=/path/to/c8_checkpoint_bundle` 代替命令行目录参数。程序会优先选择
`ema_best.pt`，其次是 `last.pt`；发现多个同优先级候选时直接报错。

## 历史配方与剩余差异

- 源代码快照：Git commit `6e688f2`。
- 历史 Stage0：epoch 100，loss `0.11062798957201711`。
- 历史 Stage1：epoch 77，loss `0.02590857886455276`。
- 历史环境：Python 3.10.19、PyTorch 2.9.1+cu128、cuDNN 91002、RTX A6000。
- 历史 Stage1 SHA-256：
  `45146fce10112b0abbc6c782d55099cb18593ec413019f4f52cd8d469725bf03`。

隔离实验已证明：使用历史 Stage0 时，当前 Stage1 可以逐参数完全复现历史 Stage1；
当前 Stage2 使用历史 Stage1 时可还原 69.31% / 74.07%。从零差异的来源是 Stage0。
原始 Stage0 只设置了 `torch.manual_seed(42)`，没有记录或强制确定性 CUDA 轨迹；
这条历史非确定性轨迹无法从现有文件反推。

## 数据指纹

- CORN-1：1,516 张图像，225,944,578 bytes，SHA-256
  `5df7ed3afc6188874e62f3903e779e44a02c93103fa0b40a7ebf4f8b67ef24d5`。
- PD JSON：98,309 bytes，SHA-256
  `d39a0f3c613d17612f94ce3c9651a2a80e47e5c10855873e6fc5966327dda9f0`。
- PD JSON 引用的 884 张图像：28,738,781 bytes，SHA-256
  `9744a982dd48ee6a4d462eb6f4a44dd0ca6105461b0f53f1455dc6296af0e0ec`。

`strict` 保证同一数据、代码、配置及兼容软硬件环境内的精确重跑，不承诺跨 GPU 架构、
CUDA/cuDNN 或 PyTorch 版本的逐位相同。当设备不支持确定性算子时，使用 `portable`。
