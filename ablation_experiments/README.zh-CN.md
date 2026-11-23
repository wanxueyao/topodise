# TopoDiSE 论文消融复现

[English](README.md) · [环境与数据](../README.zh-CN.md)

| 位置 | 实验和统计单位 |
|---|---|
| 正文表 3、附录 E | Stage0 × Stage1 目标，完整 PD 系统，预训练种子 42，15 折均值 |
| 正文表 4、附录 F | A06，Stage0 掩膜损失开关，三个种子的冻结线性探针配对 AP 差值 |
| 附录 C | 从主实验同一批候选预测比较单候选、概率平均和最终规则 |
| 附录 E | M0、A01–A05、C03/C04、部分 D/E 组件，三个预训练种子分别评估 US0、U1、TDREST5 |
| 附录 E、F | D11/D12/E06 重放同一批 U1 检查点；H05 在种子 42 比较 Stage0 与 Stage1 |

## 目标组合

在仓库根目录执行。下面的命令固定 Stage0 GT，仅切换 Stage1 目标。去掉 `--variants` 可运行六组。所有组合沿用正式下游候选。原对照的 Frangi 对应 `frangi`，论文 HGrad 对应 `hgrad_dual`。

```bash
python -m ablation_experiments.paper_targets --dry-run
python -m ablation_experiments.paper_targets --dataset-root /path/to/datasets --variants gt_frangi gt_hgrad_dual --devices cuda:0
```

从 `ablation_experiments/results/paper_pd_targets/analysis/summary.json` 读取 `accuracy_mean`、`macro_f1_mean`。候选与推理规则比较使用 F01/F02。输入 JSON 包含 `task` 和完整的 `candidates`（H&Y 三个、亚型五个），各候选包含 Stage2 配方中的 `name`、`config` 和 `result`（对象或结果文件路径）。论文数值取各折指标的均值。

```bash
python -m ablation_experiments.analyze --experiment F01 --bundle /path/to/candidates.json --output inference.json
```

## 组件和探针训练

```bash
python -m ablation_experiments.run --experiments paper --tasks pd_binary pd_stage pd_subtype_tdrest \
  --pretrain-seeds 42 43 44 --repeats 3 --unit CANDIDATE --dataset-root /path/to/datasets \
  --output /path/to/paper-components --dry-run
```

将 `--dry-run` 换为 `--execute --devices cuda:0` 开始训练。保留相同总计划，加 `--only A06` 可单独运行 Stage0 掩膜开关。C03/C04 使用同一配对队列的 M0.paired，其他组件与对应的 M0 候选比较，每次重复先合并五折患者预测，再跨重复和种子取均值。扩展研究编号保留在 `--experiments all` 中。

D11、D12、E06 每次传入同一预训练种子的全部 15 个 U1 已选检查点，其中包含查询和支持库。种子 42、43、44 分别执行。以下用 Bash 数组 `fold_checkpoints` 传递一组路径：

```bash
python -m ablation_experiments.analyze --experiment D11 --checkpoints "${fold_checkpoints[@]}" --output classifier_seed42.json
```

## 冻结 CORN-1 结构探针

在本地对 CORN-1 生成划分清单。沿用原始训练、验证、测试目录，按此顺序去除完全重复的图像，保持采集批次完整。按计划中列出的每个 A06 检查点分别提取特征并训练探针：

```bash
python -m ablation_experiments.H01_H06_structure_diagnostics.build_manifest --root /path/to/datasets/CORN-1 --output results/probe_split.json
python -m ablation_experiments.H01_H06_structure_diagnostics.extract \
  --manifest results/probe_split.json \
  --checkpoint /path/to/gt_off_seed42.pt --output off/features.npz --protocol STR-A --device cuda:0
python -m ablation_experiments.analyze --experiment H02 --features off/features.npz --output off/probe
# GT 开启时重复以上两步，输出到 on/。
python -m ablation_experiments.analyze --experiment H05 --reference off/probe/predictions.npz \
  --variants on/probe/predictions.npz --output gt_on_minus_off.json
```

Stage0 与 Stage1 对照使用种子 42 的 GT Stage0、完整 Stage1 和同一探针划分。H03 使用相同特征训练骨架描述量探针，将 H02 命令中的实验编号换为 H03 即可。附录 D 的共同 Core Stage2 对照使用[当时的共享下游实现](https://github.com/wanxueyao/topodise/tree/e7dc1791dfb7457e1faf209e18fdfeeb6c7f7b67/experiments/unified_stage2_encoder_comparison)。
