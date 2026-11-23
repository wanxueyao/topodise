# TopoDiSE 正式实现：GT + HGrad 双阈值

Stage0 在 CORN-1 上使用 GT presence/core 两通道与扩散噪声预测；Stage1 使用独立无标签区域 manifest、区域对比损失和逐图双阈值 HGrad。Stage0 图像/mask 的随机几何增强同步。Stage1 冻结拓扑头，沿用 `rscp_high` 编码器更新策略，不训练去噪器。

HGrad 对水平相邻像素做绝对差，左侧补零，平均池化至 32×32。两个目标分别为响应大于逐图均值，以及大于逐图均值加样本标准差。它们是图像结构代理，不是真实骨架/分叉标签。

[English](README.md)

## 四个完整下游任务

所有任务从该组 Stage1 的 **raw encoder** 初始化；不是 EMA，也不是旧 c8 自动回退。

| 任务 | 定义 | 固定报告规则 |
|---|---|---|
| pd_binary | 对照/PD | US0 单模型 |
| pd_stage | 三级分期 | U0/U1/U2 最低熵路由 |
| pd_subtype_tdrest | TD=0，非 TD=1 | TDREST0/1/2/5/7 多数投票，平票按平均概率 |
| dm_binary（dm_transfer） | control=0，NDR+NPDR=1 | DMMIL3F1 单模型 |

当前入口的 DM 从 GT+HGrad 迁移；论文 DM 表来自使用 Frangi Stage1 的历史 DMMIL3F1 实验。随后在 DM 患者训练折内训练自己的下游模块；不使用 PD Stage2 分类头作为 DM 初始化。默认四任务均为患者五折、重复三次，划分种子 42/1042/2042。保留原各候选的早停、适配器、MIL、融合微调、形态统计、原型和特征扩散配置。

Stage2 不是整个编码器始终冻结：PD binary/stage 和 DM 会微调融合及投影，亚型候选遵循其原冻结配置。最终固定集成规则不依据标签挑选候选或 oracle 上界。

## 运行

以下命令均从 codes 目录执行，Python >=3.10，依赖见根 requirements.txt。

```bash
# 仅查看计划，不读权重、不写配置
python -m experiments.topodise_experiments.batch_experiments --dry-run

# 核对真实数据、mask 和所有下游患者划分，不启动训练
python -m experiments.topodise_experiments.batch_experiments --phase prepare

# 预训练一次，然后完整运行四任务
python -m experiments.topodise_experiments.batch_experiments --devices cuda:0

# 预训练已完成，仅继续本批次 Stage2
python -m experiments.topodise_experiments.batch_experiments --phase stage2 --devices cuda:0

# 从之前六组实验复用已经完成、配置和哈希匹配的 GT/HGrad-dual 预训练
python -m experiments.topodise_experiments.batch_experiments --phase stage2 \
  --pretrained-from /path/to/topology_targets_v3_full --devices cuda:0

# 只运行 DM，使用独立批次名（任务集合也是锁定协议的一部分）
python -m experiments.topodise_experiments.batch_experiments \
  --section dm_transfer --name gt_hgrad_dual_dm --devices cuda:0

# 显式外部数据目录
python -m experiments.topodise_experiments.batch_experiments \
  --dataset-root /path/to/datasets --devices cuda:0

# 汇总现有完成结果，无需训练
python -m experiments.topodise_experiments.batch_experiments --phase analyze
```

`--tasks` 可选择四个任务的子集；`--section main_tasks` 表示三个 PD 任务，`--section all` 表示四任务。`--full-stage2-repeats` 默认 3；修改任务集合、重复次数或训练配置请用新 `--name`。
`--pretrained-from` 可复用旧两/六组消融或主方法已完成的父权重，核对 Stage0/1 配置、完成标记、父子哈希以及输入内容；不会静默加载仅同名但来源不同的权重。

Stage0/1 配置在 `topodise/training/runtime/batch_experiments.py` 顶部，四任务固定候选在 `recipes.py`，训练实现仍在三个 stage 的 main.py。各 stage/batch_experiments.py 只是同一主流程的阶段入口。

## 自动指标及 95% CI

每个任务完成后自动生成：

- `task_summary.json`：患者预测、逐折指标、完整统计定义与四项 CI。
- `metrics_ci95.csv` / `metrics_ci95.md`：Accuracy、BAcc、Macro-F1、AUC 的点估计和 95% CI。
- 整批 `analysis/metrics_ci95.csv`：方法 × 任务 × 预训练种子直接可用的结果表。

**点估计**：每次重复合并其全部折的患者 OOF 预测，计算指标，再在重复之间取均值。不把不同重复投票成一个新模型，不用验证选择字段冒充 accuracy。多数投票的 Accuracy/F1 使用实际投票类别；AUC 使用对应的连续平均概率，熵路由使用所选候选概率。

**95% CI**：默认 2000 次真实类别分层的患者 cluster percentile bootstrap（`--bootstrap-samples`），同一次抽样对所有重复使用同一患者权重。同一患者的多次预测不当作多个独立患者。二分类 AUC 的正类是类别 1；stage 为 macro one-vs-rest AUC。缺少必要类别时 AUC 输出 null，而不是 0。

该点估计与原“逐折指标直接平均”口径可能略有不同；逐折均值/标准差作为辅助信息保留，不应和新表混用。CI 条件于已经训练的模型，不包含重新训练的不确定性，也不能修正验证折选最佳 epoch 的乐观偏差。当前完整 Stage2 仍在验证折上选 epoch 并报告该折结果，不能表述成独立外层测试。

## 结果与续跑

默认保存到 `experiments/topodise_experiments/results/gt_hgrad_dual_full_v1/`，其中 `pretrain_seed42/stage0/gt`、`stage1/gt_hgrad_dual`、`stage2/gt_hgrad_dual/{任务}/{候选}` 相互对应。Stage0/1 用 last.pt 恢复；Stage2 复用完成候选和已完成折，未完成折重跑。配置、源码和检查点哈希不符会报错，避免不同方法结果混用。

论文 PD 表采用 `analysis/summary.json` 的逐折均值及 `analysis/fold_metrics.csv`。
