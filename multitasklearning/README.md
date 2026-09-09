# MTL-MAE 模型

本目录保存论文 MTL-MAE 实验依赖的编码器、重建与分类分支、配置和数据加载代码。训练入口统一位于：

```bash
python -m experiments.mtl_experiments.batch_experiments --list
python -m experiments.mtl_experiments.batch_experiments --dry-run
python -m experiments.mtl_experiments.batch_experiments
```

完整流程保留 CORN-1 / PD 多任务预训练、PD binary/stage/subtype 分类和 DM 迁移。参数直接写在 `experiments/mtl_experiments/batch_experiments.py` 的 `EXPERIMENTS` 中；患者级划分、损失权重、迁移缓存和TD-rest 二分类由对应实验入口控制。

任务统一为 `pd_binary`、`pd_stage`、`pd_subtype`；其中 `pd_subtype` 只执行 TD=0 / Rest（PIGD+mixed）=1 二分类。
