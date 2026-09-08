# UNet / GNN paper baselines

本目录保留论文最终使用的三个 baseline：

| 方法 | 入口 | 输入 |
|---|---|---|
| UNet-only | `batch_experiments.py` | `pd_dataset_unet_en` 分割图 |
| UNet-dual | `dual_stream_batch.py` | 原图 + 对齐的 UNet 分割图 |
| GNN | `gnn_batch.py` | UNet 分割图骨架化后的图结构 |

三个入口均只从 `../paper_baseline_experiments.json` 读取论文采用的 binary/stage 三个 seed 和 subtype 5-fold x3 最佳配置，不再包含历史扫参组。

在 `codes/` 目录下运行：

```bash
RUN_SITE=auto RUN_CONCURRENCY_PER_GPU=1 python -m experiments.gnn_experiments.batch_experiments
RUN_SITE=auto RUN_CONCURRENCY_PER_GPU=1 python -m experiments.gnn_experiments.dual_stream_batch
RUN_SITE=auto RUN_CONCURRENCY_PER_GPU=1 python -m experiments.gnn_experiments.gnn_batch
```

三种方法都需要 `datasets/pd_dataset_unet_en/patients_dataset_en.json`。如果需要从 CORN-1 重建该数据集，请按根目录 README 和 `unet/README.md` 先训练 nnU-Net 并生成分割版 PD 数据。
