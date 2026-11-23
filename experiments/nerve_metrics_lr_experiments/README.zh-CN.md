# Nerve-LR

[English](README.md)

论文表 1 的入口为 `python -m experiments.nerve_metrics_lr_experiments.paper --dataset-root /path/to/datasets`。每个外层训练折在三个内层折中按 macro-F1 选择 C（0.01、0.1、1、10），输出 `results/nerve_metrics_lr/paper.json`，含 15 折均值及描述性区间。以下为另一个固定 C 的批量入口。

固定使用 CNFD、CNBD、CNFL、CTBD、CNFA、CNFW、CNFractal、IWL、IWFA、IWFW、IWFrD 全部 11 个指标。每位患者的多次测量先取均值，再进行患者级分层交叉验证。每个训练折独立拟合中位数插补、标准化和 LR；不先在全数据上标准化，也不按结果挑选特征。

批量入口的 LR 使用 C=1、balanced 类别权重、lbfgs；这些参数固定，不使用验证结果调参。

在 codes 目录运行：

```bash
python -m experiments.nerve_metrics_lr_experiments.batch_experiments
# 指标表和患者 JSON 均在外部 datasets 下
python -m experiments.nerve_metrics_lr_experiments.batch_experiments --dataset-root /path/to/datasets
```

可编辑 batch_experiments.py 中 EXPERIMENTS。默认三个 PD 任务，5 折 × 3 次，随机种子 42/1042/2042。每组及 batch 根目录直接生成 `metrics_ci95.json/csv/md`；批次汇总另有 `combined_metrics_ci95.json/csv/md`。四项指标为患者级 accuracy、bacc、macro-F1、AUC，均附 95% CI。三分类 AUC 为 macro OvR，subtype 为 TD=0、非 TD=1。

默认输入：

- `datasets/pd_dataset_en/patients_dataset_en.json`：权威临床标签。
- `pd_patient_feature_measurements.csv`：63 名 PD 患者的 252 次测量。
- `non_pd_patient_feature_measurements.csv`：已对齐至 31 名白内障对照的 118 次测量。

不会读取 `non_pd_feature_measurements_unaligned.csv`，也不会用姓名、行号猜测患者对应关系。可通过 `--feature-csvs` 显式更换指标表，要求非空 patient_id 和全部 11 列；同时支持上述小写列名或 `_mean` 后缀。不应同时传入同一患者的原始表和其汇总表，以免重复计权。

当前 binary 指标队列为 94 人，stage/subtype 为 63 人。20 名健康对照缺少指标，会在 `cohort_audit.json` 中明确排除。**binary 不能直接声称与 110 名配对图像患者的 TopoDiSE 是相同队列比较**；需要共同患者子集或补齐测量。使用 `--require-complete-cohort` 可要求任何缺失都报错。

DM 支持相同 LR 流程，但本地没有已对齐的 DM 11 指标表。只有提供真实表后才可运行：

```bash
python -m experiments.nerve_metrics_lr_experiments.batch_experiments --tasks dm_binary \
  --patient-json /path/to/dm_patients_dataset_en.json --feature-csvs /path/to/dm_metrics.csv
```

DM 标签为 control=0、NDR/NPDR=1。程序不会从图像生成或虚构缺失的临床测量。

CI 使用与其余分类实验相同的按真实类别分层的患者 cluster bootstrap；重复之间使用同一组患者抽样权重。它衡量给定已训练模型的患者抽样不确定性，不表示重新训练的全部不确定性。当前 LR 无验证集调参，但其他深度基线有验证折选轮次，比较时仍应注明这一差异。
