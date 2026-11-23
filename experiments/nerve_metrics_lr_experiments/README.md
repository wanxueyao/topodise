# Nerve-LR

[中文说明](README.zh-CN.md)

To compute the Nerve-LR row in Table 1 from authorized measurements, run from the repository root:

```bash
python -m experiments.nerve_metrics_lr_experiments.paper --dataset-root /path/to/datasets
```

Under `pd_dataset_en/`, supply `patients_dataset_en.json`, `pd_patient_feature_measurements.csv` and `non_pd_patient_feature_measurements.csv`. Both CSVs require an aligned `patient_id` and all 11 nerve measurements. The paper includes 63 PD patients and 31 cataract controls in diagnosis, and the 63 PD patients in H&Y and subtype screening.

Average measurements within patient. For each outer training fold, fit a pipeline of median imputation, standardization and class-balanced logistic regression. Three stratified inner folds select C from 0.01, 0.1, 1 and 10 by macro-F1. The output `results/nerve_metrics_lr/paper.json` contains 15 fold scores per task, their means and descriptive 95% half-widths. Use `--output` to change the destination.

The separate `batch_experiments` entry supports fixed-C experiments, custom input tables and patient bootstrap reports.
