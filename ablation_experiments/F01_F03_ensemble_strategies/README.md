# F01_F03_ensemble_strategies

对应设计指南第 9 节。参数以本目录 JSON 为唯一消融定义；公共训练不复制。

| 编号 | 问题 | 优先级 |
|---|---|---|
| F01 | PD 分期 | core |
| F02 | TD-rest | core |
| F03 | 不确定性与失败样本 | supplementary |

从 `codes/` 执行：

```bash
python -m ablation_experiments.run --experiments F --dry-run
```

补充实验需显式选择编号或 `--include-supplementary`。执行顺序、依赖和复用条件见[总执行指南](../README.md)。
