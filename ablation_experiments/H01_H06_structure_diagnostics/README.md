# H01_H06_structure_diagnostics

对应设计指南第 11 节。参数以本目录 JSON 为唯一消融定义；公共训练不复制。

| 编号 | 问题 | 优先级 |
|---|---|---|
| H01 | 监督目标构造对比 | core |
| H02 | 冻结结构网格线性探针 | core |
| H03 | 冻结结构描述量回归 | core |
| H04 | 最近邻检索 | core |
| H05 | 跨阶段结构信息变化 | core |
| H06 | 特征空间与区域可分性 | supplementary |

从 `codes/` 执行：

```bash
python -m ablation_experiments.run --experiments H --dry-run
```

补充实验需显式选择编号或 `--include-supplementary`。执行顺序、依赖和复用条件见[总执行指南](../README.md)。
