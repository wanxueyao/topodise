# I01_I09_sensitivity

对应设计指南第 12 节。参数以本目录 JSON 为唯一消融定义；公共训练不复制。

| 编号 | 问题 | 优先级 |
|---|---|---|
| I01 | Stage1 更新范围 | supplementary |
| I02 | Stage1 结构头策略 | supplementary |
| I03 | Stage1 去噪信号 | supplementary |
| I04 | 预训练损失权重 | supplementary |
| I05 | 适配器秩 | supplementary |
| I06 | 扩散强度 | supplementary |
| I07 | 无标签数据效率 | supplementary |
| I08 | 下游标注效率 | supplementary |
| I09 | 运行成本 | supplementary |

从 `codes/` 执行：

```bash
python -m ablation_experiments.run --experiments I --dry-run
```

补充实验需显式选择编号或 `--include-supplementary`。执行顺序、依赖和复用条件见[总执行指南](../README.md)。
