# A01_A07_pretraining_objectives

对应设计指南第 4 节。参数以本目录 JSON 为唯一消融定义；公共训练不复制。

| 编号 | 问题 | 优先级 |
|---|---|---|
| A01 | 无 Stage1 | core |
| A02 | 无 Stage0 图像扩散 | core |
| A03 | 无 Stage1 结构约束 | core |
| A04 | 无 Stage1 区域损失 | core |
| A05 | 全程无结构监督 | core |
| A06 | Stage0 GT 的独立贡献 | core |
| A07 | A07 | supplementary |

从 `codes/` 执行：

```bash
python -m ablation_experiments.run --experiments A --dry-run
```

补充实验需显式选择编号或 `--include-supplementary`。执行顺序、依赖和复用条件见[总执行指南](../README.md)。
