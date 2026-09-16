# G01_G05_encoder_comparison

对应设计指南第 10 节。参数以本目录 JSON 为唯一消融定义；公共训练不复制。

| 编号 | 问题 | 优先级 |
|---|---|---|
| G01 | 编码器 × 完整 Stage2 | core |
| G02 | 编码器 × 冻结线性分类 | core |
| G03 | 冻结随机编码器参照 | core |
| G04 | 编码器 × 冻结小型 MLP | pending |
| G05 | 统一维度表示对照 | supplementary |

从 `codes/` 执行：

```bash
python -m ablation_experiments.run --experiments G --dry-run
```

补充实验需显式选择编号或 `--include-supplementary`。G04 不进入自动批次。执行顺序、依赖和复用条件见[总执行指南](../README.md)。
