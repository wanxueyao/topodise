# E01_E09_feature_diffusion

对应设计指南第 8 节。参数以本目录 JSON 为唯一消融定义；公共训练不复制。

| 编号 | 问题 | 优先级 |
|---|---|---|
| E01 | 无 Stage2 特征扩散 | core |
| E02 | 高斯特征扰动替代 | core |
| E03 | 无类别条件 | core |
| E04 | 无相似度门控 | core |
| E05 | 无增强分类损失 | core |
| E06 | 无生成支持库增强 | core |
| E07 | 无特征去噪损失 | supplementary |
| E08 | 门控方式 hard/soft、类别范围全体/少数类 | supplementary |
| E09 | E09 | supplementary |

从 `codes/` 执行：

```bash
python -m ablation_experiments.run --experiments E --dry-run
```

补充实验需显式选择编号或 `--include-supplementary`。执行顺序、依赖和复用条件见[总执行指南](../README.md)。
