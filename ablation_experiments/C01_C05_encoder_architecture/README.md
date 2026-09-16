# C01_C05_encoder_architecture

对应设计指南第 6 节。参数以本目录 JSON 为唯一消融定义；公共训练不复制。

| 编号 | 问题 | 优先级 |
|---|---|---|
| C01 | 去 cross-attention | core |
| C02 | 两分支共享骨干权重 | core |
| C03 | 只提供 central 信息 | core |
| C04 | 只提供 whorl 信息 | core |
| C05 | 配对方式 zip vs cartesian | core |

从 `codes/` 执行：

```bash
python -m ablation_experiments.run --experiments C --dry-run
```

补充实验需显式选择编号或 `--include-supplementary`。执行顺序、依赖和复用条件见[总执行指南](../README.md)。
