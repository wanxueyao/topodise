# B01_B06_topology_targets

对应设计指南第 5 节。参数以本目录 JSON 为唯一消融定义；公共训练不复制。

| 编号 | 问题 | 优先级 |
|---|---|---|
| B01 | Stage0：GT、HGrad-dual、Frangi-dual、Sobel-dual | core |
| B02 | Stage0：presence-only、core-only、presence+core | core |
| B03 | Stage1：均值阈值-only、均值+标准差-only、双阈值 | core |
| B04 | Stage1：HGrad-dual、Frangi-dual、Sobel-dual | core |
| B05 | Stage0 目标空间打乱 | supplementary |
| B06 | Stage1 目标空间打乱 | supplementary |

从 `codes/` 执行：

```bash
python -m ablation_experiments.run --experiments B --dry-run
```

补充实验需显式选择编号或 `--include-supplementary`。执行顺序、依赖和复用条件见[总执行指南](../README.md)。
