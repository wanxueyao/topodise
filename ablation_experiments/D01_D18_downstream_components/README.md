# D01_D18_downstream_components

对应设计指南第 7 节。参数以本目录 JSON 为唯一消融定义；公共训练不复制。

| 编号 | 问题 | 优先级 |
|---|---|---|
| D01 | 无残差适配器 | core |
| D02 | 无融合微调 | core |
| D03 | 无适配器且无融合微调 | core |
| D04 | 全编码器微调 | core |
| D05 | mean MIL 替代 gated MIL | core |
| D06 | 图像对训练后患者聚合 | supplementary |
| D07 | 无图像统计 | core |
| D08 | 仅图像统计 | core |
| D09 | 打乱图像统计对应关系 | supplementary |
| D10 | 完全无原型模块 | core |
| D11 | 仅原型辅助训练 | core |
| D12 | 原型推断规则 | core |
| D13 | support-topk | supplementary |
| D14 | 关闭均衡患者包采样 | core |
| D15 | 去类别权重 | core |
| D16 | focal 改为加权 CE | core |
| D17 | balanced accuracy 与 accuracy 选择 checkpoint | core |
| D18 | TD 类额外权重及 focal gamma 的预定小网格 | supplementary |

从 `codes/` 执行：

```bash
python -m ablation_experiments.run --experiments D --dry-run
```

补充实验需显式选择编号或 `--include-supplementary`。执行顺序、依赖和复用条件见[总执行指南](../README.md)。
