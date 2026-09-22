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

### 匹配参照的适用范围

C03/C04 的 `M0.paired` 和 C05 的 `M0.matched_pairs` 继承原实验的适用条件。
DM 清单没有 central/whorl 区域身份，`dm_pair_mode=duplicate` 不能作为真实区域配对；
因此在 CANDIDATE 计划中，这些实验及匹配参照的 `dm_binary` 视图均为
`not_applicable`，不生成对应 Stage2 作业。PD 匹配参照及普通 `M0.default` 不受影响。
FINAL 沿用既有集成规则：不适用的候选保留未干预成员，不施加配对子队列或匹配预算变换。

旧版计划可能错误包含这些 DM 参照作业。运行中的锁定协议不能原地改写，也不能伪造
`DONE.json`；规划源码参与哈希校验，更新代码应在独立检出中进行，或先停止旧运行再明确迁移。
新协议应由修复后的规划器重新生成。此修复不修改已有协议、失败状态或服务器排除记录。

在仓库根目录使用 Python 3.10+ 验证规划回归，无需训练数据或 GPU：

```bash
python -m unittest discover -s tests -v
```
