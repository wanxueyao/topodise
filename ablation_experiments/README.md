# TopoDiSE 消融实验执行指南

全部消融按设计指南的 **9 组、68 个编号**组织；各编号 JSON 定义干预，`topodise/training/` 提供唯一的三阶段训练实现。本目录只维护实验执行、统计和诊断，不包含测试目录或论文排版生成器。正文/附录的展示分配在工作区 `brains/topodise_ablation_experiment_design_guide.md`；服务器只拿到本代码仓库时，也可以按本文安排运行。

| 目录 | 内容 |
|---|---|
| `A01_A07_pretraining_objectives/` | 阶段、预训练损失、结构头重置交互 |
| `B01_B06_topology_targets/` | 结构目标、通道、双阈值、空间置乱 |
| `C01_C05_encoder_architecture/` | 融合、共享骨干、区域输入、配对预算 |
| `D01_D18_downstream_components/` | 适配、MIL、统计、原型、类别不平衡 |
| `E01_E09_feature_diffusion/` | 特征生成、条件、门控、支持库、交互 |
| `F01_F03_ensemble_strategies/` | 固定集成、单候选、失败分析 |
| `G01_G05_encoder_comparison/` | 外部编码器、冻结线性/MLP/PCA 探针 |
| `H01_H06_structure_diagnostics/` | 目标图、结构预测、描述量、检索 |
| `I01_I09_sensitivity/` | 策略、敏感性、数据效率、运行成本 |
| `common/` | 注册、依赖调度、干预注入、输入审计和配对统计 |

## 1. 服务器准备与执行原则

从 `codes/`（本 Git 仓库根目录）执行全部命令。Python **3.10+**；先安装适合服务器 CUDA 的 PyTorch，再安装 `requirements.txt`。数据通过 `--dataset-root` 指定，目录至少包含：

```text
datasets/
  CORN-1/                                   # 图像与对应 GT mask
  unlabeled_dataset_en/unlabeled_dataset_en.json
  pd_dataset_en/patients_dataset_en.json
  dm_dataset_en/dm_patients_dataset_en.json   # 运行 DM 时需要
```

清单中的图像路径相对于数据根目录。Stage1 只使用独立无标签来源；不能用 PD 有标签队列补缺失数据。DINOv2 权重和结构探针的分组标注清单另行准备，见第 5 节。

**给服务器 Codex 的约定：**先检查数据、GPU、磁盘和已有产物，生成计划并记录 Git commit；按下表分批执行。不要把 `--experiments all --execute` 当作第一条训练命令，也不要按成绩改参照、换种子或重选可视化案例。`--dry-run` 只检查配置与依赖，不证明数据/权重可用；真实执行会审计输入。本文没有声称真实消融已经跑完。

## 2. 推荐先后顺序

| 顺序 | 批次 | 前置依赖与复用 | 完成判据 |
|---|---|---|---|
| 0 | 固定总计划、数据路径、任务、种子及预算 | 无需训练；先检查计划中的 N/A 和待提供输入项 | 保存 `protocol` 预览，确定运行资源 |
| 1 | M0 Full | Stage0 → Stage1 → Stage2；先得到后续共同参照 | 所需任务/种子/重复均完成，有患者预测和检查点 |
| 2 | 核心组件：A05、A02、A01.default、A03、A04、D01、D05、D07、E01、D10 | 自动使用第 1 步的匹配父作业；A02/A05 要重训上游 | 核心有效单元齐全；未启用组件的单元为 N/A |
| 3 | 直接证据：D11/D12/E06、G02/G03、H01–H05 | M0/核心变体检查点；重放用同一选中模型的精确支持库，探针用冻结表示 | 重放覆盖全折；探针、结构图有独立产物和来源 |
| 4 | 编码器与最终系统：G01、F01/F02 | 外部编码器权重；F 需要全体正式候选预测 | G01 完整 Stage2 比较与冻结探针分别报告；固定集成齐全 |
| 5 | 扩展机制：A06、B01–B04、C01–C05、D02/D03/D04.fixed/D08/D14–D17、E02–E05 | 复用可用父权重；A06/B02 的训练只到 Stage0，还需单独探针 | 各变体使用正确参照，训练和后处理均完成 |
| 6 | 补充：A07、A01.source_continue、B05/B06、D04.tuned/D06/D09/D13/D18、E07–E09、F03、G05、H06、I 组及其余补充子变体 | 核心完成后再做；I09 直接读取实际运行成本 | 补充项独立标记；G04 仍待决定，不自动运行 |

第 3、4 步可在对应检查点完成后，使用空闲 GPU/CPU 与第 2、5 步并行；不要让两个调度进程写同一批次根目录。`priority=core` 表示设计中保留，并不表示全部都必须在第二批跑完。

## 3. 固定总计划，按编号分批执行

**`--experiments` 定义总计划；`--only` 仅选择本次执行范围，并自动补齐参照和上游依赖。** 续跑时保持总计划参数不变，只改变 `--only` 和设备。所有批次共用同一个 `--output`，相同作业校验通过后直接复用。

下面使用 Bash 数组，减少分批时漏写参数；在服务器同一 Bash 会话执行。把 `/srv/topodise` 改为实际路径，预算不足时在首次执行前确定种子数与任务范围。

```bash
# bash；此示例预先登记核心及补充，但不会自动启动所有实验
ABLATION_ARGS=(
  --experiments all --include-supplementary
  --tasks pd_binary pd_stage pd_subtype_tdrest dm_binary
  --pretrain-seeds 42 43 44 --repeats 3 --unit CANDIDATE
  --dataset-root /srv/topodise/datasets
  --output /srv/topodise/results/ablation_v2
)

python -m ablation_experiments.run "${ABLATION_ARGS[@]}" \
  --dry-run --write-plan /srv/topodise/ablation_plan.json

# 第 1 批：完整方法参照
python -m ablation_experiments.run "${ABLATION_ARGS[@]}" \
  --only M0.default --devices cuda:0 cuda:1 --execute

# 第 2 批：核心组件；已完成的 M0、Stage0/1 自动校验并跳过
python -m ablation_experiments.run "${ABLATION_ARGS[@]}" \
  --only A05 A02 A01.default A03 A04 D01 D05 D07 E01 D10 \
  --devices cuda:0 cuda:1 --execute

# 第 5 批：扩展机制，先预览再用相同参数把 --dry-run 换为 --execute
python -m ablation_experiments.run "${ABLATION_ARGS[@]}" \
  --only A06 B01 B02 B03 B04 C D02 D03 D04.fixed D08 D14.default D15 D16 D17.default E02 E03 E04 E05 \
  --devices cuda:0 cuda:1 --dry-run

# 第 6 批示例：结构监督交互，自动采用 A07.11 参照
python -m ablation_experiments.run "${ABLATION_ARGS[@]}" \
  --only A07 --devices cuda:0 cuda:1 --execute
```

`--only A01` 会选择总计划内 A01 的全部子变体；只跑无 Stage1 用 `A01.default`。`--only D04` 同样包括 tuned 待输入项。G04 不在 `all` 中，即使加 `--include-supplementary` 也不会纳入；以后决定保留时应另建明确计划。

每张卡同时运行一个作业；调度器按依赖波次推进，同波次作业并行，父波次完成后开始子波次。`--only` 选择的是科学变体，不是单个 epoch 或折。它也会训练该变体所需的匹配参照。某些任务组件未启用时标为 `not_applicable`，不硬凑一条“无变化消融”。

**待输入项不会自动后处理。** 总计划中的 `requires_input` / `requires_probe_data` 是待办清单；`--execute` 只调度训练作业。A06/B02 会先完成 Stage0，G/H/F、重放、调参等需要第 5 节的专门入口。没有训练任务的 `--only H02` 不会运行结构探针。

## 4. 复用规则与禁止混用的情况

| 干预位置 | 可以复用 | 必须重新执行 |
|---|---|---|
| Stage0：A02/A05、B01/B05、C01/C02、I04 的 Stage0 权重等 | 数据清单与固定划分 | 受影响的 Stage0 → Stage1 → Stage2 |
| Stage1：A03/A04、B03/B04/B06、I01–I03/I07 等 | 同种子、同配置的 Stage0 | 对应 Stage1 → Stage2 |
| Stage2：C03–C05、D 的训练项、E 的训练项、I05/I06/I08 等 | 同种子 M0 Stage0/Stage1 | 对应 Stage2；不能沿用 Full 的预测 |
| A01.default | M0 Stage0 | 跳过 Stage1，单独训练 Stage2 |
| A06 | 同配置的 M0/A05 Stage0（实际键完全相同时） | 冻结探针；无需 Stage1/Stage2 |
| D11/D12/D13/E06 | Full 选中 checkpoint、查询嵌入、支持库 | 仅重放推断；不重新训练或采样生成支持 |
| F01/F02/F03 | 完整正式候选的逐患者预测 | 固定规则汇总/失败分析，无需重训 |
| G02/G03/G05、H 组 | 指定 checkpoint 的冻结表示与匹配提取配置 | 线性/PCA/结构探针、检索或绘图；随机编码器有独立参照 |
| I09 | 实际运行的 `cost.json`、折元数据、计时 | 成本汇总，无需训练 |

同一计划内按完整配置、父作业、种子与数据路径生成作业键；执行时再核验源码、输入文件与父权重哈希。**仅算法名称相同或结果接近，不构成复用依据。** 显式扩展字段与隐式默认值未必得到同一键，以 `protocol.json` 的 job ID 为准；不要人工合并。

额外参照：C03/C04 用配对患者子队列 Full，C05 用匹配配对预算 Full；A07 使用结构头重置后的 `R-RESET11`，不能当成正式冻结结构头的 M0。D17 必须重新保存固定训练轨迹，再在同一轨迹上比较 Accuracy/BAcc；不能用 M0 的单个最佳 checkpoint 代替。D04.tuned 双方进行同预算内层调参，独立于 D04.fixed。

**复用边界：**当前编号调度器只自动复用同一总计划、同一输出根目录的作业；不同根目录、CANDIDATE/FINAL、正式主方法目录和 G01 专用目录之间没有统一缓存导入器。不要复制/改名权重或伪造 DONE 来跳过审计。G01 的旧式 `--topodise-pretrained-from` 只接受其兼容的历史批次结构，不能直接指向本编号 DAG 的 `runs/`。另起计划前先核对这项成本。

默认 **CANDIDATE** 固定 US0/U1/TDREST5/DMMIL3F1；**FINAL** 保留每任务全部正式候选（1/3/5/1 个）和固定集成。候选级 Full 不等于最终系统 Full。F 组需要 FINAL 来源；可用 G01 的 TopoDiSE 完整结果，或单独运行：

```bash
python -m ablation_experiments.run --experiments M0 --unit FINAL \
  --tasks pd_stage pd_subtype_tdrest --pretrain-seeds 42 43 44 --repeats 3 \
  --dataset-root /srv/topodise/datasets --output /srv/topodise/results/full_final \
  --devices cuda:0 cuda:1 --execute
```

这条命令使用独立协议，不会自动复用前述 CANDIDATE 目录。提前确定需要哪些 FINAL 结果，避免为每个 F 子项重复跑完整系统。

## 5. 依赖产物的实验入口

下列 `/path/to/...` 是待替换的实际路径，不是可直接运行的默认值。先从 `protocol.json` 的 `views[].members[].job_id` 或 `analyses[].checkpoint` 定位来源，记录变体、任务、种子、fold、checkpoint 哈希。不同任务/种子分别运行后处理，不要混在一个输入列表中。

### 推断重放、固定集成、调参

```bash
# D11 同入口使用 --experiment D11；D13、E06 类推
python -m ablation_experiments.analyze --experiment D12 \
  --checkpoints /path/to/best_fold1.pt /path/to/best_fold2.pt \
  --output /path/to/D12_replay.json

python -m ablation_experiments.F01_F03_ensemble_strategies.analyze \
  --bundle /path/to/candidate_bundle.json --output /path/to/ensemble.json

python -m ablation_experiments.D01_D18_downstream_components.tuning \
  --config /path/to/full_stage2_config.json --output /path/to/D04_tuned --device cuda
```

重放列表仅示意；完整报告须覆盖该任务/种子的全部重复和折。输入必须保存所选模型真实使用的支持库，不能重新抽噪声、重新选 epoch；E06 仅从同一库中去除生成条目。

`candidate_bundle.json` 格式：`{"task":"pd_stage","candidates":[{"name":"正式候选完整名称","config":{...},"result":"result.json绝对路径"},...]}`。提供全体正式候选的实际配置及全部折结果，模块会检查患者、标签、概率列和成员完整性。F01/F02/F03 共用该输入与汇总产物。

### 外部编码器和冻结表示

```bash
# 独立完整系统比较；默认 seed42，repeats=3；不会读取编号 DAG 的结果
python -m ablation_experiments.G01_G05_encoder_comparison.run \
  --dataset-root /srv/topodise/datasets --dry-run
python -m ablation_experiments.G01_G05_encoder_comparison.run \
  --dataset-root /srv/topodise/datasets --devices cuda:0 cuda:1

# 一个编码器/任务/种子提取一次；G02/G05 可共用原始特征
python -m ablation_experiments.G01_G05_encoder_comparison.extract \
  --encoder topodise --checkpoint /path/to/stage1/ckpt/ema_best.pt \
  --task pd_binary --dataset-root /srv/topodise/datasets \
  --device cuda:0 --output /path/to/features.npz
python -m ablation_experiments.analyze --experiment G02 \
  --features /path/to/features.npz --output /path/to/linear.json
python -m ablation_experiments.analyze --experiment G05 \
  --features /path/to/features.npz --pca-dim 128 --output /path/to/pca128.json
```

G01 的编码器设置在 `config.py`，本地路径覆盖可通过 `--config configs路径.yaml` 提供，样例在该组 `configs/default.yaml`。DINOv2 需要兼容的本地权重，缺失时不回退随机模型；SimCLR 缺权重时按 CORN-1→独立无标签区域流程预训练。G01 可按 `--phase prepare/pretrain/stage2/analyze` 分阶段，阶段间保持名称、编码器列表和配置一致；它目前不支持编号入口的多预训练种子参数，不能把 seed42 结果声称为三种子比较。

G03 将提取器改成 `--encoder random_topodise`，不传 checkpoint，然后运行同一个 G02 线性探针。外部编码器使用 `simclr_resnet18` / `dinov2_vits14` 及其 checkpoint。A06/B02 使用计划中各自 Stage0 的 checkpoint 走同一提取/探针流程。

每患者固定均值向量；TopoDiSE 使用原生 z_sem，外部编码器用两区原生特征拼接，不用 Stage2 随机投影。标准化/PCA 在每个内层训练子集重拟合，分类器 C={.01,.1,1,10,100} 在训练内选择，外层只报告。128 仅为维度示例，运行前固定所有比较编码器都可达到的共同维度；不得预先在全数据拟合 PCA。

### 结构证据与可视化

准备含 `samples` 的 JSON，每行至少有 `id,image,mask,group_id,split`；split 为 train/validation/test，各组不得跨集合。缺少身份时不能把每幅图 ID 当成患者 ID。案例 JSON 在查看结果前固定为 `{"case_ids":["id1","id2"]}`。

```bash
python -m ablation_experiments.H01_H06_structure_diagnostics.extract \
  --manifest /path/to/structure_manifest.json --encoder topodise \
  --checkpoint /path/to/stage1/ckpt/ema_best.pt --protocol STR-A \
  --device cuda:0 --output /path/to/structure_features.npz
python -m ablation_experiments.analyze --experiment H02 \
  --features /path/to/structure_features.npz --output /path/to/Full_probe
python -m ablation_experiments.H01_H06_structure_diagnostics.visualize \
  --sources /path/to/Full_probe/predictions.npz /path/to/A05_probe/predictions.npz \
  --cases /path/to/cases.json --output /path/to/structure_comparison.png
```

H03/H04 可共用同一 `structure_features.npz`，分别输出描述量回归和检索；H05 为同一分组下 Stage0/Stage1 各自提取、运行 H02 后的配对比较；H06 使用带 `region_labels` 的分组 NPZ。H01 使用提取产物加 `--kind targets` 画监督目标；检索图加 `--kind retrieval --retrieval-report ...`。所有模型使用同样本、同色阶，全评估集 AP 单独报告。此类图是实验诊断，保留在代码内。

**STR-A** 衡量预训练分布上的可恢复性；**STR-B** 必须额外提供 `--pretraining-inventory`，包含 `complete: true`、完整预训练 `image_sha256`、`group_ids` 与 `checkpoint_sha256`，并核验测试来源不重叠。不能把已见过全 CORN-1 的 M0 事后改标 STR-B；要么所有模型按留出来源重新预训练，要么准备独立标注集。

## 6. 产物、续跑和交接

- `protocol.json`：冻结的总计划、科学视图和源码哈希；`configs/`：逐作业配置。
- `runs/<job_id>/input_audit.json`、`run_contract.json`：输入文件、患者折、父权重及源码锁。
- `runs/<job_id>/train/ckpt/`：Stage0/1 权重；`ema_best.pt` 沿用历史文件名，当前锚定训练保存的是 raw 权重，不凭文件名猜 EMA；`last.pt` 用于恢复。
- `runs/<job_id>/train_<task>/ckpt/`：Stage2 全部折的模型、支持库及查询嵌入。
- `runs/<job_id>/result.json`、`DONE.json`：训练结果与产物哈希；目录存在不代表完成。
- `logs/<job_id>.log`、`status.json`、`cost.json`：日志、状态和实际成本。
- `analysis/results.json`：已完成 SYS 结果、配对差值、缺失项及待输入分析清单。

分批完成后，汇总中的 `pending` 正常反映尚未执行的总计划单元；单独完成 G/H/F 后处理也不会自动改写该清单。服务器交接须另记这些产物的路径和来源，不能仅凭进程退出码或顶层 `complete` 判断全部科学实验完成。

```bash
python -m ablation_experiments.common.reporting /srv/topodise/results/ablation_v2
python -m ablation_experiments.analyze --experiment I09 \
  /srv/topodise/results/ablation_v2 --output /srv/topodise/results/ablation_v2/analysis/costs.json
```

恢复时使用原参数和同一输出根目录：已完成作业重新核验后跳过，Stage0/1 从有效 `last.pt` 恢复，Stage2 复用已完成折；未完成折可能重跑，不承诺任意 batch 精确续训。不要同时对同一根目录启动两个执行器。源码、数据、任务、种子、重复数或总计划变化会触发锁冲突；需新批次，不能删除锁绕过检查。整批迁移服务器时绝对路径变化也会影响协议，Git 只传源码，不传数据/权重/结果。

SYS 默认仍为 **SYS-C**：验证折同时用于选 epoch 和报告成绩，不能表述为独立外层测试。每次完整 OOF 后跨重复/种子汇总；配对 bootstrap 对同一患者保持相同抽样，置信区间不代替训练种子差异。最终交接记录：代码 commit、完整命令、数据根目录、输出根目录、完成/失败/待输入编号、精确产物路径及下一批 `--only` 范围。
