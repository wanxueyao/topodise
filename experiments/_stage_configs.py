"""TopoDiSE stage configurations shared by experiment runners."""
from dataclasses import dataclass, asdict, field
from typing import Literal, Tuple, Optional


@dataclass
class Stage0Config:
    # --- 数据
    corn1_root: str = "datasets/CORN-1"
    # c8bf6b548656 参考配方（2026-06-10）
    image_size: int = 224
    num_workers: int = 4
    persistent_workers: bool = False
    prefetch_factor: int = 2
    # None 保留历史行为：CUDA 上锁页内存，其他设备不锁页。
    pin_memory: Optional[bool] = None
    non_blocking_transfers: bool = False
    expected_samples: Optional[int] = 1516

    # --- 模型维度
    encoder_base_channels: int = 64
    encoder_channel_mult: Tuple[int, ...] = (1, 2, 4, 8)
    encoder_blocks_per_stage: Tuple[int, ...] = (2, 2, 2, 2)
    z_dim: int = 512
    unet_model_channels: int = 64
    unet_channel_mult: Tuple[int, ...] = (1, 2, 2, 4)

    # --- 扩散过程
    T: int = 1000
    schedule: str = "cosine"

    # --- 训练
    epochs: int = 200
    batch_size: int = 16
    lr: float = 1e-4
    weight_decay: float = 1e-2
    grad_clip: float = 1.0
    ema_decay: float = 0.9999
    z_dropout: float = 0.1   # 无分类器引导训练

    # --- 损失权重
    lambda_simple: float = 1.0
    lambda_topo: float = 0.1

    # --- 拓扑监督目标（P0 改进；历史默认值为 hgrad）
    #   'hgrad'  ：单方向水平梯度占位目标（历史实现）
    #   'sobel'  ：梯度幅值脊线目标
    #   'frangi' ：多尺度 Frangi 管状度计算代理
    #   'gt'     ：CORN-1 神经纤维真值分割掩码；按文件名对齐
    #              labels/<split>_label/ 中的真实标签
    topo_target: str = "hgrad"
    frangi_sigmas: Tuple[float, ...] = (1.0, 2.0, 3.0)
    frangi_beta: float = 0.5
    frangi_c: float = 0.5

    # --- 视图对比辅助任务（P0；两个分支视图上的 SimCLR NT-Xent）
    use_view_contrastive: bool = False
    lambda_contrastive: float = 0.5
    tau_contrastive: float = 0.2

    # --- 输入输出
    output_dir: str = "results"
    run_name: str = "topodise_stage0"
    resume: bool = True
    resume_ckpt: str = ""
    strict_resume: bool = True
    device: str = "auto"
    # 仅控制是否在训练初始化时执行 torch.manual_seed(42)。
    reproducibility: bool = True
    amp: bool = False
    channels_last: bool = False
    fused_adamw: bool = False

    # 烟雾测试开关，由主入口控制
    smoke: bool = False

    def asdict(self):
        return asdict(self)


@dataclass
class Stage1Config:
    # --- 模型维度继承自父检查点；检查点内含超参
    parent_ckpt: str = ""              # Stage0 ema_best.pt 的路径
    # --- 数据
    pd_json: str = "datasets/pd_dataset_en/patients_dataset_en.json"
    region_pair_manifest: Optional[str] = None
    base_path: str = "datasets"
    patient_groups: Tuple[str, ...] = ("control", "control_cataract")
    allow_dm_disease: bool = False
    # c8bf6b548656 参考配方（2026-06-10）
    image_size: int = 224
    num_workers: int = 4
    persistent_workers: bool = False
    prefetch_factor: int = 2
    pin_memory: Optional[bool] = None
    non_blocking_transfers: bool = False
    expected_region_pairs: Optional[int] = 182

    # --- 训练
    epochs: int = 80
    batch_size: int = 16
    lr: float = 2e-5
    weight_decay: float = 1e-2
    grad_clip: float = 1.0
    ema_decay: float = 0.9999

    # --- 损失权重
    lambda_simple: float = 1.0
    lambda_topo: float = 0.05
    lambda_region: float = 0.5
    tau_region: float = 0.2

    # --- 拓扑监督目标（P0；与 Stage0 选项一致）
    topo_target: str = "hgrad"
    frangi_sigmas: tuple = (1.0, 2.0, 3.0)
    frangi_beta: float = 0.5
    frangi_c: float = 0.5

    # --- RSCP 微调时允许冻结权重的去噪器向编码器反传生成信号（P1）。
    #     关闭时保留历史的仅记录日志行为。
    train_denoiser: bool = False

    # --- 输入输出
    output_dir: str = "results"
    run_name: str = "topodise_stage1_rscp"
    device: str = "auto"
    # 仅控制是否在训练初始化时执行 torch.manual_seed(42)。
    reproducibility: bool = True
    # 参考 Stage1 从 Stage0 的原始 ``encoder`` 状态初始化。
    # ``ema`` 仍可用于显式消融，但不属于论文检查点配方。
    parent_encoder_weights: str = "raw"  # 可选 raw 或 ema
    amp: bool = False
    channels_last: bool = False
    fused_adamw: bool = False
    resume: bool = True
    resume_ckpt: str = ""
    strict_resume: bool = True

    smoke: bool = False

    def asdict(self):
        return asdict(self)


@dataclass
class Stage2Config:
    # --- parent ckpt (stage1 if available else stage0)
    parent_ckpt: str = ""

    # --- task: PD has 3 sub-tasks; DM has binary/subtype plus an auxiliary 3-way task.
    task: Literal["pd_binary", "pd_stage", "pd_subtype",
                  "pd_subtype_td_pigd", "pd_subtype_tdrest", "pd_subtype_pigd_mixed",
                  "pd_stage_ge2", "pd_stage_ge3",
                  "dm_binary", "dm_subtype", "dm_threeway"] = "pd_binary"

    # --- training mode:
    #   'single'    = original per-task image-pair training (eval may aggregate);
    #   'multitask' = joint training of the three PD heads sharing encoder/adapters
    #                 (the `task` field is ignored, uses MultiTaskDownstreamDataset);
    #   'mil'       = per-task Attention-MIL patient-bag training: each patient is a
    #                 bag of image-pairs, gated-attention pooled to one patient
    #                 embedding and trained directly against the patient label.
    #   'mil_dual_subtype' = subtype-only: train TWO binary Attention-MIL models on
    #                 the SAME folds, TD-vs-rest and PIGD-vs-rest, then fuse their
    #                 per-patient predictions into a 3-class (TD/PIGD/mixed) label.
    #                 Records both binary accuracies and the final 3-class accuracy.
    mode: Literal["single", "multitask", "mil", "mil_dual_subtype"] = "single"
    # task loss weights for multitask: (w_binary, w_stage, w_subtype)
    mt_task_weights: Tuple[float, float, float] = (1.0, 1.0, 1.0)

    # --- data
    pd_json: str = "datasets/pd_dataset_en/patients_dataset_en.json"
    base_path: str = "datasets"
    image_size: int = 224
    few_shot_k: int = -1    # -1 = use all data
    val_split: float = 0.2
    # K-fold controls (used when num_folds >= 2)
    # eval_folds <= 0 means evaluate all folds.
    num_folds: int = 5
    eval_folds: int = 2
    kfold_mode: Literal["patient", "image"] = "patient"
    # Repeated k-fold for variance reduction: run the whole k-fold cross
    # validation `kfold_repeats` times with different shuffles (seed offsets),
    # then aggregate all folds together. 1 = standard single k-fold.
    kfold_repeats: int = 1
    # The downstream datasets are tiny and decode each unique image exactly once
    # into a process-global cache (see downstream_dataset._GRAY_IMAGE_CACHE).
    # DataLoader worker processes do NOT share that cache and, without
    # persistent_workers, rebuild it from disk every epoch -- which defeats the
    # cache entirely. The MIL runner doesn't use a DataLoader at all. So we keep
    # data loading in the main process (num_workers=0) so the decode/tensor
    # caches are warmed once and reused across all folds and epochs.
    num_workers: int = 0

    # central<->whorl pairing for PD DownstreamDataset:
    #   'zip'       = pair central_i with whorl_i (min(K_c, K_w) pairs/patient)
    #   'cartesian' = all central x whorl combinations (up to K_c*K_w pairs)
    pair_mode: Literal["zip", "cartesian"] = "zip"
    # cap on pairs per patient under cartesian mode (0 = unlimited); when
    # exceeded a random subset is sampled to avoid large-patient dominance.
    max_pairs_per_patient: int = 0

    # --- patient-level aggregation at evaluation
    # When enabled, per-image-pair predictions are aggregated by patient_id to
    # produce a patient-level accuracy (clinically correct metric).
    eval_patient_agg: bool = False
    patient_agg: Literal["mean_logit", "vote"] = "mean_logit"
    # which validation metric drives best-epoch selection / early stopping.
    # Balanced variants are useful for small imbalanced transfer tasks where
    # overall accuracy can be maximized by predicting only the majority class.
    select_metric: Literal[
        "image", "patient", "balanced_image", "balanced_patient",
        "subtype_binary", "balanced_subtype_binary",
    ] = "image"

    # --- test-time augmentation (TTA) at evaluation.
    # Averages softmax over label-preserving geometric views (identity, hflip,
    # vflip, rot180) per image-pair before argmax / patient aggregation.
    use_tta: bool = False
    tta_views: int = 4                  # 1..4 deterministic views

    # --- Attention-MIL patient-bag training (mode='mil').
    mil_attn_hidden: int = 128          # hidden dim of gated-attention pooling
    mil_dropout: float = 0.25           # dropout on pooled patient embedding
    mil_bags_per_step: int = 8          # patient bags accumulated per optimizer step
    mil_max_bag: int = 16               # cap on image-pairs per patient bag
    mil_pooling: Literal["gated", "mean"] = "gated"
    # class-balanced bag sampling: per epoch, oversample minority-class patient
    # bags so each class contributes ~equally to the gradient. Cuts fold
    # variance on imbalanced small-sample tasks. Only affects MIL training.
    mil_balanced_bags: bool = False

    # --- P2: lightweight CCM morphology late-fusion for MIL.
    # Extracts fixed image/gradient statistics from each patient bag and projects
    # them into a small token concatenated with the learned MIL embedding.
    use_morphology_features: bool = False
    morph_embed_dim: int = 32
    shuffle_morphology_features: bool = False
    morphology_only: bool = False

    # --- P3: flexible pairwise-threshold ordinal head for pd_stage.
    # Trains two independent thresholds: y>=stage2 and y>=stage3, then predicts
    # stage = number of positive thresholds. Only active for mode='mil' + pd_stage.
    use_stage_thresholds: bool = False

    # --- P5: prototype/metric-learning head (borrowed from the few-shot
    # prototypical classifier). When enabled in MIL mode, an auxiliary episodic
    # prototypical loss is added on the patient-bag embeddings, and (optionally)
    # patient predictions are blended with distance-to-prototype classification.
    use_prototype: bool = False
    proto_dim: int = 128                 # metric embedding dim
    proto_dropout: float = 0.65          # strong feature dropout (prototype-style)
    proto_metric: Literal["euclidean", "cosine"] = "euclidean"
    proto_temp: float = 1.0              # distance temperature
    lambda_proto: float = 0.5            # weight of the episodic prototypical loss
    proto_min_per_class: int = 2         # min samples/class to form an episode
    # Evaluation blending in [0,1]: 0 = train-only regularizer (predict via
    # cls_head); >0 = blend softmax(cls_head) with softmax(prototype distance
    # logits) using prototypes built from the training fold. Nominal tasks only.
    proto_eval_blend: float = 0.0
    # 'mean' = one centroid per class; 'support_lse' = class-wise log-mean-exp
    # over the support bank; 'support_topk' = kNN-style average over nearest
    # support embeddings. The latter two better preserve multi-modal classes.
    proto_eval_mode: Literal["mean", "support_lse", "support_topk"] = "mean"
    proto_support_k: int = 3
    # If >0 and fdiff is enabled, add this many fdiff-generated support-bank
    # embeddings per training patient at eval time. This makes fdiff act as
    # explicit small-sample support augmentation for prototype classification.
    proto_fdiff_support_aug: int = 0
    # Optional per-class override for support augmentation counts. Example for
    # 3-class subtype: (0, 0, 8) targets class-2 only, addressing the observed
    # class-2 instability without bloating every class equally.
    proto_fdiff_support_aug_by_class: Tuple[int, ...] = ()

    # DM-only: how to construct (central, whorl) from a flat image list
    # 'duplicate' = use the same image for both branches
    # 'adjacent'  = pair image i with image i+1 from the same patient
    dm_pair_mode: Literal["duplicate", "adjacent"] = "duplicate"

    # --- adapters
    adapter_rank: int = 64
    use_adapters: bool = True
    finetune_encoder_policy: Literal["none", "high", "all"] = "none"

    # --- training
    epochs: int = 50
    batch_size: int = 16
    lr: float = 5e-4
    weight_decay: float = 1e-2
    grad_clip: float = 1.0
    classifier_dropout: float = 0.2

    # --- optimization controls
    scheduler: Literal["none", "cosine", "plateau"] = "cosine"
    min_lr: float = 1e-6
    warmup_epochs: int = 3
    early_stopping: bool = True
    patience: int = 8
    min_delta: float = 1e-4

    # --- loss
    lambda_ce: float = 1.0
    lambda_supcon: float = 1.0
    lambda_recon: float = 0.05
    tau_supcon: float = 0.1
    label_smooth_eps: float = 0.0       # label smoothing for CE; 0 = standard CE
    use_class_weights: bool = False     # inverse-frequency weighting for CE
    class_weight_multipliers: Tuple[float, ...] = field(default_factory=tuple)

    # Optional binary inference threshold. When binary_threshold_class >= 0 and
    # n_classes == 2, predict that class if P(class) >= threshold, otherwise the
    # other class. Useful for TD-vs-rest recall calibration (TD is class 0).
    binary_threshold_class: int = -1
    binary_threshold: float = 0.5

    # --- focal loss (multi-class). When enabled, CE is replaced by focal CE.
    use_focal: bool = False
    focal_gamma: float = 2.0

    # --- ordinal regression (CORAL, Cao et al. 2020) for the ordinal task
    # pd_stage (H&Y s1<s2<s3). Only takes effect when task=='pd_stage'; other
    # tasks fall back to standard nominal classification. Currently wired into
    # the MIL runner (mode='mil').
    use_ordinal: bool = False

    # --- surgical fine-tuning of the central<->whorl fusion (cross-attn + proj).
    # Backbone stays frozen; only the fusion module is unfrozen with a scaled lr.
    finetune_fusion: bool = False
    fusion_lr_scale: float = 0.1

    # --- weight EMA over trainable params (adapters + heads [+ fusion]).
    # Evaluation/checkpointing use the EMA weights when enabled.
    use_weight_ema: bool = False
    weight_ema_decay: float = 0.999

    # --- feature-space MixUp on encoder output z (CE path only).
    use_feature_mixup: bool = False
    mixup_alpha: float = 0.2
    mixup_start_epoch: int = 3

    # --- logit adjustment (Menon et al. ICLR'21)
    # logits_train = logits + tau_logit_adjust * log(class_prior)
    # logits used for eval/inference are *not* adjusted.
    use_logit_adjust: bool = False
    tau_logit_adjust: float = 1.0

    # --- feature-space diffusion augmentation (see topodise/feature_diffusion.py)
    use_feature_diffusion: bool = False
    fdiff_hidden: int = 256
    fdiff_layers: int = 3
    fdiff_t_emb_dim: int = 128
    fdiff_T_aug: int = 100              # sample t in [1, T_aug]; T_aug << schedule.T
    fdiff_k_aug: int = 1                # number of z_aug per sample per step
    fdiff_start_epoch: int = 3          # warmup before enabling the augmentation
    fdiff_class_cond: bool = False      # condition the head on class label
    fdiff_minority_only: bool = False   # only train aug-CE on minority-class samples
    fdiff_min_cos: float = 0.85         # sanity gate: cos(z_aug, z) > this else fallback
    fdiff_gate_mode: Literal["hard_cos", "soft_cos", "none"] = "hard_cos"
    fdiff_gate_threshold: float = 0.85
    fdiff_gate_temperature: float = 0.10
    # scale the gradient flowing back into the source feature anchor used by
    # feature diffusion. 0.0 = old full stop-grad; small >0 lets adapters move.
    fdiff_target_grad_scale: float = 0.0
    lambda_fdiff: float = 0.5           # weight on L_fdiff (denoising MSE)
    lambda_ce_aug: float = 0.3          # weight on L_ce computed on z_aug
    fdiff_lr_scale: float = 0.5         # lr for fdiff_head = lr * this

    # --- io
    output_dir: str = "results"
    run_name: str = "topodise_stage2_racf"
    seed: int = 42
    device: str = "auto"

    smoke: bool = False

    def asdict(self):
        return asdict(self)
