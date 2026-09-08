"""Stage1（RSCP）配置数据类。"""
from dataclasses import dataclass, asdict
from typing import Optional, Tuple


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
