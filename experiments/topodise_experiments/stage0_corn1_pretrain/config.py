"""Stage0 配置数据类。"""
from dataclasses import dataclass, asdict
from typing import Tuple, Optional


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
