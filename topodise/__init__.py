"""TopoDiSE: Topology-preserving Diffusion Semantic Encoder for CCM.

See ``brains/TOPODISE_IMPLEMENTATION.md`` for the full design document.
"""
from .encoder import TopoDiSEEncoder, EncoderConfig
from .unet import ConditionalUNet, UNetConfig
from .topo_head import TopoHead
from .adapter import ResidualAdapter, attach_adapters
from .classifier import ClassifierHead, ProjectionHead, CoralHead, PrototypeHead, episodic_proto_loss
from .diffusion import DiffusionSchedule, q_sample, ddim_sample
from .feature_diffusion import FeatureDiffusionHead, q_sample_feature
from .losses import (
    supcon_loss, focal_bce_loss, focal_ce_loss, region_infonce_loss, nt_xent_loss,
    coral_loss, coral_predict,
)
from .topo_targets import compute_topo_target
from .ema import ModelEMA
from .freezing import apply_freeze_policy
from .logging_utils import TrainLogger, EMAMeter, RateMeter, count_params, grad_global_norm, env_banner

__all__ = [
    "TopoDiSEEncoder", "EncoderConfig",
    "ConditionalUNet", "UNetConfig",
    "TopoHead",
    "ResidualAdapter", "attach_adapters",
    "ClassifierHead", "ProjectionHead", "CoralHead", "PrototypeHead", "episodic_proto_loss",
    "DiffusionSchedule", "q_sample", "ddim_sample",
    "FeatureDiffusionHead", "q_sample_feature",
    "supcon_loss", "focal_bce_loss", "focal_ce_loss", "region_infonce_loss", "nt_xent_loss",
    "coral_loss", "coral_predict",
    "compute_topo_target",
    "ModelEMA",
    "apply_freeze_policy",
    "TrainLogger", "EMAMeter", "RateMeter", "count_params", "grad_global_norm", "env_banner",
]
