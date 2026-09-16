"""TopoDiSE public API with lazy imports for dependency-free planning."""
from importlib import import_module

_EXPORTS = {'TopoDiSEEncoder': 'encoder', 'EncoderConfig': 'encoder', 'ConditionalUNet': 'unet', 'UNetConfig': 'unet', 'TopoHead': 'topo_head', 'ResidualAdapter': 'adapter', 'attach_adapters': 'adapter', 'ClassifierHead': 'classifier', 'ProjectionHead': 'classifier', 'CoralHead': 'classifier', 'PrototypeHead': 'classifier', 'episodic_proto_loss': 'classifier', 'DiffusionSchedule': 'diffusion', 'q_sample': 'diffusion', 'ddim_sample': 'diffusion', 'FeatureDiffusionHead': 'feature_diffusion', 'q_sample_feature': 'feature_diffusion', 'supcon_loss': 'losses', 'focal_bce_loss': 'losses', 'focal_ce_loss': 'losses', 'region_infonce_loss': 'losses', 'nt_xent_loss': 'losses', 'coral_loss': 'losses', 'coral_predict': 'losses', 'compute_topo_target': 'topo_targets', 'ModelEMA': 'ema', 'apply_freeze_policy': 'freezing', 'TrainLogger': 'logging_utils', 'EMAMeter': 'logging_utils', 'RateMeter': 'logging_utils', 'count_params': 'logging_utils', 'grad_global_norm': 'logging_utils', 'env_banner': 'logging_utils'}
__all__ = list(_EXPORTS)

def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    value = getattr(import_module("." + _EXPORTS[name], __name__), name)
    globals()[name] = value
    return value
