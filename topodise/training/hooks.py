"""Scoped extension interface. Default hooks preserve formal training behavior.

Factories and policies are installed by callers; this module never imports a
research experiment. ContextVar prevents concurrent jobs leaking interventions.
"""
from contextlib import contextmanager
from contextvars import ContextVar


class TrainingHooks:
    def build_external_encoder(self, cfg, device):
        raise ValueError('An external encoder requires an explicitly installed encoder factory')

    def build_encoder(self, config, metadata):
        from topodise.encoder import TopoDiSEEncoder
        return TopoDiSEEncoder(config)

    def target(self, function, cfg, **kwargs):
        return function(**kwargs)

    def topology_loss(self, logits, target, cfg):
        from torch.nn.functional import binary_cross_entropy_with_logits
        return binary_cross_entropy_with_logits(logits.float(), target.float())

    def configure_pretrain(self, encoder, unet, head, cfg, stage):
        pass

    def dataset(self, dataset, cfg):
        return dataset

    def feature_diagnostics(self, raw, reference, gated, labels, cfg, scope):
        pass

    def feature_head(self, head, cfg):
        return head

    def sampling_budget(self, bags, cfg, default):
        return default

    def bag_order(self, bags, cfg, rng, default):
        return default

    def morphology_sources(self, train_bags, val_bags, cfg, rng, epoch):
        return {}, {}

    def fold_bags(self, train_bags, val_bags, cfg, fold):
        return train_bags, val_bags

    def split_indices(self, dataset, cfg):
        return None

    def batches(self, loader, cfg):
        return loader

    def instance_loss(self, z, morph, head, y, cfg, class_weights):
        return None

    def record_epoch(self, cfg, fold, epoch, predictions, directory):
        pass


_DEFAULT = TrainingHooks()
_HOOKS = ContextVar('topodise_training_hooks', default=_DEFAULT)


def current_hooks():
    return _HOOKS.get()


@contextmanager
def training_hooks(hooks):
    token = _HOOKS.set(hooks)
    try:
        yield hooks
    finally:
        _HOOKS.reset(token)
