"""Same-width zero conditioning and forward-diffusion Gaussian replacement."""
from torch import nn


class GaussianGenerator(nn.Module):
    def forward(self, z_t, t, y=None):
        return z_t


class ZeroConditionGenerator(nn.Module):
    def __init__(self, generator):
        super().__init__();self.generator=generator

    def forward(self, z_t, t, y=None):
        # FeatureDiffusionHead(None) supplies zeros at the SAME input width.
        return self.generator(z_t,t,None)


def build(head,cfg):
    if cfg.extensions.get('feature_generator')=='gaussian': return GaussianGenerator()
    if cfg.extensions.get('zero_class_condition'): return ZeroConditionGenerator(head)
    return head
