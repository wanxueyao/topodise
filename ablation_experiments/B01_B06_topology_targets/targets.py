"""Matched dual-threshold targets and stochastic information-destruction control."""
import torch
import torch.nn.functional as F
from topodise.topo_targets import _sobel, _frangi_vesselness, _binarize


def target(function, cfg, **kwargs):
    mode=kwargs['mode'];x=kwargs['x'];size=kwargs.get('out_size',32)
    if mode in ('frangi_dual','sobel_dual'):
        if mode=='frangi_dual':
            response=_frangi_vesselness(x,kwargs.get('sigmas',(1.,2.,3.)),kwargs.get('beta',.5),kwargs.get('c',.5))
        else:
            gx,gy=_sobel(x);response=(gx.square()+gy.square()+1e-12).sqrt()
        value=torch.cat([_binarize(response,size,0.),_binarize(response,size,1.)],1)
    else: value=function(**kwargs)
    if cfg.extensions.get('shuffle_target'):
        # New permutation per sample and visit; shared channels preserve nesting.
        value=torch.stack([v.flatten(1)[:,torch.randperm(v[0].numel(),device=v.device)].reshape_as(v) for v in value])
    return value


def loss(logits, value, cfg):
    mode=cfg.extensions.get('target_channels','both')
    channels={'presence':[0],'core':[1],'both':[0,1]}[mode]
    return F.binary_cross_entropy_with_logits(logits[:,channels].float(),value[:,channels].float())
