"""Matched sampler, train-only statistics randomization, and loss-unit control."""
from collections import Counter
import random


def balanced_slots(bags):
    counts=Counter(b[2] for b in bags)
    return max(counts.values())*len(counts) if counts else 0


def morphology_sources(train,val,cfg,epoch):
    if not cfg.shuffle_morphology_features: return {},{}
    if len(train)<2: raise ValueError('Shuffled statistics needs >=2 training patients')
    # Validation donors are training patients only, fixed across epochs.
    vrng=random.Random(cfg.seed+8713)
    val_map={pid:vrng.choice(train)[1] for pid,_,_ in val}
    trng=random.Random(cfg.seed+104729*(max(epoch,0)+1))
    train_map={pid:trng.choice([b for b in train if b[0]!=pid])[1] for pid,_,_ in train}
    return train_map,val_map


def instance_loss(z,morph,head,y,cfg,class_weights):
    if not cfg.extensions.get('instance_training'): return None
    import torch
    from topodise.training.stage2 import _ce_or_focal
    if z is None: raise ValueError('Instance training requires image features')
    features=torch.cat([z,morph.expand(z.size(0),-1)],1) if morph is not None else z
    labels=torch.full((len(features),),int(y),device=z.device,dtype=torch.long)
    return _ce_or_focal(head(features),labels,cfg,class_weights)
