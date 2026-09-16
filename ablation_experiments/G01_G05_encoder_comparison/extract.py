"""Deterministic native frozen features: no random Stage2 projection or adapters."""
from pathlib import Path
import argparse,json
import numpy as np
import torch
from topodise.training.hooks import training_hooks
from topodise.training.configs import Stage2Config
from topodise.training.runtime.full_stage2 import build_full_config
from topodise.training.runtime.protocol import write,file_hash,digest
from ..common.registry import ANCHORS,TASKS
from ..common.interventions import AblationHooks


def load_encoder(backend,checkpoint,device='cpu',seed=42):
    torch.manual_seed(seed)
    if backend in ('topodise','random_topodise'):
        from topodise.encoder import EncoderConfig
        from topodise.data import build_eval_transform
        if backend=='topodise':
            saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
            metadata=saved.get('parent_config') or saved['config']
            config=EncoderConfig(base_channels=metadata['encoder_base_channels'],channel_mult=tuple(metadata['encoder_channel_mult']),blocks_per_stage=tuple(metadata['encoder_blocks_per_stage']),z_dim=metadata['z_dim'])
        else: config=EncoderConfig();metadata={}
        encoder=AblationHooks().build_encoder(config,metadata)
        if backend=='topodise': encoder.load_state_dict(saved['encoder'],strict=True)
        encoder.eval().requires_grad_(False).to(device)
        return encoder,build_eval_transform(image_size=224)
    if backend=='simclr_resnet18':
        from .encoders.simclr import SimCLRResNet18Frozen
        encoder=SimCLRResNet18Frozen(Path(checkpoint)).to(device)
    elif backend=='dinov2_vits14':
        from .encoders.dinov2 import DINOv2ViTS14Frozen
        encoder=DINOv2ViTS14Frozen(Path(checkpoint),'vit_small_patch14_dinov2').to(device)
    else: raise ValueError(backend)
    from topodise.data import build_eval_transform
    gray_transform=build_eval_transform(image_size=224)
    def transform(image):
        gray=(gray_transform(image)+1)*.5
        return (gray.expand(3,-1,-1)-torch.tensor([.485,.456,.406])[:,None,None])/torch.tensor([.229,.224,.225])[:,None,None]
    return encoder,transform


@torch.no_grad()
def encode_pairs(encoder,backend,central,whorl):
    if backend in ('topodise','random_topodise'): return encoder(central,whorl)[0]
    return torch.cat([encoder.encode_region(central,'central'),encoder.encode_region(whorl,'whorl')],1)


def extract(backend,checkpoint,task,data,output,device='cpu',repeats=3,seed=42):
    from topodise.training.stage2 import _make_dataset,_build_kfold_splits
    from topodise.data.downstream_dataset import _load_gray
    bundle=build_full_config('',Path('/unused'),data/'pd_dataset_en/patients_dataset_en.json',data,tasks=[task],repeats=repeats)
    cfg=Stage2Config(**next(c['config'] for c in bundle['recipes'][0]['candidates'] if c['name']==ANCHORS[task]))
    ds=_make_dataset(cfg,None);encoder,transform=load_encoder(backend,checkpoint,device,seed)
    groups={}
    for i,s in enumerate(ds.samples): groups.setdefault(str(s[3]),[]).append(i)
    ids=sorted(groups);lookup={p:i for i,p in enumerate(ids)};features=[];labels=[];files=set()
    for pid in ids:
        values=[];pairs=groups[pid][:cfg.mil_max_bag]
        for i in pairs:
            c,w,y,_=ds.samples[i];files.update((c,w));labels_for_patient={ds.samples[j][2] for j in groups[pid]}
            if len(labels_for_patient)!=1: raise ValueError('Inconsistent patient labels')
            xc=transform(_load_gray(c)).unsqueeze(0).to(device);xw=transform(_load_gray(w)).unsqueeze(0).to(device)
            values.append(encode_pairs(encoder,backend,xc,xw).cpu())
        features.append(torch.cat(values).mean(0).numpy());labels.append(y)
    assignments=np.zeros((repeats,len(ids)),dtype=int)
    for i,(tr,va) in enumerate(_build_kfold_splits(ds,cfg)):
        for j in va: assignments[i//cfg.num_folds,lookup[str(ds.samples[j][3])]]=i%cfg.num_folds+1
    if (assignments==0).any(): raise ValueError('Incomplete patient fold manifest')
    output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output,features=np.stack(features),labels=labels,patient_ids=ids,fold_ids=assignments)
    write(output.with_suffix('.json'),dict(backend=backend,task=task,checkpoint_sha256=file_hash(checkpoint) if checkpoint else None,seed=seed,
        input_sha256={p:file_hash(p) for p in sorted(files)},pair_slots=cfg.mil_max_bag,aggregation='fixed pair mean; native representation',frozen=True,raw_weights=True))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--encoder',choices=['topodise','random_topodise','simclr_resnet18','dinov2_vits14'],required=True);p.add_argument('--checkpoint',type=Path);p.add_argument('--task',choices=TASKS,required=True);p.add_argument('--dataset-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cpu');a=p.parse_args(argv)
    if a.encoder!='random_topodise' and not a.checkpoint: p.error('Pretrained encoder requires --checkpoint')
    extract(a.encoder,a.checkpoint,a.task,a.dataset_root.resolve(),a.output,a.device)

if __name__=='__main__': main()
