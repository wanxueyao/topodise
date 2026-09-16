"""D11/D12/D13/E06 exact-support replay from a selected Stage2 checkpoint."""
from pathlib import Path
import argparse
import torch
from topodise.classifier import PrototypeHead
from topodise.training.runtime.protocol import file_hash,write


@torch.no_grad()
def replay(checkpoint,mode='mean',top_k=3):
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False);cfg=saved['config']
    rows=saved['patient_predictions'];cache=saved.get('replay')
    if not cache or cache['support'] is None or cache['queries'] is None:
        raise ValueError('Exact replay cache absent: rerun Full with the new artifact contract; do not regenerate random supports')
    queries,support,labels=cache['queries'],cache['support'],cache['support_labels']
    cls=torch.tensor([r['classifier_probs'] for r in rows]);classes=cls.shape[1]
    head=PrototypeHead(z_dim=1,proj_dim=cfg['proto_dim'],metric=cfg['proto_metric'])
    if mode=='classifier': probabilities=cls
    else:
        if mode=='no_generated_support':
            keep=~torch.tensor(cache['support_generated'],dtype=torch.bool)
            if len(keep)!=len(support): raise ValueError('Support provenance mismatch')
            support,labels=support[keep],labels[keep];mode=cfg['proto_eval_mode']
        if mode=='mean':
            prototypes,present=head.compute_prototypes(support,labels,classes)
            logits=head.distance_logits(queries,prototypes,temp=cfg['proto_temp'])
        else: logits,present=head.support_logits(queries,support,labels,classes,temp=cfg['proto_temp'],mode=mode,top_k=top_k)
        logits=logits.masked_fill(present.unsqueeze(0)==0,-1e4)
        blend=cfg['proto_eval_blend'];probabilities=(1-blend)*cls+blend*logits.softmax(-1)
    out=[]
    for row,p in zip(rows,probabilities):
        cls_index=int(cfg.get('binary_threshold_class',-1))
        pred=(cls_index if p[cls_index]>=cfg.get('binary_threshold',.5) else 1-cls_index) if classes==2 and cls_index in (0,1) else int(p.argmax())
        out.append({**row,'fold':saved['fold'],'patient_probs':p.tolist(),'y_pred':pred})
    return dict(checkpoint_sha256=file_hash(checkpoint),selected_epoch=saved['epoch'],evaluation_unit='CANDIDATE',intervention_scope='inference_replay',predictions=out,
                reference_predictions=[{**r,'fold':saved['fold']} for r in rows],selection_limitation='conditional on the original selected checkpoint')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoints',nargs='+',type=Path,required=True);p.add_argument('--mode',choices=['classifier','mean','support_lse','support_topk','no_generated_support'],required=True);p.add_argument('--top-k',type=int,default=3);p.add_argument('--output',type=Path,required=True);a=p.parse_args(argv)
    results=[replay(x,a.mode,a.top_k) for x in a.checkpoints]
    folds=[r['predictions'][0]['fold'] for r in results]
    if len(set(folds))!=len(folds): raise ValueError('Duplicate replay folds')
    write(a.output,dict(mode=a.mode,fold_results=results,predictions=[p for r in results for p in r['predictions']]))

if __name__=='__main__': main()
