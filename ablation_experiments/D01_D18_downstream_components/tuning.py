"""D04.tuned: equal inner learning-rate search budgets for both adaptation policies.

Outer reporting retains SYS-C epoch selection. This is explicitly not SYS-N.
"""
from pathlib import Path
from copy import deepcopy
from dataclasses import asdict
import argparse
import numpy as np
from sklearn.model_selection import StratifiedKFold
from topodise.training.configs import Stage2Config
from topodise.training.stage2 import _make_dataset,_build_kfold_splits
from topodise.training.runtime.protocol import read,write
from topodise.training.hooks import training_hooks
from ..common.interventions import AblationHooks
from ..common.execution import run_job
from utils.confidence_intervals import scores


def tune(config,output,device='cpu'):
    cfg=Stage2Config(**config)
    if cfg.task not in ('pd_binary','pd_stage','pd_subtype_tdrest','dm_binary'): raise ValueError('Formal tasks only')
    classes=3 if cfg.task=='pd_stage' else 2
    with training_hooks(AblationHooks()): ds=_make_dataset(cfg,None);outer=_build_kfold_splits(ds,cfg)
    result={policy:[] for policy in ('limited','full')};search=[]
    def train(policy,lr,tr,va,name):
        values=deepcopy(config);values.update(lr=lr,eval_folds=1,kfold_repeats=1,output_dir=str(output/name),run_name='train')
        values['extensions']={**values.get('extensions',{}),'explicit_splits':[[tr,va]]}
        if policy=='full': values.update(use_adapters=False,finetune_encoder_policy='all')
        return run_job(dict(id=name,stage='stage2',config=values,dependencies=[]),device)['fold_results'][0]
    for fold,(train_indices,test_indices) in enumerate(outer,1):
        by_patient={}
        for i in train_indices: by_patient.setdefault(str(ds.samples[i][3]),[]).append(i)
        ids=sorted(by_patient);labels=[ds.samples[by_patient[p][0]][2] for p in ids]
        inner_n=min(3,min(np.bincount(labels,minlength=classes)))
        if inner_n<2: raise ValueError('Insufficient inner training patients')
        inner=list(StratifiedKFold(inner_n,shuffle=True,random_state=cfg.seed+fold).split(ids,labels))
        for policy in result:
            candidates=[]
            for factor in (.02,.1,1.):
                rows=[]
                for inner_fold,(a,b) in enumerate(inner,1):
                    tr=[i for j in a for i in by_patient[ids[j]]];va=[i for j in b for i in by_patient[ids[j]]]
                    r=train(policy,cfg.lr*factor,tr,va,f'outer{fold}_{policy}_lr{factor}_inner{inner_fold}')
                    rows.extend(r['best_patient_predictions'])
                score=scores([r['y_true'] for r in rows],[r['y_pred'] for r in rows],[r['patient_probs'] for r in rows],classes)['bacc']
                candidates.append((score,factor))
            selected=max(candidates,key=lambda x:(x[0],-x[1]))[1]
            r=train(policy,cfg.lr*selected,train_indices,test_indices,f'outer{fold}_{policy}_selected');r['fold']=fold;result[policy].append(r)
            search.append(dict(fold=fold,policy=policy,inner_candidates=candidates,selected_factor=selected,train_patients=ids))
    report=dict(protocol_id='SYS-C-innerLR-v1',reference_id='R-FT-TUNED',limitation='Outer validation still selects epochs as in SYS-C; only LR selection is train-internal',fold_results=result,search=search)
    write(output/'tuned_result.json',report);return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cpu');a=p.parse_args(argv);tune(read(a.config),a.output,a.device)

if __name__=='__main__': main()
