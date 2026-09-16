"""Matched patient draws shared across repeats, seeds, Full and intervention."""
import numpy as np
from utils.confidence_intervals import scores,METRICS


def paired(reference,variant,classes,draws=2000,seed=20260915):
    def index(rows):
        result={}
        for r in rows:
            k=(int(r.get('pretrain_seed',42)),int(r['fold']),str(r['patient_id']))
            if k in result: raise ValueError('Duplicate prediction identity')
            result[k]=r
        return result
    a,b=index(reference),index(variant)
    if set(a)!=set(b): raise ValueError('Cannot pair different seeds/folds/patients')
    ids=sorted({k[2] for k in a});labels={}
    for k in a:
        if a[k]['y_true']!=b[k]['y_true']: raise ValueError('Mismatched comparison labels')
        label=int(a[k]['y_true'])
        if labels.setdefault(k[2],label)!=label: raise ValueError('Patient labels changed across repeats')
    groups={}
    for k in a:
        group=(k[0],int(a[k].get('repeat',(k[1]-1)//5)))
        groups.setdefault(group,[]).append(k)
    for keys in groups.values():
        if len(keys)!=len(ids) or {k[2] for k in keys}!=set(ids): raise ValueError('Incomplete repeat in paired comparison')
    index_id={p:i for i,p in enumerate(ids)}
    arrays=[]
    for keys in groups.values():
        keys=sorted(keys,key=lambda k:k[2]);item=[]
        for source in (a,b):
            rows=[source[k] for k in keys];item.append(([r['y_true'] for r in rows],[r['y_pred'] for r in rows],[r.get('ensemble_probs',r.get('patient_probs')) for r in rows]))
        arrays.append(item)
    def delta(weights=None):
        values=[]
        for x,y in arrays:
            before=scores(*x,classes,weights=weights);after=scores(*y,classes,weights=weights)
            values.append([after[m]-before[m] for m in METRICS])
        return np.mean(values,axis=0)
    point=delta();rng=np.random.default_rng(seed);y=np.array([labels[p] for p in ids]);strata=[np.flatnonzero(y==c) for c in np.unique(y)];samples=[]
    for _ in range(draws):
        selected=np.concatenate([rng.choice(s,len(s),replace=True) for s in strata]);samples.append(delta(np.bincount(selected,minlength=len(ids))))
    samples=np.array(samples);result={}
    for i,m in enumerate(METRICS):
        valid=samples[:,i][np.isfinite(samples[:,i])]
        result[m]=dict(delta=float(point[i]) if np.isfinite(point[i]) else None,ci95=np.quantile(valid,[.025,.975]).tolist() if len(valid) else [None,None])
    return dict(metrics=result,n_patients=len(ids),bootstrap_samples=draws,method='paired class-stratified patient-cluster bootstrap shared over seeds and repeats; conditional on trained models')
