"""Linear grid decoding, descriptor regression and held-out cosine retrieval."""
from pathlib import Path
import argparse,copy
import numpy as np
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score,roc_auc_score
from topodise.training.runtime.protocol import write,file_hash,read


def split_indices(data):
    split=data['splits'];groups=data['groups'];result=[np.flatnonzero(split==s) for s in ('train','validation','test')]
    if any(not len(i) for i in result): raise ValueError('Empty structure partition')
    if any(set(groups[a])&set(groups[b]) for j,a in enumerate(result) for b in result[j+1:]): raise ValueError('Group leakage')
    return result


def grid_metrics(target,probabilities):
    rows=[]
    for y,p in zip(target,probabilities):
        channels=[]
        for a,b in zip(y,p):
            union=np.logical_or(a,b>=.5).sum()
            channels.append(dict(ap=float(average_precision_score(a.ravel(),b.ravel())) if a.any() else None,
                auc=float(roc_auc_score(a.ravel(),b.ravel())) if len(np.unique(a))==2 else None,
                iou=float(np.logical_and(a,b>=.5).sum()/union) if union else None))
        rows.append(channels)
    return rows


def grouped_interval(values,groups,seed=42,n=2000):
    values=np.array([np.nan if v is None else v for v in values],float);groups=np.asarray(groups);ids=np.unique(groups)
    rng=np.random.default_rng(seed);draws=[]
    for _ in range(n):
        sample=np.concatenate([values[groups==g] for g in rng.choice(ids,len(ids),replace=True)])
        if np.isfinite(sample).any(): draws.append(float(np.nanmean(sample)))
    return dict(mean=float(np.nanmean(values)) if np.isfinite(values).any() else None,
                ci95=np.quantile(draws,[.025,.975]).tolist() if draws else [None,None],valid_images=int(np.isfinite(values).sum()))


def train_grid(data,epochs=200,seed=42):
    torch.manual_seed(seed);tr,va,te=split_indices(data)
    scaler=StandardScaler().fit(data['features'][tr]);X=torch.tensor(scaler.transform(data['features']),dtype=torch.float32)
    Y=torch.tensor(data['targets'].reshape(len(X),-1),dtype=torch.float32)
    head=nn.Linear(X.shape[1],Y.shape[1]);optimizer=torch.optim.AdamW(head.parameters(),lr=1e-3,weight_decay=1e-4)
    best=float('inf');weights=None;selected=0
    for epoch in range(epochs):
        head.train();optimizer.zero_grad();loss=nn.functional.binary_cross_entropy_with_logits(head(X[tr]),Y[tr]);loss.backward();optimizer.step()
        head.eval()
        with torch.no_grad(): value=float(nn.functional.binary_cross_entropy_with_logits(head(X[va]),Y[va]))
        if value<best: best=value;weights=copy.deepcopy(head.state_dict());selected=epoch+1
    head.load_state_dict(weights)
    with torch.no_grad(): pred=head(X[te]).sigmoid().numpy().reshape(data['targets'][te].shape)
    baseline=np.broadcast_to(data['targets'][tr].mean(0),pred.shape).copy()
    model_metrics=grid_metrics(data['targets'][te],pred);baseline_metrics=grid_metrics(data['targets'][te],baseline)
    summary={}
    for c,name in enumerate(('presence','core')):
        summary[name]={m:grouped_interval([r[c][m] for r in model_metrics],data['groups'][te]) for m in ('ap','auc','iou')}
        summary[name]['mean_target_baseline']={m:grouped_interval([r[c][m] for r in baseline_metrics],data['groups'][te]) for m in ('ap','auc','iou')}
        summary[name]['paired_ap_delta']=grouped_interval([a[c]['ap']-b[c]['ap'] if a[c]['ap'] is not None and b[c]['ap'] is not None else None for a,b in zip(model_metrics,baseline_metrics)],data['groups'][te])
    return dict(summary=summary,selected_epoch=selected,parameter_count=sum(p.numel() for p in head.parameters()),threshold=.5,
                interpretation='linear recoverability of 32x32 grid; not high-resolution segmentation'),dict(predictions=pred,baseline=baseline,targets=data['targets'][te],images=data['images'][te],sample_ids=data['sample_ids'][te],groups=data['groups'][te]),dict(head=weights,scaler_mean=scaler.mean_,scaler_scale=scaler.scale_)


def regress(data):
    tr,va,te=split_indices(data);scaler=StandardScaler().fit(data['features'][tr]);X=scaler.transform(data['features']);Y=data['descriptors']
    ys=StandardScaler().fit(Y[tr]);Z=ys.transform(Y);models=[]
    for alpha in (.01,.1,1.,10.,100.):
        model=Ridge(alpha=alpha).fit(X[tr],Z[tr]);error=np.abs(model.predict(X[va])-Z[va]).mean();models.append((error,alpha,model))
    _,alpha,model=min(models,key=lambda x:x[0]);pred=ys.inverse_transform(model.predict(X[te]));baseline=np.broadcast_to(Y[tr].mean(0),pred.shape)
    return dict(alpha=alpha,mae=np.abs(pred-Y[te]).mean(0).tolist(),baseline_mae=np.abs(baseline-Y[te]).mean(0).tolist(),
        per_descriptor=[grouped_interval(np.abs(pred-Y[te])[:,i],data['groups'][te]) for i in range(3)],
        predictions=pred.tolist(),targets=Y[te].tolist(),sample_ids=data['sample_ids'][te].tolist())


def retrieve(data,seed=42,k=5):
    tr,_,te=split_indices(data);scale=StandardScaler().fit(data['features'][tr]);X=scale.transform(data['features']);X/=np.linalg.norm(X,axis=1,keepdims=True).clip(1e-12)
    rng=np.random.default_rng(seed);rows=[]
    for i in te:
        gallery=[j for j in tr if data['groups'][j]!=data['groups'][i] and data['image_hashes'][j]!=data['image_hashes'][i]]
        if len(gallery)<k: raise ValueError('Too few independent gallery images for Top-5')
        ranked=sorted(gallery,key=lambda j:(-float(X[i]@X[j]),str(data['sample_ids'][j])))[:k]
        random=list(rng.choice(gallery,k,replace=False));record=dict(query=str(data['sample_ids'][i]))
        for name,neighbors in [('cosine',ranked),('random',random)]:
            quality=[]
            for j in neighbors:
                a,b=data['targets'][i,0]>0,data['targets'][j,0]>0;union=(a|b).sum()
                quality.append(dict(id=str(data['sample_ids'][j]),similarity=float(X[i]@X[j]),presence_iou=float((a&b).sum()/union) if union else None,
                    descriptor_mae=np.abs(data['descriptors'][i]-data['descriptors'][j]).tolist()))
            record[name]=quality
        rows.append(record)
    stats={name:grouped_interval([np.mean([n['presence_iou'] for n in r[name] if n['presence_iou'] is not None]) if any(n['presence_iou'] is not None for n in r[name]) else None for r in rows],data['groups'][te]) for name in ('cosine','random')}
    return dict(rows=rows,statistics=stats,top_k=k,seed=seed)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--features',type=Path,required=True);p.add_argument('--kind',choices=['grid','descriptors','retrieval'],required=True);p.add_argument('--epochs',type=int,default=200);p.add_argument('--output',type=Path,required=True);a=p.parse_args(argv)
    if a.epochs<1: p.error('epochs must be positive')
    data=np.load(a.features,allow_pickle=False);a.output.mkdir(parents=True,exist_ok=True)
    provenance=read(a.features.with_suffix('.json'));meta=dict(protocol_id=provenance['protocol_id'],features_sha256=file_hash(a.features),encoder=provenance['encoder'])
    if a.kind=='grid':
        result,arrays,weights=train_grid(data,a.epochs);np.savez_compressed(a.output/'predictions.npz',**arrays);torch.save(weights,a.output/'linear_probe.pt')
    elif a.kind=='descriptors':
        result=regress(data)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(2,3,figsize=(12,7))
        prediction=np.asarray(result['predictions']);truth=np.asarray(result['targets'])
        for i,name in enumerate(('length density','branch pixels / length','endpoints / length')):
            axes[0,i].scatter(truth[:,i],prediction[:,i],s=16);axes[0,i].set_title(name);axes[0,i].set_xlabel('GT');axes[0,i].set_ylabel('predicted')
            lo=min(truth[:,i].min(),prediction[:,i].min());hi=max(truth[:,i].max(),prediction[:,i].max());axes[0,i].plot([lo,hi],[lo,hi],color='gray')
            axes[1,i].hist(prediction[:,i]-truth[:,i],bins=15);axes[1,i].set_xlabel('prediction error')
        fig.tight_layout();fig.savefig(a.output/'descriptors.png',dpi=200);plt.close(fig)
    else: result=retrieve(data)
    write(a.output/'report.json',{**meta,**result})

if __name__=='__main__': main()
