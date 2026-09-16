"""Predeclared sample galleries built from targets or actual saved predictions."""
from pathlib import Path
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from topodise.training.runtime.protocol import read,write,file_hash


def predictions(sources,case_ids,output):
    protocols=[read(p.parent/'report.json')['protocol_id'] for p in sources]
    if len(set(protocols))!=1: raise ValueError('Do not mix STR-A/STR-B in a comparison figure')
    data=[np.load(p,allow_pickle=False) for p in sources]
    lookups=[{str(id):i for i,id in enumerate(d['sample_ids'])} for d in data]
    if not case_ids or len(case_ids)!=len(set(case_ids)): raise ValueError('Predeclare distinct case IDs')
    if any(any(id not in lookup for id in case_ids) for lookup in lookups): raise ValueError('Every model must cover every predeclared case')
    for id in case_ids:
        anchor=data[0]['targets'][lookups[0][id]]
        if any(not np.array_equal(d['targets'][lookup[id]],anchor) for d,lookup in zip(data,lookups)): raise ValueError('Model target/geometry mismatch')
    fig,axes=plt.subplots(len(case_ids)*2,len(data)+2,squeeze=False,figsize=(3*(len(data)+2),2.8*len(case_ids)*2))
    for i,id in enumerate(case_ids):
        for channel,name in enumerate(('presence','core')):
            row=axes[2*i+channel];row[0].imshow(data[0]['images'][lookups[0][id]],cmap='gray');row[0].set_title(id+' / image')
            row[1].imshow(data[0]['targets'][lookups[0][id],channel],vmin=0,vmax=1,cmap='viridis');row[1].set_title('GT '+name)
            for ax,d,lookup,path in zip(row[2:],data,lookups,sources):
                ax.imshow(d['predictions'][lookup[id],channel],vmin=0,vmax=1,cmap='viridis');ax.set_title(path.parent.name+' / predicted')
            for ax in row: ax.axis('off')
    output.parent.mkdir(parents=True,exist_ok=True);fig.tight_layout();fig.savefig(output,dpi=200);plt.close(fig)
    write(output.with_suffix('.json'),dict(case_ids=case_ids,sources={str(p):file_hash(p) for p in sources},kind='actual linear-probe predictions; same [0,1] scale',quantitative_reports=[str(p.parent/'report.json') for p in sources]))


def targets(features,case_ids,output):
    import torch
    from types import SimpleNamespace
    from ..B01_B06_topology_targets.targets import target
    from topodise.topo_targets import compute_topo_target
    data=np.load(features,allow_pickle=False);lookup={str(s):i for i,s in enumerate(data['sample_ids'])};modes=['hgrad_dual','frangi_dual','sobel_dual']
    fig,axes=plt.subplots(2*len(case_ids),5,figsize=(15,5*len(case_ids)),squeeze=False)
    for i,id in enumerate(case_ids):
        j=lookup[id];x=torch.tensor(data['images'][j],dtype=torch.float32)[None,None]/127.5-1
        values=[target(compute_topo_target,SimpleNamespace(extensions={}),x=x,mode=m) for m in modes]
        for c in (0,1):
            arrays=[data['images'][j],data['targets'][j,c],*[v[0,c].numpy() for v in values]]
            for ax,array,name in zip(axes[2*i+c],arrays,['image','GT',*modes]): ax.imshow(array,cmap='gray');ax.set_title(f'{id} {name} channel {c}');ax.axis('off')
    output.parent.mkdir(parents=True,exist_ok=True);fig.tight_layout();fig.savefig(output,dpi=200);plt.close(fig)
    write(output.with_suffix('.json'),dict(kind='supervision targets, NOT predictions',features_sha256=file_hash(features),case_ids=case_ids))


def retrieval_gallery(features,retrieval_report,case_ids,output):
    data=np.load(features,allow_pickle=False);rows=read(retrieval_report)['rows'];lookup={str(s):i for i,s in enumerate(data['sample_ids'])};rows={r['query']:r for r in rows}
    k=len(rows[case_ids[0]]['cosine']);fig,axes=plt.subplots(len(case_ids)*2,k+1,figsize=(3*(k+1),5*len(case_ids)),squeeze=False)
    for i,id in enumerate(case_ids):
        neighbors=[id]+[r['id'] for r in rows[id]['cosine']]
        for col,sid in enumerate(neighbors):
            j=lookup[sid];axes[2*i,col].imshow(data['images'][j],cmap='gray');axes[2*i,col].set_title(sid)
            axes[2*i+1,col].imshow(data['masks'][j],cmap='gray')
            axes[2*i,col].axis('off');axes[2*i+1,col].axis('off')
    output.parent.mkdir(parents=True,exist_ok=True);fig.tight_layout();fig.savefig(output,dpi=200);plt.close(fig)
    write(output.with_suffix('.json'),dict(case_ids=case_ids,retrieval_sha256=file_hash(retrieval_report),features_sha256=file_hash(features)))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--kind',choices=['predictions','targets','retrieval'],default='predictions');p.add_argument('--sources',nargs='+',type=Path,required=True);p.add_argument('--cases',type=Path,required=True,help='JSON with predeclared case_ids');p.add_argument('--retrieval-report',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args(argv);ids=read(a.cases)['case_ids']
    if a.kind=='predictions': predictions(a.sources,ids,a.output)
    elif a.kind=='targets': targets(a.sources[0],ids,a.output)
    else:
        if not a.retrieval_report: p.error('retrieval requires --retrieval-report')
        retrieval_gallery(a.sources[0],a.retrieval_report,ids,a.output)

if __name__=='__main__': main()
