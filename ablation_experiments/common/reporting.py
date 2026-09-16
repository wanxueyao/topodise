"""Collect hash-verified complete runs into candidate/final experiment results."""
from pathlib import Path
from collections import defaultdict
import statistics
from topodise.training.runtime.protocol import read,write,file_hash,digest
from topodise.training.runtime.full_stage2 import prediction_index
from experiments.topodise_experiments.main import fixed_patient_ensemble
from utils.confidence_intervals import report
from .statistics import paired


def collect(root,bootstrap_samples=2000):
    root=Path(root);plan=read(root/'protocol.json');jobs={j['id']:j for j in plan['jobs']};groups=defaultdict(list);pending=[]
    for view in plan['views']:
        if view['status']=='not_applicable': continue
        indices=[];sources=[];audits=[]
        for member in view['members']:
            job=jobs[member['job_id']];directory=Path(job['config']['output_dir'])
            if not (directory/'DONE.json').exists(): pending.append(dict(view=view,missing=member['job_id']));break
            for path,sha in read(directory/'DONE.json')['artifacts'].items():
                if file_hash(directory/path)!=sha: raise RuntimeError('Changed completed artifact: '+str(directory/path))
            result=read(directory/'result.json')
            if view.get('selection_metric'):
                from ..D01_D18_downstream_components.selection import select_trajectory
                trajectory=select_trajectory(directory/f'train_{view["task"]}/ckpt/trajectory',3 if view['task']=='pd_stage' else 2)
                selected=trajectory['results'][view['selection_metric']]
                result={'fold_results':[{'fold':f['fold'],'best_patient_predictions':f['predictions'],'n_val':len(f['predictions'])} for f in selected]}
            indices.append(prediction_index(result,job['config']));sources.append(dict(run_id=job['id'],result_sha256=file_hash(directory/'result.json')));audits.append(read(directory/'input_audit.json'))
        else:
            if any(set(x)!=set(indices[0]) for x in indices): raise ValueError('Final candidate alignment failure')
            if any(x[k]['y_true']!=indices[0][k]['y_true'] for x in indices for k in x): raise ValueError('Final candidate label mismatch')
            if view['report_method']=='single_model': rows=[dict(p,fold=k[0]) for k,p in sorted(indices[0].items())]
            else: rows=fixed_patient_ensemble(indices,[m['candidate_id'] for m in view['members']],view['report_method'])
            for row in rows: row['pretrain_seed']=view['pretrain_seed']
            groups[(view['variant_id'],view['task'],view['evaluation_unit'])].append(dict(view=view,rows=rows,sources=sources,
                split_hash=digest(audits[0]['folds']),cohort_hash=digest(sorted({(r['patient_id'],r['y_true']) for r in rows}))))
    records=[]
    for key,items in groups.items():
        if sorted(i['view']['pretrain_seed'] for i in items)!=sorted(plan['seeds']): pending.append(dict(group=key,reason='incomplete pretraining seeds'));continue
        rows=[];per_seed=[];classes=3 if key[1]=='pd_stage' else 2
        for slot,item in enumerate(sorted(items,key=lambda i:i['view']['pretrain_seed'])):
            per_seed.append(report(item['rows'],classes,5,bootstrap_samples))
            rows.extend(dict(r,repeat=slot*plan['repeats']+(int(r['fold'])-1)//5) for r in item['rows'])
        stats=report(rows,classes,5,bootstrap_samples)
        stats['pretrain_seeds']=plan['seeds'];stats['n_repeats_per_seed']=plan['repeats']
        record=dict(variant_id=key[0],task=key[1],evaluation_unit=key[2],evidence_track='SYS',protocol_id='SYS-C',
            reference_id=items[0]['view']['reference_id'],reference_variant_id=items[0]['view'].get('reference_variant_id','M0.default'),
            status='complete',statistics=stats,predictions=rows,sources=[s for item in items for s in item['sources']],
            cohort_hash=items[0]['cohort_hash'],split_hash=items[0]['split_hash'],seed_set=plan['seeds'],
            per_pretrain_seed=per_seed,pretrain_seed_bacc_sd=statistics.stdev(s['metrics']['bacc']['estimate'] for s in per_seed) if len(per_seed)>1 else None)
        records.append(record)
    index={(r['variant_id'],r['task'],r['evaluation_unit']):r for r in records}
    for r in records:
        ref=index.get((r['reference_variant_id'],r['task'],r['evaluation_unit']))
        if ref and ref is not r:
            for field in ('cohort_hash','split_hash','seed_set','protocol_id'):
                if ref[field]!=r[field]: raise ValueError(f'Unmatched {field}: {r["variant_id"]}')
            r['paired_difference']=paired(ref['predictions'],r['predictions'],3 if r['task']=='pd_stage' else 2,bootstrap_samples)
    result=dict(schema_version=2,records=records,pending=pending,analyses=plan['analyses'],complete=not pending and not plan['analyses'])
    write(root/'analysis/results.json',result);return result

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();collect(a.root)
