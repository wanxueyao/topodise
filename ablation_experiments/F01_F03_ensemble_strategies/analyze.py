"""Fixed candidate strategies on the same aligned OOF predictions; no oracle."""
from pathlib import Path
import argparse,math
from topodise.training.runtime.protocol import read,write
from topodise.training.runtime.full_stage2 import prediction_index
from experiments.topodise_experiments.main import fixed_patient_ensemble
from utils.confidence_intervals import report


def compare(candidates,task,bootstrap_samples=2000):
    expected=3 if task=='pd_stage' else 5 if task=='pd_subtype_tdrest' else None
    if expected is None or len(candidates)!=expected: raise ValueError('All 3 stage or 5 subtype candidates are required')
    names=[c['name'] for c in candidates]
    from topodise.training.runtime.full_stage2 import paper_recipes
    expected_names={leaf['_group'] for recipe in paper_recipes() if recipe['stage2_tasks'][0]==task for leaf in recipe.get('oracle_experiments', [recipe])}
    if set(names)!=expected_names: raise ValueError('Candidate membership differs from the predeclared formal system')
    if len(set(names))!=len(names): raise ValueError('Duplicate candidate identity')
    indices=[prediction_index(c['result'],c['config']) for c in candidates]
    if any(set(x)!=set(indices[0]) for x in indices): raise ValueError('Unaligned candidate folds/patients')
    if any(x[k]['y_true']!=indices[0][k]['y_true'] for x in indices for k in x): raise ValueError('Unaligned labels')
    classes=3 if task=='pd_stage' else 2
    strategies={name:[dict(p,fold=k[0]) for k,p in sorted(index.items())] for name,index in zip(names,indices)}
    mean=[];diagnostics=[]
    for key in sorted(indices[0]):
        values=[x[key] for x in indices]
        if any(len(v['patient_probs'])!=classes for v in values): raise ValueError('Wrong class probability order/width')
        probs=[sum(v['patient_probs'][c] for v in values)/len(values) for c in range(classes)]
        pred=max(range(classes),key=lambda c:(probs[c],-c))
        mean.append(dict(fold=key[0],patient_id=key[1],y_true=values[0]['y_true'],y_pred=pred,patient_probs=probs))
        diagnostics.append(dict(fold=key[0],patient_id=key[1],label=int(values[0]['y_true']),candidate_predictions=[v['y_pred'] for v in values],
            disagreement=len(set(v['y_pred'] for v in values))>1,mean_confidence=max(probs),mean_error=pred!=values[0]['y_true']))
    strategies['average_probability']=mean
    method='min_entropy_routing' if task=='pd_stage' else 'majority_vote_tie_avg_prob'
    strategies[method]=fixed_patient_ensemble(indices,names,method)
    return dict(task=task,candidate_names=names,strategies={name:dict(predictions=rows,statistics=report(rows,classes,candidates[0]['config']['num_folds'],bootstrap_samples)) for name,rows in strategies.items()},diagnostics=diagnostics)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bundle',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args(argv)
    bundle=read(a.bundle)
    for c in bundle['candidates']:
        if isinstance(c['result'],str): c['result']=read(c['result'])
    write(a.output,compare(bundle['candidates'],bundle['task']))

if __name__=='__main__': main()
