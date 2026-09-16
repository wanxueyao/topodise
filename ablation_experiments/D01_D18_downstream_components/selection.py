"""D17: two selection metrics on one fixed, non-early-stopped trajectory."""
from pathlib import Path
import argparse
from topodise.training.runtime.protocol import read,write
from utils.confidence_intervals import scores


def select_trajectory(directory,classes):
    folds={}
    for path in Path(directory).glob('fold*_epoch*.json'):
        item=read(path);folds.setdefault(item['fold'],[]).append(item)
    if not folds: raise ValueError('No cached trajectory; old best-only results cannot support D17')
    result={metric:[] for metric in ('accuracy','bacc')}
    for fold,epochs in sorted(folds.items()):
        epochs.sort(key=lambda r:r['epoch']);count=epochs[-1]['epoch']
        if [r['epoch'] for r in epochs]!=list(range(1,count+1)): raise ValueError('Incomplete training trajectory')
        anchor={(r['patient_id'],r['y_true']) for r in epochs[0]['predictions']}
        if any({(r['patient_id'],r['y_true']) for r in ep['predictions']}!=anchor for ep in epochs): raise ValueError('Trajectory cohort changed')
        for metric in result:
            def score(ep):
                rows=ep['predictions'];return scores([r['y_true'] for r in rows],[r['y_pred'] for r in rows],[r['patient_probs'] for r in rows],classes)[metric]
            best=max(epochs,key=lambda ep:(score(ep),-ep['epoch']))
            result[metric].append(dict(fold=fold,epoch=best['epoch'],predictions=best['predictions']))
    return dict(reference_id='R-SELECT',protocol_id='SYS-C-fixed-trajectory-v1',results=result,limitation='Same validation cohort selects epoch and reports performance')

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--trajectory',type=Path,required=True);p.add_argument('--classes',type=int,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args(argv);write(a.output,select_trajectory(a.trajectory,a.classes))


if __name__=="__main__": main()
