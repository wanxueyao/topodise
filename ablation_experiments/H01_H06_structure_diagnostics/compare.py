"""Paired structure AP differences for H05 and main-figure method comparisons."""
from pathlib import Path
import argparse
import numpy as np
from .probes import grid_metrics,grouped_interval
from topodise.training.runtime.protocol import read,write,file_hash


def compare(reference,variants):
    ref=np.load(reference,allow_pickle=False);meta=read(reference.parent/'report.json');out=[]
    before=grid_metrics(ref['targets'],ref['predictions'])
    for path in variants:
        data=np.load(path,allow_pickle=False);other=read(path.parent/'report.json')
        if meta['protocol_id']!=other['protocol_id']: raise ValueError('Different structure protocols')
        for key in ('sample_ids','groups','targets'):
            if not np.array_equal(data[key],ref[key]): raise ValueError('Unaligned structure comparison: '+key)
        after=grid_metrics(data['targets'],data['predictions']);metrics={}
        for c,name in enumerate(('presence','core')):
            values=[a[c]['ap']-b[c]['ap'] if a[c]['ap'] is not None and b[c]['ap'] is not None else None for a,b in zip(after,before)]
            metrics[name]=grouped_interval(values,ref['groups'])
        out.append(dict(source=str(path),source_sha256=file_hash(path),paired_AP_delta=metrics))
    return dict(protocol_id=meta['protocol_id'],reference=str(reference),reference_sha256=file_hash(reference),comparisons=out)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--reference',type=Path,required=True);p.add_argument('--variants',nargs='+',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args(argv);write(a.output,compare(a.reference,a.variants))

if __name__=='__main__': main()
