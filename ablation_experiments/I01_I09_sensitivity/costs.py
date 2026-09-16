"""Summarize measured costs; absent GPU/inference measurements remain null."""
from pathlib import Path
import argparse
from topodise.training.runtime.protocol import read,write


def collect(root):
    rows=[]
    for path in Path(root).glob('runs/*/cost.json'):
        record=read(path);directory=path.parent
        record.update(run_id=directory.name,status=read(directory/'status.json')['status'],fold_metadata=[read(p) for p in directory.glob('train*/ckpt/fold*_metadata.json')])
        record['inference_timings']=[read(p) for p in directory.glob('train*/ckpt/*_timing.json')]
        rows.append(record)
    return dict(runs=rows,totals=dict(wall_seconds=sum(r['wall_seconds'] for r in rows)),note='Measured process wall times; sum is compute occupancy, not concurrent batch makespan; null means unmeasured')

def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args(argv);write(a.output,collect(a.root))


if __name__=="__main__": main()
