from pathlib import Path
import argparse,os
from topodise.training.runtime.protocol import read
from topodise.training.runtime.worker import normalize_device,execution_lock
from .execution import run_job,source_hashes


def main():
    p=argparse.ArgumentParser();p.add_argument('--job',type=Path,required=True);p.add_argument('--device',default='cpu');a=p.parse_args()
    device=normalize_device(a.device)
    os.environ.pop('TOPODISE_REPRODUCIBILITY',None);os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    job=read(a.job);protocol=read(a.job.parent.parent/'protocol.json')
    if protocol['source_sha256']!=source_hashes() or job not in protocol['jobs']: raise RuntimeError('Source/job changed since plan lock')
    with execution_lock(a.job.with_suffix('.running.lock')): run_job(job,device)

if __name__=='__main__': main()
