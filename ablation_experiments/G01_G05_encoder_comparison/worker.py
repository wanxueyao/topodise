"""Full-system encoder comparison worker with explicit factory injection."""
from pathlib import Path
import argparse,os
from topodise.training.runtime.protocol import read
from topodise.training.runtime.worker import normalize_device,execution_lock,run_pretrain,run_stage2
from topodise.training.hooks import training_hooks
from ..common.interventions import AblationHooks
from ..common.execution import source_hashes

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--job',type=Path,required=True);p.add_argument('--device',default='cpu');a=p.parse_args();device=normalize_device(a.device)
    job=read(a.job);protocol=read(a.job.parent.parent/'protocol.json')
    if job not in protocol['jobs'][job['stage']] or protocol['source_sha256']!=source_hashes(): raise RuntimeError('Encoder comparison source/job changed')
    os.environ.pop('TOPODISE_REPRODUCIBILITY',None);os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    with execution_lock(a.job.with_suffix('.running.lock')),training_hooks(AblationHooks()):
        (run_stage2 if job['stage']=='stage2' else run_pretrain)(job,device)
