"""Audited DAG execution. Completion is a verified artifact, never a directory."""
from importlib import import_module
from pathlib import Path
import os, time
from topodise.training.runtime.protocol import file_hash, read, write, lock, source_hashes as formal_hashes
from topodise.training.hooks import training_hooks
from .registry import HOME
from .interventions import AblationHooks


def source_hashes():
    values=formal_hashes()
    for p in HOME.rglob('*'):
        if p.suffix in ('.py','.json','.yaml') and not {'results','__pycache__','tests'}.intersection(p.parts):
            values[str(p.relative_to(HOME.parent))]=file_hash(p)
    return values


def audit_job(job):
    """Audit actual inputs after installing the SAME intervention as training."""
    from topodise.training.configs import Stage0Config,Stage1Config,Stage2Config
    from topodise.data import CORN1Dataset,RegionPairDataset
    from topodise.data.corn1_dataset import _derive_label_path
    stage=job['stage'];cfg={'stage0':Stage0Config,'stage1':Stage1Config,'stage2':Stage2Config}[stage](**job['config'])
    files=set();info={}
    if stage=='stage0':
        ds=CORN1Dataset(cfg.corn1_root)
        if cfg.expected_samples is not None and len(ds)!=cfg.expected_samples: raise ValueError('Unexpected CORN-1 image count')
        files.update(ds.paths)
        if cfg.topo_target=='gt':
            masks=[_derive_label_path(p) for p in ds.paths]
            if any(p is None for p in masks): raise ValueError('Missing CORN-1 masks')
            files.update(masks)
        info['samples']=len(ds)
        reference=cfg.extensions.get('steps_reference_manifest')
        if reference:
            regions=RegionPairDataset(manifest_path=reference,base_path=cfg.extensions['reference_data_root'])
            files.add(reference);files.update(row[k] for row in regions.pairs for k in ('central','whorl'))
            info['matched_steps_per_epoch']=len(regions)//cfg.batch_size
    elif stage=='stage1':
        ds=RegionPairDataset(manifest_path=cfg.region_pair_manifest,base_path=cfg.base_path,strict_whitelist=True)
        if any('unlabeled_dataset_en' not in Path(row[k]).parts for row in ds.pairs for k in ('central','whorl')):
            raise ValueError('Stage1 must use independent unlabeled data')
        original=list(ds.pairs)
        ds=AblationHooks().dataset(ds,cfg)
        paired=sum(row['is_paired'] for row in ds.pairs)
        if cfg.lambda_region>0 and paired<cfg.batch_size: raise ValueError('Insufficient genuine region pairs')
        files.add(cfg.region_pair_manifest)
        # Hash full source pool as well as selected groups to define data efficiency.
        files.update(row[k] for row in original for k in ('central','whorl'))
        info.update(samples=len(ds),paired=paired,source_samples=len(original))
    else:
        import json
        from topodise.training.stage2 import _make_dataset,_build_kfold_splits
        from topodise.training.hooks import current_hooks
        # Missing declared images must not silently shrink the cohort.
        for patient in json.loads(Path(cfg.pd_json).read_text()):
            paths=patient.get('image_paths',{})
            groups=paths.values() if isinstance(paths,dict) else [paths]
            files.update(str((Path(cfg.base_path)/p).resolve()) for group in groups for p in group)
        ds=_make_dataset(cfg,None)
        files.add(cfg.pd_json);files.update(p for sample in ds.samples for p in sample[:2])
        folds=[];effective_folds=[]
        for fold,(tr,va) in enumerate(_build_kfold_splits(ds,cfg),1):
            a=sorted(set(str(ds.samples[i][3]) for i in tr));b=sorted(set(str(ds.samples[i][3]) for i in va))
            if set(a)&set(b): raise ValueError('Patient leakage across split')
            folds.append(dict(fold=fold,train=a,validation=b))
            from topodise.training.stage2 import _group_bags
            active_train,active_val=current_hooks().fold_bags(_group_bags(ds,tr),_group_bags(ds,va),cfg,fold)
            effective_folds.append(dict(fold=fold,train_patients=sorted(str(bag[0]) for bag in active_train),validation_patients=sorted(str(bag[0]) for bag in active_val)))
        info.update(samples=ds.samples,folds=folds,effective_folds=effective_folds,input_slots=getattr(ds,'intervention_input_manifest',None))
    if stage!='stage2' and len(ds)<cfg.batch_size: raise ValueError('Dataset cannot provide a full training batch')
    return dict(**info,files={str(Path(p).resolve()):file_hash(p) for p in sorted(files)})


def run_job(job,device='cpu'):
    from dataclasses import asdict
    import torch
    from topodise.training.configs import Stage0Config,Stage1Config,Stage2Config
    stage=job['stage'];values=dict(job['config']);out=Path(values['output_dir']);out.mkdir(parents=True,exist_ok=True)
    parent=values.get('parent_ckpt') or values.get('extensions',{}).get('initial_checkpoint')
    parent_hash=file_hash(parent) if parent else None
    if parent:
        from topodise.training.runtime.protocol import require_complete
        require_complete(Path(parent).parent.parent)
    hooks=AblationHooks()
    with training_hooks(hooks):
        audit=audit_job(job)
        lock(out/'input_audit.json',audit)
        lock(out/'run_contract.json',dict(job=job,parent_sha256=parent_hash,source_sha256=source_hashes()))
        done=out/'DONE.json'
        if done.exists():
            saved=read(done)
            for relative,sha in saved['artifacts'].items():
                if file_hash(out/relative)!=sha: raise RuntimeError(f'Completed artifact changed: {out/relative}')
            return read(out/'result.json')
        values['device']=device
        cfg={'stage0':Stage0Config,'stage1':Stage1Config,'stage2':Stage2Config}[stage](**values)
        if torch.cuda.is_available() and str(device).startswith('cuda'): torch.cuda.reset_peak_memory_stats()
        write(out/'status.json',dict(status='running',started=time.time()))
        start=time.monotonic()
        try:
            result=import_module('topodise.training.'+stage).run_single_experiment(cfg)
            if stage=='stage2':
                from topodise.training.runtime.full_stage2 import prediction_index
                prediction_index(result,values)
            else:
                last=torch.load(out/'train/ckpt/last.pt',map_location='cpu',weights_only=False)
                if int(last['epoch'])!=cfg.epochs: raise RuntimeError('Training did not reach its requested endpoint')
                ckpt=out/'train/ckpt/ema_best.pt'
                write(out/'train/TRAINING_DONE.json',dict(schema_version=2,checkpoint_sha256=file_hash(ckpt),parent_checkpoint_sha256=parent_hash,epochs=cfg.epochs))
            write(out/'result.json',result)
            artifacts={'result.json':file_hash(out/'result.json')}
            checkpoint_dir=out/(f'train_{cfg.task}' if stage=='stage2' else 'train')/'ckpt'
            for path in checkpoint_dir.rglob('*'):
                if path.is_file() and path.suffix in ('.pt','.json'): artifacts[str(path.relative_to(out))]=file_hash(path)
            write(out/'cost.json',dict(wall_seconds=time.monotonic()-start,device=device,torch_version=torch.__version__,
                peak_cuda_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_available() and str(device).startswith('cuda') else None))
            write(done,dict(status='complete',artifacts=artifacts,parent_sha256=parent_hash))
            write(out/'status.json',dict(status='complete'))
            return result
        except Exception as exc:
            write(out/'status.json',dict(status='failed',error=repr(exc)))
            raise


def execute(plan,devices):
    """Topological waves, isolated subprocesses, one process per device slot."""
    from concurrent.futures import ThreadPoolExecutor
    import queue, subprocess, sys
    root=Path(plan['output']);completed=set();remaining={j['id']:j for j in plan['jobs']}
    slots=queue.Queue()
    for d in devices: slots.put(d)
    def worker(job):
        device=slots.get()
        try:
            path=root/'configs'/f'{job["id"]}.json';log=root/'logs'/f'{job["id"]}.log';log.parent.mkdir(exist_ok=True)
            print(f'[run] {job["id"]} on {device}',flush=True)
            with log.open('a') as stream:
                subprocess.run([sys.executable,'-m','ablation_experiments.common.worker','--job',str(path),'--device',device],cwd=HOME.parent,
                    env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'},stdout=stream,stderr=subprocess.STDOUT,check=True)
        finally: slots.put(device)
    while remaining:
        ready=[j for j in remaining.values() if set(j['dependencies'])<=completed]
        if not ready: raise ValueError('Cycle or missing stage dependency')
        with ThreadPoolExecutor(max_workers=len(devices)) as pool: list(pool.map(worker,ready))
        for j in ready: completed.add(j['id']);del remaining[j['id']]
