"""Content-addressed stage DAG and explicit candidate/reference memberships."""
from dataclasses import asdict
from pathlib import Path
from .registry import ANCHORS, REFERENCES, FINAL_REFERENCES, TASKS, applicable, merge
from topodise.training.configs import Stage0Config, Stage1Config
from topodise.training.runtime.batch_experiments import STAGE0_SETTINGS, STAGE1_SETTINGS
from topodise.training.runtime.full_stage2 import build_full_config
from topodise.training.runtime.protocol import digest


def build_plan(specs, output, data, seeds=(42,), tasks=TASKS, repeats=3, unit='CANDIDATE'):
    output,data=Path(output).resolve(),Path(data).resolve()
    if not seeds or len(set(seeds))!=len(seeds) or repeats<1: raise ValueError('Unique seeds and positive repeats required')
    jobs, views, analyses = {}, [], []
    def stage_job(stage, values, parents=()):
        # Device/physical output location do not define an experiment identity.
        identity={k:v for k,v in values.items() if k not in ('output_dir','run_name','device','parent_ckpt')}
        key=f'{stage}_{digest(dict(config=identity,parents=list(parents)))[:20]}'
        cfg={**values,'output_dir':str(output/'runs'/key),'run_name':'train'}
        jobs.setdefault(key,dict(id=key,stage=stage,config=cfg,dependencies=list(parents)))
        return key,output/'runs'/key/'train/ckpt/ema_best.pt'
    def parents(spec,seed):
        c0=asdict(Stage0Config(**STAGE0_SETTINGS));c0.update(seed=seed,corn1_root=str(data/'CORN-1'))
        j0,p0=stage_job('stage0',merge(c0,spec.get('stage0',{})))
        if spec.get('skip_stage1'): return j0,p0
        if spec.get('kind')=='source_continue':
            values=merge(c0,{'epochs':STAGE1_SETTINGS['epochs'],'lr':Stage1Config().lr,
                'extensions':{'initial_checkpoint':str(p0),'steps_reference_manifest':str(data/'unlabeled_dataset_en/unlabeled_dataset_en.json'),'reference_data_root':str(data)}})
            return stage_job('stage0',values,[j0])
        c1=asdict(Stage1Config(**STAGE1_SETTINGS));c1.update(seed=seed,parent_ckpt=str(p0),base_path=str(data),pd_json=str(data/'pd_dataset_en/patients_dataset_en.json'),region_pair_manifest=str(data/'unlabeled_dataset_en/unlabeled_dataset_en.json'))
        return stage_job('stage1',merge(c1,spec.get('stage1',{})),[j0])
    def downstream(spec,seed,reference=False,applicability_spec=None):
        applicability_spec = spec if applicability_spec is None else applicability_spec
        parent_id,parent_path=parents(spec,seed)
        bundle=build_full_config(parent_path,output/'unused',data/'pd_dataset_en/patients_dataset_en.json',data,repeats=repeats,tasks=tasks)
        for recipe in bundle['recipes']:
            task=recipe['task']; candidates=recipe['candidates']
            if unit=='CANDIDATE': candidates=[c for c in candidates if c['name']==ANCHORS[task]]
            if not candidates: raise ValueError(f'Missing formal anchor for {task}')
            members=[]
            for candidate in candidates:
                base=candidate['config'];active=applicable(applicability_spec,base)
                # FINAL retains inactive members, including duplicate-equivalent votes.
                if not active and unit=='CANDIDATE':
                    views.append(dict(variant_id=spec['variant_id'],task=task,pretrain_seed=seed,status='not_applicable',candidate_id=candidate['name'],evaluation_unit=unit));continue
                cfg=merge(base,spec.get('stage2',{}) if active else {})
                key,_=stage_job('stage2',cfg,[parent_id]);members.append(dict(candidate_id=candidate['name'],job_id=key,intervened=active))
            if members:
                views.append(dict(experiment_id=spec['experiment_id'],variant_id=spec['variant_id'],task=task,
                    evaluation_unit=unit,evidence_track='SYS',protocol_id='SYS-C',pretrain_seed=seed,
                    reference_id=spec.get('reference_id',(FINAL_REFERENCES if unit=='FINAL' else REFERENCES)[task])+('.'+spec['reference_suffix'] if spec.get('reference_suffix') else ''),
                    status='planned',members=members,
                    reference_variant_id=spec.get('reference_variant_id','M0.'+spec['reference_suffix'] if spec.get('reference_suffix') else 'M0.default'),
                    report_method=recipe['report_method'] if unit=='FINAL' else 'single_model',
                    selection_metric=spec.get('selection_metric')))
    for spec in specs:
        kind=spec['kind']
        if kind not in ('train','selection_trajectory','pretrain_probe','source_continue'):
            analyses.append(dict(spec=spec,kind=kind,status='requires_input',entry=analysis_entry(kind),evaluation_unit='FINAL' if kind in ('ensemble','encoder_comparison') else 'STRUCTURE_PROBE' if spec['experiment_id'].startswith('H') else 'LINEAR_PROBE' if 'probe' in kind else 'CANDIDATE'))
            continue
        for seed in seeds:
            if kind=='pretrain_probe':
                pid,path=parents(spec,seed)
                analyses.append(dict(spec=spec,kind=kind,parent_job=pid,checkpoint=str(path),pretrain_seed=seed,status='requires_probe_data'))
                continue
            reference=dict(experiment_id='M0',variant_id='M0.default',stage0={},stage1={},stage2={},kind='train')
            if kind=='selection_trajectory':
                reference.update(variant_id='D17.reference',stage2=spec['stage2'],reference_id='R-SELECT',reference_variant_id='D17.reference',selection_metric='bacc')
                spec={**spec,'reference_variant_id':'D17.reference','selection_metric':'accuracy'}
            if spec['experiment_id']=='A07':
                reference.update(experiment_id='A07',variant_id='A07.11',stage0={'lambda_topo':.1},stage1={'lambda_topo':.05,'extensions':{'reset_topology_head':True,'train_topology_head':True}},reference_id='R-RESET11',reference_variant_id='A07.11')
                spec={**spec,'reference_variant_id':'A07.11'}
            if spec.get('reference_stage2'):
                reference.update(variant_id='M0.'+spec['reference_suffix'],stage2=spec['reference_stage2'],reference_suffix=spec['reference_suffix'])
                # Matched cohorts inherit the originating experiment's domain;
                # the shared M0.default reference must retain its own applicability.
                downstream(reference,seed,True,applicability_spec=spec)
            else:
                downstream(reference,seed,True)
            downstream(spec,seed)
    unique_views={}
    for view in views:
        key=(view['variant_id'],view['task'],view.get('pretrain_seed'),view['evaluation_unit'])
        if key in unique_views:
            previous=unique_views[key]
            if previous != view:
                raise ValueError(f'Conflicting scientific view: {key}')
        else: unique_views[key]=view
    views=list(unique_views.values())
    return dict(schema_version='ablation-dag-v2',guide_version='1.2',protocol_id='SYS-C',evaluation_unit=unit,
                seeds=list(seeds),tasks=list(tasks),repeats=repeats,jobs=list(jobs.values()),views=views,analyses=analyses)


def execution_slice(plan, names=None):
    """Choose registered views plus their complete parent closure, without changing the protocol."""
    if not names:
        return plan
    variants = {v['variant_id'] for v in plan['views']} | {a['spec']['variant_id'] for a in plan['analyses']}
    selected = set()
    for name in names:
        matches = {v for v in variants if name in ('all', v, v.split('.')[0], v[0])}
        if not matches:
            raise ValueError(f'Execution selection not present in the master plan: {name}')
        selected.update(matches)
    views = [v for v in plan['views'] if v['variant_id'] in selected]
    # A matched reference may differ from M0 (A07, C03-C05, D17).
    references = {v.get('reference_variant_id') for v in views}
    views += [v for v in plan['views'] if v['variant_id'] in references and v['variant_id'] not in selected]
    analyses = [a for a in plan['analyses'] if a['spec']['variant_id'] in selected]
    needed = {m['job_id'] for v in views for m in v.get('members', [])}
    needed.update(a['parent_job'] for a in analyses if 'parent_job' in a)
    jobs = {j['id']: j for j in plan['jobs']}
    pending = list(needed)
    while pending:
        for parent in jobs[pending.pop()]['dependencies']:
            if parent not in needed:
                needed.add(parent)
                pending.append(parent)
    return {**plan, 'jobs': [j for j in plan['jobs'] if j['id'] in needed], 'views': views, 'analyses': analyses}


def analysis_entry(kind):
    entries={
        'replay':'ablation_experiments.D01_D18_downstream_components.replay',
        'nested_finetune':'ablation_experiments.D01_D18_downstream_components.tuning',
        'ensemble':'ablation_experiments.F01_F03_ensemble_strategies.analyze',
        'encoder_comparison':'ablation_experiments.G01_G05_encoder_comparison.run',
        'linear_probe':'ablation_experiments.G01_G05_encoder_comparison.linear_probe',
        'random_probe':'ablation_experiments.G01_G05_encoder_comparison.extract',
        'mlp_probe':'ablation_experiments.G01_G05_encoder_comparison.linear_probe',
        'pca_probe':'ablation_experiments.G01_G05_encoder_comparison.linear_probe',
        'target_gallery':'ablation_experiments.H01_H06_structure_diagnostics.visualize',
        'structure_probe':'ablation_experiments.H01_H06_structure_diagnostics.probes',
        'descriptor_probe':'ablation_experiments.H01_H06_structure_diagnostics.probes',
        'retrieval':'ablation_experiments.H01_H06_structure_diagnostics.probes',
        'stage_probe':'ablation_experiments.H01_H06_structure_diagnostics.compare',
        'region_probe':'ablation_experiments.H01_H06_structure_diagnostics.region_probe',
        'cost_report':'ablation_experiments.I01_I09_sensitivity.costs',
    }
    return entries[kind]
