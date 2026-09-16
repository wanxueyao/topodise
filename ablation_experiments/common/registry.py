"""Strict, dependency-free experiment registry, backed by numbered group files."""
from copy import deepcopy
from dataclasses import fields
import json
from pathlib import Path
from topodise.training.configs import Stage0Config, Stage1Config, Stage2Config

HOME = Path(__file__).resolve().parents[1]
TASKS = ('pd_binary', 'pd_stage', 'pd_subtype_tdrest', 'dm_binary')
ANCHORS = dict(zip(TASKS, ('PAPER_BINARY_US0_stage2_on_stage1',
    'USTAGELOGITS_U1_stage_lse_gentle_global_15fold',
    'TDREST15LOGITS_TDREST5_minor_fdiff_focal_g15_balanced_15fold', 'DMMIL3F1_weak_strict_fdiff')))
REFERENCES = dict(zip(TASKS, ('R-BIN','R-STG','R-SUB','R-DM')))
FINAL_REFERENCES = {**REFERENCES, 'pd_stage':'R-FINAL-STG', 'pd_subtype_tdrest':'R-FINAL-SUB'}


def registry():
    result = {}
    for path in sorted(HOME.glob('[A-I][0-9]*/*[0-9].json')):
        spec = json.loads(path.read_text())
        id = spec['experiment_id']
        if id in result:
            raise ValueError(f'Duplicate experiment: {id}')
        for v in spec['variants']:
            for n, cls in enumerate((Stage0Config,Stage1Config,Stage2Config)):
                unknown = set(v[f'stage{n}']) - {f.name for f in fields(cls)}
                if unknown:
                    raise ValueError(f'{v["variant_id"]}: unknown Stage{n} fields {unknown}')
        result[id] = {**spec, 'directory':str(path.parent.relative_to(HOME))}
    return result


def select(names, include_supplementary=False):
    all_specs = registry()
    selected = []
    for name in names:
        matches = [s for id,s in all_specs.items() if name in ('all',id,id[0]) or name.startswith(id+'.')]
        if name == 'M0':
            selected.append(dict(experiment_id='M0', title='GT/HGrad-dual Full',priority='core',
                                 variant_id='M0.default',kind='train',stage0={},stage1={},stage2={}))
            continue
        if not matches:
            raise ValueError(f'Unknown experiment {name}')
        for spec in matches:
            for v in spec['variants']:
                if '.' in name and v['variant_id']!=name: continue
                explicit = name == spec['experiment_id'] or name == v['variant_id']
                if not explicit and (v['priority']=='pending' or (v['priority']=='supplementary' and not include_supplementary)): continue
                selected.append({k:deepcopy(val) for k,val in {**spec,**v}.items() if k!='variants'})
    return list({s['variant_id']:s for s in selected}.values())


def applicable(spec, cfg):
    task=cfg['task']
    if spec.get('tasks') and task not in spec['tasks']: return False
    if spec['experiment_id'] in ('C03','C04','C05') and task=='dm_binary': return False
    requirements = {
        'fusion':cfg['finetune_fusion'], 'stats':cfg['use_morphology_features'],
        'prototype':cfg['use_prototype'], 'balanced':cfg['mil_balanced_bags'],
        'class_weights':cfg['use_class_weights'], 'focal':cfg['use_focal'],
        'focal_and_weights':cfg['use_focal'] and cfg['use_class_weights'],
        'fdiff':cfg['use_feature_diffusion'], 'class_condition':cfg['use_feature_diffusion'] and cfg['fdiff_class_cond'],
        'balanced_selection':cfg['select_metric'] in ('balanced_patient','balanced_accuracy','balanced_acc'),
        'generated_support':cfg['use_prototype'] and cfg['use_feature_diffusion'] and
            bool(cfg['proto_fdiff_support_aug'] or cfg['proto_fdiff_support_aug_by_class']),
    }
    requirement=spec.get('requires')
    if requirement is not None and requirement not in requirements: raise ValueError('Unknown applicability requirement: '+requirement)
    return True if requirement is None else requirements[requirement]


def merge(base, changes):
    out=deepcopy(base)
    for key,value in changes.items():
        if key=='extensions': out[key]={**out.get(key,{}),**deepcopy(value)}
        else: out[key]=deepcopy(value)
    return out
