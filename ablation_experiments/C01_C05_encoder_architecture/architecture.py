"""Checkpoint-driven architecture interventions; output contract is unchanged."""
from dataclasses import replace
from topodise.encoder import TopoDiSEEncoder


def build(config, metadata):
    options=metadata.get('extensions',{}).get('architecture',{})
    config=replace(config,fusion=options.get('fusion',config.fusion))
    encoder=TopoDiSEEncoder(config)
    if options.get('shared_branches'):
        encoder.branch_w=encoder.branch_c
    return encoder


def dataset(ds,cfg):
    from collections import Counter
    from copy import copy
    options=cfg.extensions
    if options.get('paired_cohort'):
        ds.samples=[s for s in ds.samples if s[0]!=s[1]]
        if not ds.samples: raise ValueError('No real paired patients for C03/C04')
    if options.get('matched_pair_budget'):
        from topodise.data import DownstreamDataset
        reference=DownstreamDataset(cfg.pd_json,cfg.base_path,task=cfg.task,pair_mode='zip',transform=None,
                                    max_pairs_per_patient=cfg.max_pairs_per_patient,seed=cfg.seed)
        budgets=Counter(s[3] for s in reference.samples)
        actual=Counter(s[3] for s in ds.samples)
        if set(actual)!=set(budgets): raise ValueError('zip/cartesian cohorts differ')
        seen=Counter();selected=[]
        for sample in ds.samples:
            pid=sample[3]
            if seen[pid]<min(budgets[pid],actual[pid]): selected.append(sample);seen[pid]+=1
        ds.samples=selected
    ds.intervention_input_manifest=[dict(patient_id=s[3],central=s[0],whorl=s[1]) for s in ds.samples]
    region=options.get('region')
    if region:
        idx={'central':0,'whorl':1}[region]
        ds.samples=[(s[idx],s[idx],*s[2:]) for s in ds.samples]
    # Refresh the loader's cached aggregate fields after a cohort intervention.
    if hasattr(ds,'_final_cache'): ds._final_cache=None
    return ds
