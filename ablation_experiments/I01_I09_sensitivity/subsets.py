"""Nested group subsets; no image-wise sampling masquerading as patient budgets."""
from collections import defaultdict
import hashlib, math


def ranked(values, seed):
    return sorted(values,key=lambda s:hashlib.sha256(f'{seed}:{s}'.encode()).hexdigest())


def labeled(bags, fraction, seed):
    classes=defaultdict(list)
    for b in bags: classes[b[2]].append(b)
    chosen=[]
    for rows in classes.values():
        ids=ranked([b[0] for b in rows],seed);n=math.floor(len(ids)*fraction)
        if n<2: raise ValueError('Label fraction leaves fewer than two training patients in a class')
        selected=set(ids[:n]);chosen.extend(b for b in rows if b[0] in selected)
    return chosen


def unlabeled(ds,fraction,seed):
    # Stratify by whether a source has at least one genuine pair.
    groups=defaultdict(list)
    for row in ds.pairs:
        sid=str(row.get('subject_id',''))
        if not sid or sid=='?': raise ValueError('I07 requires explicit subject/source identifiers')
        groups[sid].append(row)
    strata=defaultdict(list)
    for sid,rows in groups.items(): strata[any(r['is_paired'] for r in rows)].append(sid)
    selected=set()
    for ids in strata.values():
        n=math.floor(len(ids)*fraction)
        if n<1: raise ValueError('Fraction removes an entire pairing stratum')
        selected.update(ranked(ids,seed)[:n])
    ds.pairs=[r for sid,rows in groups.items() if sid in selected for r in rows]
    return ds
