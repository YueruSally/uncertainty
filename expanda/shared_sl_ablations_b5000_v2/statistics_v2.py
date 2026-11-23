"""Paired seed bootstrap conditional on the 18 fixed development instances."""
from collections import defaultdict
from statistics import mean, stdev
import numpy as np

BUDGETS = (400, 1000, 5000)
VOLUME = 1.331
TOLERANCE = 1e-12/VOLUME
DRAWS = 50000
BOOTSTRAP_SEED = 20261123


class Panel:
    def __init__(self, entries, seeds):
        self.instances = sorted(e['instance_id'] for e in entries)
        self.seeds = sorted(seeds)
        cfg = {e['instance_id']:e['configuration'] for e in entries}
        self.configurations = sorted(set(cfg.values()), key=lambda c:int(c[1:]))
        if len(self.instances) != 18 or len(self.seeds) != 3 or len(self.configurations) != 9:
            raise ValueError('Expected 18 instances / 3 seeds / 9 configurations')
        self.config_ids = {c:[i for i,n in enumerate(self.instances) if cfg[n] == c] for c in self.configurations}
        if any(len(ids) != 2 for ids in self.config_ids.values()):
            raise ValueError('Expected two instances per configuration')
        self.cfg = cfg
        self.draws = np.random.default_rng(BOOTSTRAP_SEED).integers(
            0, len(self.seeds), size=(DRAWS,len(self.instances),len(self.seeds)), dtype=np.uint8)

    def aggregate(self, x):
        per_instance = x.mean(axis=-1)
        return np.mean([per_instance[...,ids].mean(axis=-1) for ids in self.config_ids.values()], axis=0)

    def interval(self, x):
        sampled = x[np.arange(len(self.instances))[None,:,None],self.draws]
        return [float(v) for v in np.percentile(self.aggregate(sampled), [2.5,97.5])]

    def values(self, rows, budget):
        by = {(r['instance'],r['seed']):r for r in rows}
        if len(by) != len(rows) or set(by) != {(n,s) for n in self.instances for s in self.seeds}:
            raise ValueError('Unmatched panel')
        return np.array([[next(s['hv'] for s in by[n,seed]['budget_fronts'] if s['evaluation']==budget)/VOLUME
                          for seed in self.seeds] for n in self.instances])

    def compare(self, own, reference, budget, method, ref, names):
        # Caller validates metadata/scenarios before normalization.
        a, b = self.values(own,budget), self.values(reference,budget)
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError('Nonfinite HV')
        if (b <= 0).any():
            raise ValueError('Cannot define percentage gain against zero reference HV')
        absolute = a-b; percent = 100*absolute/b
        low,high = self.interval(percent); alo,ahi = self.interval(absolute)
        endpoint = [r['optimisation_seconds'] for r in own]
        if any(t is None or not np.isfinite(t) or t < 0 for t in endpoint):
            raise ValueError('Missing original runtime')
        times = [next(s for s in r['budget_fronts'] if s['evaluation']==budget).get('controller_seconds') for r in own]
        wtl = [int((absolute>TOLERANCE).sum()), int((np.abs(absolute)<=TOLERANCE).sum()),
               int((absolute< -TOLERANCE).sum())] if method != ref else ['—']*3
        result = dict(method=method,reference_method=ref,评价次数=budget,方法=names[method],参照=names[ref],配对数=54,
                      平均HV=float(self.aggregate(a)),参照平均HV=float(self.aggregate(b)),
                      平均配对提升百分比=float(self.aggregate(percent)),提升95区间下限=low,提升95区间上限=high,
                      平均HV差=float(self.aggregate(absolute)),HV差95区间下限=alo,HV差95区间上限=ahi,
                      胜=wtl[0],平=wtl[1],负=wtl[2],
                      节点累计平均秒=mean(times) if all(t is not None for t in times) else None,
                      节点累计标准差秒=stdev(times) if all(t is not None for t in times) else None,
                      B5000搜索平均秒=mean(endpoint) if budget==5000 else None,
                      B5000搜索标准差秒=stdev(endpoint) if budget==5000 else None)
        details = [dict(method=method,reference_method=ref,evaluation=budget,instance=n,configuration=self.cfg[n],seed=s,
                        own=float(a[i,j]),reference=float(b[i,j]),absolute=float(absolute[i,j]),percent=float(percent[i,j]))
                   for i,n in enumerate(self.instances) for j,s in enumerate(self.seeds)]
        return result, details


def wide(rows, names):
    by = defaultdict(dict)
    for r in rows: by[r['method']][r['评价次数']] = r
    output = []
    for method in names:
        if method not in by: continue
        item = dict(method=method,方法=names[method],参照='Base',配对数=54)
        for budget in BUDGETS:
            r = by[method][budget]
            for key in ('平均HV','平均配对提升百分比','提升95区间下限','提升95区间上限'):
                item[f'{budget}_{key}'] = r[key]
        r = by[method][5000]
        for key in ('胜','平','负','B5000搜索平均秒','B5000搜索标准差秒'): item[key] = r[key]
        output.append(item)
    return output
