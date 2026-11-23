#!/usr/bin/env python3
"""Describe B5000 ICM wins/losses against Base in the completed unified Table 5.

Read saved paired HV only; no searches, objective evaluations or HV recomputation.
"""
import argparse
from collections import defaultdict
import csv
import hashlib
import io
import json
import math
from pathlib import Path
from statistics import fmean, median
import sys

sys.dont_write_bytecode = True
VERSION = 'table5-icm-outcomes-v1'
DEFAULT = Path(__file__).resolve().parent / 'transformer_ppo_runs/diagnostics/ccp_shared_sl_ablations_b5000_v2'


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def close(a, b, label):
    if not math.isclose(a, b, rel_tol=1e-11, abs_tol=1e-10):
        raise ValueError(f'{label}: {a} != {b}')


def summarize(rows, tolerance):
    wins = [r['gain'] for r in rows if r['difference'] > tolerance]
    losses = [r['gain'] for r in rows if r['difference'] < -tolerance]
    ties = [r['gain'] for r in rows if abs(r['difference']) <= tolerance]
    gains = [r['gain'] for r in rows]
    if not gains or len(wins) + len(losses) + len(ties) != len(gains):
        raise ValueError('Invalid paired outcomes')
    win_mean = fmean(wins) if wins else None
    loss_drop = -fmean(losses) if losses else None
    tie_mean = fmean(ties) if ties else None
    reconstructed = (len(wins)*(win_mean or 0) - len(losses)*(loss_drop or 0)
                     + len(ties)*(tie_mean or 0)) / len(gains)
    close(reconstructed, fmean(gains), 'Win/loss decomposition')
    return dict(配对数=len(gains),胜=len(wins),平=len(ties),负=len(losses),
                获胜运行平均提升百分比=win_mean,失败运行平均下降百分比=loss_drop,
                全部运行中位数提升百分比=median(gains),全部运行平均提升百分比=fmean(gains),
                持平运行平均提升百分比=tie_mean,加权重构平均提升百分比=reconstructed)


def main(root, output=None):
    root = root.resolve()
    result = root / 'results'
    out = (output or root / 'ICM_OUTCOME_SUMMARY_v1').resolve()
    hashes = {}

    def load(path, kind='json'):
        if out == path or out in path.parents:
            raise ValueError('Output must be separate from original report files')
        raw = path.read_bytes()
        hashes[str(path)] = digest(raw)
        text = raw.decode('utf-8-sig')
        return json.loads(text) if kind == 'json' else list(csv.DictReader(io.StringIO(text)))

    status = load(root/'STATUS.json')
    if not status.get('completed'):
        raise ValueError('统一重算尚未完成；待 STATUS.json 的 completed=true 后重复运行本命令。')
    plan = load(root/'PLAN.json')
    if plan['protocol'] != 'single-shared-random-rate-controlled-union-v2':
        raise ValueError('Source is not the corrected unified Table 5 experiment')
    canonical = json.dumps({k:v for k,v in plan.items() if k != 'sha256'},
                           sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    if digest(canonical) != plan['sha256']:
        raise ValueError('Source plan checksum does not match')
    report = load(result/'REPORT.json')
    if (not report['completed'] or report['plan_sha256'] != plan['sha256']
            or report['pairs_per_method'] != 54 or report['primary_reference'] != 'BASE'
            or not report['hv_already_divided'] or report['hv_divisor'] != 1.331):
        raise ValueError('Unified report metadata does not match')
    tolerance = float(report['wtl_tolerance'])
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('Invalid W/T/L tolerance')
    table = load(result/'TABLE5.csv', 'csv')
    pairs = load(result/'PER_PAIR.csv', 'csv')
    by_method = {r['method']:r for r in table}
    if len(by_method) != len(table) or set(by_method) != set(report['methods']):
        raise ValueError('Table 5 method identities do not match the unified report')
    methods = [m for m in by_method if m.startswith('ICM_')]
    shared = 'ICM_SHARED_SUPERVISED'
    if shared not in methods:
        raise ValueError('Shared SL is missing')
    methods = [shared] + [m for m in methods if m != shared]
    groups = defaultdict(list)
    for r in pairs:
        if int(r['evaluation']) == 5000 and r['reference_method'] == 'BASE':
            groups[r['method']].append(r)
    key = lambda r:(r['instance'],r['seed'])
    base = {key(r):r for r in groups['BASE']}
    if len(base) != len(groups['BASE']) or len(base) != 54:
        raise ValueError('Expected 54 unique Base pairs')
    cfg_instances = defaultdict(set)
    instance_seeds = defaultdict(set)
    for r in base.values():
        cfg_instances[r['configuration']].add(r['instance'])
        instance_seeds[r['instance']].add(r['seed'])
        own, ref = float(r['own']), float(r['reference'])
        if not math.isfinite(own) or own <= 0 or own != ref:
            raise ValueError('Invalid Base self-pair')
    if (set(cfg_instances) != {f'S{i}' for i in range(9)}
            or any(len(v) != 2 for v in cfg_instances.values())
            or len(instance_seeds) != 18 or any(len(v) != 3 for v in instance_seeds.values())
            or len({tuple(sorted(v)) for v in instance_seeds.values()}) != 1):
        raise ValueError('Expected nine configurations, two instances each, three common seeds')
    output_rows = []
    for method in methods:
        raw = groups[method]
        if len(raw) != 54 or {key(r) for r in raw} != set(base):
            raise ValueError('Missing/duplicate pairs: '+method)
        runs = []
        for r in raw:
            b = base[key(r)]
            own, ref = float(r['own']), float(r['reference'])
            if not math.isfinite(own) or not math.isfinite(ref) or ref <= 0 or own < 0:
                raise ValueError('Invalid paired HV: '+method)
            if r['configuration'] != b['configuration'] or ref != float(b['own']):
                raise ValueError('Reference does not match Base: '+method)
            difference = own - ref
            gain = 100*difference/ref
            close(difference, float(r['absolute']), 'Saved absolute HV difference')
            close(gain, float(r['percent']), 'Saved percentage gain')
            runs.append(dict(gain=gain,difference=difference))
        stats = summarize(runs,tolerance)
        original = by_method[method]
        if int(original['配对数']) != 54 or tuple(stats[k] for k in ('胜','平','负')) != tuple(int(original[k]) for k in ('胜','平','负')):
            raise ValueError('W/T/L does not reproduce Table 5: '+method)
        close(stats['全部运行平均提升百分比'],float(original['5000_平均配对提升百分比']), 'Table 5 mean '+method)
        output_rows.append(dict(method=method,方法=original['方法'],参照='Base',评价次数=5000,**stats))

    provenance = dict(protocol=VERSION,source=str(result),files=hashes,wtl_tolerance=tolerance,
        script_sha256=digest(Path(__file__).read_bytes()),
        percent='100 * (own HV - paired Base HV) / paired Base HV',
        mean_wins='Arithmetic mean of per-run percent over W only',
        mean_loss_decline='Negative of arithmetic mean of per-run percent over L only; positive magnitude',
        median='Median over all 54 paired percentages, including ties',
        table5_verified=True,new_searches=0,new_objective_evaluations=0,new_hv_calculations=0)
    if (out/'SOURCE.json').exists() and json.loads((out/'SOURCE.json').read_text()) != provenance:
        raise ValueError('Existing supplement uses different source files; choose a new --out')
    for name,expected in hashes.items():
        if digest(Path(name).read_bytes()) != expected:
            raise ValueError('Source changed during calculation: '+name)
    out.mkdir(parents=True,exist_ok=True)
    def write_atomic(name,content):
        p = out/name
        temp = p.with_name(p.name+'.writing')
        temp.write_text(content,encoding='utf-8')
        temp.replace(p)
    buf = io.StringIO()
    writer = csv.DictWriter(buf,fieldnames=list(output_rows[0]))
    writer.writeheader();writer.writerows(output_rows)
    write_atomic('SUMMARY.csv','\ufeff'+buf.getvalue())
    write_atomic('SOURCE.json',json.dumps(provenance,ensure_ascii=False,indent=2)+'\n')
    fmt = lambda x:'—' if x is None else f'{x:.4f}%'
    lines = ['统一 Table 5：ICM 在 B5000 相对 Base 的54次配对结果',
        '方法 | 胜/平/负 | 获胜平均提升 | 失败平均下降（正数） | 全部运行中位数提升 | 全部运行平均提升']
    for r in output_rows:
        lines.append(f"{r['方法']} | {r['胜']}/{r['平']}/{r['负']} | "+' | '.join(fmt(r[k]) for k in
            ('获胜运行平均提升百分比','失败运行平均下降百分比','全部运行中位数提升百分比','全部运行平均提升百分比')))
    lines += ['',f'W/T/L沿用统一报告中的HV差容差：{tolerance:.16g}（HV已除以1.331）。',
        '每次提升的分母均为同实例、同种子的Base HV；不是均值之比。',
        '获胜/失败均值在相应运行子集内等权；中位数取全部54次，含持平。无胜或无负时对应均值记为—。',
        '全体均值=(胜数×平均胜幅−负数×平均负幅+平数×平均平幅)/54，已核对新Table 5的均值和W/T/L。',
        '该面板每配置2实例、每实例3种子，54次等权均值与Table 5逐层等权均值一致。',
        '胜/负是数值比较，不代表每次运行的统计显著性；本次不新增检验、不改置信区间。']
    write_atomic('REPORT.txt','\n'.join(lines)+'\n')
    print('\n'.join(lines))
    print('\n[ICM-OUTCOMES] COMPLETE',out)
    return output_rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=DEFAULT)
    parser.add_argument('--out',type=Path)
    args = parser.parse_args()
    main(args.root,args.out)
