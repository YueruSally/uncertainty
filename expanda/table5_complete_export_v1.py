#!/usr/bin/env python3
"""Export the completed unified 13-method Table 5 without mixing earlier reports."""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import sys

sys.dont_write_bytecode = True
DEFAULT = Path(__file__).resolve().parent/'transformer_ppo_runs/diagnostics/ccp_shared_sl_ablations_b5000_v2'
NAMES = {
    'BASE':'Base NSGA-II', 'I_RULE':'I (Directed)', 'I_LEARN':'I Transformer (SL)',
    'C_RULE':'C (Simple)', 'C_LEARN':'C Transformer (SL)',
    'M_RULE':'M (Balanced)', 'M_LEARN':'M Transformer (SL)',
    'ICM_HEURISTIC':'ICM (rules)',
    'ICM_INDEPENDENT':'ICM Transformer (independent SL)',
    'ICM_SHARED_SUPERVISED':'ICM Transformer (shared SL)',
    'ICM_TERMINAL_PPO':'ICM Transformer (shared SL + PPO)',
    'ICM_RANDOM_RATE_CONTROLLED':'ICM (random replacement)',
    'ICM_SHARED_ZERO_THRESHOLD':'ICM Transformer (shared SL, threshold = 0)',
}
HEADERS = ['Method', '400提升(%)', '1000提升(%)', '1000平均HV', '1000提升95%区间(%)',
           '1000累计时间(s)', '1000 W/T/L', '5000提升(%)', '5000平均HV',
           '5000提升95%区间(%)', '5000运行时间(s)', '5000 W/T/L']


def number(row, field):
    value = float(row[field])
    if not math.isfinite(value):
        raise ValueError('Nonfinite field: '+field)
    return value


def equal(a, b, label):
    if not math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(f'TABLE5/ALL_RESULTS mismatch ({label}): {a} != {b}')


def main(root, output=None):
    root = root.resolve()
    source = root/'results'
    out = (output or root/'TABLE5_COMPLETE_EXPORT_v1').resolve()
    hashes = {}
    def load(path, kind='json'):
        if out == path or out in path.parents:
            raise ValueError('Keep exports separate from original source files')
        raw = path.read_bytes()
        hashes[str(path)] = hashlib.sha256(raw).hexdigest()
        text = raw.decode('utf-8-sig')
        return json.loads(text) if kind == 'json' else list(csv.DictReader(io.StringIO(text)))
    status = load(root/'STATUS.json')
    meta = load(source/'REPORT.json')
    plan = load(root/'PLAN.json')
    if (not status.get('completed') or not meta.get('completed')
            or plan['protocol'] != 'single-shared-random-rate-controlled-union-v2'
            or meta['plan_sha256'] != plan['sha256']
            or meta['primary_reference'] != 'BASE' or meta['pairs_per_method'] != 54
            or meta['hv_divisor'] != 1.331 or not meta['hv_already_divided']
            or set(meta['methods']) != set(NAMES)):
        raise ValueError('Requires completed unified v2 report: 13 methods, 54 pairs, HV already divided by 1.331')
    payload = json.dumps({k:v for k,v in plan.items() if k != 'sha256'},
                         sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    if hashlib.sha256(payload).hexdigest() != plan['sha256']:
        raise ValueError('Source PLAN checksum differs')
    table = load(source/'TABLE5.csv','csv')
    all_rows = load(source/'ALL_RESULTS.csv','csv')
    by_method = {r['method']:r for r in table}
    if len(by_method) != 13 or len(table) != 13 or set(by_method) != set(NAMES):
        raise ValueError('Expected all thirteen Table 5 rows')
    lookup = {}
    for row in all_rows:
        if row['reference_method'] != 'BASE':
            continue
        key = row['method'], int(row['评价次数'])
        if key in lookup:
            raise ValueError('Duplicate method/budget: '+str(key))
        lookup[key] = row
    if set(lookup) != {(m,b) for m in NAMES for b in (400,1000,5000)}:
        raise ValueError('Incomplete unified method/budget panel')
    result = []
    for method, label in NAMES.items():
        old = by_method[method]
        if int(old['配对数']) != 54:
            raise ValueError('Wrong Table 5 pair count')
        for budget in (400,1000,5000):
            row = lookup[method,budget]
            if int(row['配对数']) != 54 or row['参照'] != 'Base':
                raise ValueError('Wrong comparator or pair count')
            for name in ('平均HV','平均配对提升百分比','提升95区间下限','提升95区间上限'):
                equal(number(row,name),number(old,f'{budget}_{name}'),method+'/'+str(budget)+'/'+name)
            if number(row,'提升95区间下限') > number(row,'提升95区间上限'):
                raise ValueError('Reversed interval')
            if method != 'BASE':
                counts = [int(row[k]) for k in ('胜','平','负')]
                if min(counts) < 0 or sum(counts) != 54:
                    raise ValueError('Invalid W/T/L: '+str((method,budget)))
            if budget == 5000:
                if [row[k] for k in ('胜','平','负')] != [old[k] for k in ('胜','平','负')]:
                    raise ValueError('B5000 W/T/L differs from Table 5')
                for name in ('B5000搜索平均秒','B5000搜索标准差秒'):
                    equal(number(row,name),number(old,name),method+'/'+name)
        cells = [label,f"{number(lookup[method,400],'平均配对提升百分比'):.2f}"]
        for budget in (1000,5000):
            row = lookup[method,budget]
            avg,sd = ('节点累计平均秒','节点累计标准差秒') if budget == 1000 else ('B5000搜索平均秒','B5000搜索标准差秒')
            mean_time,sd_time = number(row,avg),number(row,sd)
            if mean_time < 0 or sd_time < 0:
                raise ValueError('Invalid time')
            ci = '—' if method == 'BASE' else f"[{number(row,'提升95区间下限'):+.2f}, {number(row,'提升95区间上限'):+.2f}]"
            wtl = '—' if method == 'BASE' else '/'.join(row[k] for k in ('胜','平','负'))
            cells += [f"{number(row,'平均配对提升百分比'):.2f}",f"{number(row,'平均HV'):.6f}",
                      ci,f'{mean_time:.2f} ± {sd_time:.2f}',wtl]
        result.append(cells)
    evidence = dict(version='table5-complete-export-v1',source=str(source),input_sha256=hashes,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        methods=13,pairs_per_method=54,reference='BASE',hv_already_divided_by=1.331,
        time1000='ALL_RESULTS: 节点累计平均秒 / 节点累计标准差秒',
        time5000='ALL_RESULTS: B5000搜索平均秒 / B5000搜索标准差秒',
        searches=0,hv_recomputations=0,bootstrap_recomputations=0)
    if (out/'SOURCE.json').exists() and json.loads((out/'SOURCE.json').read_text()) != evidence:
        raise ValueError('Existing export has different inputs; use a new --out')
    for path,h in hashes.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != h:
            raise ValueError('Source report changed while reading')
    out.mkdir(parents=True,exist_ok=True)
    def save(name,text):
        target = out/name
        temp = target.with_name(target.name+'.writing')
        temp.write_text(text,encoding='utf-8')
        temp.replace(target)
    stream = io.StringIO()
    writer = csv.writer(stream);writer.writerow(HEADERS);writer.writerows(result)
    save('TABLE5_COMPLETE.csv','\ufeff'+stream.getvalue())
    lines = ['全部使用统一重算v2报告；每方法54对；提升、区间和W/T/L均相对Base。',
             'HV已除以1.331；时间为均值±样本标准差。1,000次用节点累计时间，5,000次用原完整搜索时间。',
             'Base自比较的区间与W/T/L显示—；0.00表示数值为零或四舍五入到零。','',
             '| '+' | '.join(HEADERS)+' |','| '+' | '.join(['---']*len(HEADERS))+' |']
    lines += ['| '+' | '.join(cells)+' |' for cells in result]
    save('TABLE5_COMPLETE.txt','\n'.join(lines)+'\n')
    save('SOURCE.json',json.dumps(evidence,ensure_ascii=False,indent=2)+'\n')
    print('\n'.join(lines))
    print('\n[TABLE5-COMPLETE] COMPLETE',out)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=DEFAULT)
    parser.add_argument('--out',type=Path)
    args = parser.parse_args()
    main(args.root,args.out)
