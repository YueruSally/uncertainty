"""Frozen-input adapters for full final-front reevaluation; no search entry points."""
import csv
import json
import math
from pathlib import Path

from inventory import (read, save, retain, require, canonical, digest, exact_id,
                       vector, DATA)
from run import RTOL, ATOL, REF, DIVISOR


def freeze_json(path, value):
    path = Path(path)
    if path.exists():
        require(read(path) == value, '已保存的新文件与当前输入不同: ' + str(path))
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(path.suffix + '.writing')
        temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        temp.replace(path)


def load_scenarios(np, ev, meta_path, data_path, legacy=False):
    meta = read(meta_path)
    if not legacy:
        require(digest(data_path) == meta['data_sha256'], '场景文件哈希不符')
    with np.load(data_path, allow_pickle=False) as data:
        tk = meta['travel_keys']
        bk = meta['event_keys'] if legacy else meta['border_keys']
        # NPZ decompresses an entire array on each __getitem__; load each once.
        travel_array, border_array = data['travel'], data['border']
        require(travel_array.shape == (len(tk), meta['size']) and
                border_array.shape == (len(bk), meta['size']), '场景数组维度不符')
        travel = {tuple(k): travel_array[i].copy() for i, k in enumerate(tk)}
        border = {tuple(k): border_array[i].copy() for i, k in enumerate(bk)}
    for a in list(travel.values()) + list(border.values()):
        require(np.isfinite(a).all(), '场景含非有限值')
        a.setflags(write=False)
    scenario = ev.base.ScenarioSet(size=meta['size'], seed=meta['seed'], stochastic=meta['stochastic'],
        travel_multiplier=travel, border_delay_h=border,
        arc_border_event={tuple(k): tuple(v) for k, v in meta['arc_border_event']},
        border_event_mean_h={tuple(k): v for k, v in meta['event_means' if legacy else 'border_event_mean_h']})
    if not legacy:
        require(ev.scenario_digest(scenario) == meta['scenario_digest'], '场景摘要不符')
    return scenario


def save_scenarios(np, ev, path, scenario):
    path = Path(path)
    if path.with_suffix('.json').exists():
        saved = load_scenarios(np, ev, path.with_suffix('.json'), path.with_suffix('.npz'))
        require(ev.scenario_digest(saved) == ev.scenario_digest(scenario), '冻结场景发生变化')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tk, bk = sorted(scenario.travel_multiplier), sorted(scenario.border_delay_h)
    temp = path.with_suffix('.writing')
    with temp.open('wb') as f:
        np.savez_compressed(f,
            travel=np.asarray([scenario.travel_multiplier[k] for k in tk]).reshape(-1, scenario.size),
            border=np.asarray([scenario.border_delay_h[k] for k in bk]).reshape(-1, scenario.size))
    temp.replace(path.with_suffix('.npz'))
    freeze_json(path.with_suffix('.json'), dict(size=scenario.size, seed=int(scenario.seed),
        stochastic=scenario.stochastic, travel_keys=tk, border_keys=bk,
        arc_border_event=[[k, v] for k, v in sorted(scenario.arc_border_event.items())],
        border_event_mean_h=[[k, v] for k, v in sorted(scenario.border_event_mean_h.items())],
        scenario_digest=ev.scenario_digest(scenario), data_sha256=digest(path.with_suffix('.npz'))))
    # Round-trip through the exact array reader before using the new file.
    load_scenarios(np, ev, path.with_suffix('.json'), path.with_suffix('.npz'))


def evaluator(ev, entry_path, config, workbook, alpha):
    """Same frozen restore/evaluate methods, with an explicit workbook for Table 7."""
    class WorkbookEvaluator(ev.Evaluator):
        def __init__(self):
            base = ev.base
            self.entry, self.batches = ev.read_instance(entry_path)
            self.uncertainty = config['uncertainty']
            base.BORDER_EVENT_DEFINITIONS = base.load_border_event_definitions(base.DEFAULT_BORDER_EVENT_DATA_FILE)
            (_, _, self.hold, self.proc, self.trans_cost, self.arcs, timetables, _,
             self.wait_cost, self.wait_emission, self.carbon, _, _, self.trans_map,
             border, self.theta, _) = base.load_network_from_extended(str(workbook))
            self.border = (base.load_border_delay_map_from_workbook(workbook,
                include_customs=self.uncertainty['border_include_customs'])
                if self.uncertainty['border_mean_source'] == 'node_border' else border)
            self.tt = base.build_timetable_dict(timetables)
            self.arcs_lookup, self.paths = base.build_arc_lookup(self.arcs), {}
            for batch in self.batches:
                batch.penalty_per_teu_h = base.DEFAULT_LATE_PENALTY_USD_PER_TEU_H
            base.ACTIVE_MUTATION_LOGGER = None
            base.RISK_METRIC = 'ccp'
            base.CONFIDENCE_COST = base.CONFIDENCE_EMISSION = base.CONFIDENCE_TIME = alpha
    return WorkbookEvaluator()


def project_scenarios(np, ev, evaluator, master):
    """Project large-network draws onto small network, exactly as original Table 7."""
    u = evaluator.uncertainty
    expected = ev.base.build_scenario_set(evaluator.arcs, evaluator.border, 1, master.seed,
        stochastic=False, border_event_definitions=ev.base.BORDER_EVENT_DEFINITIONS,
        **{k: u[k] for k in ('mode_time_cv', 'mode_time_max_factor', 'border_delay_cv',
                            'border_delay_max_factor', 'border_mean_source')})
    keys = sorted({(a.from_node, a.to_node, a.mode) for a in evaluator.arcs})
    require(all(master.arc_border_event.get(k) == expected.arc_border_event.get(k) for k in keys),
            '共有弧的口岸事件映射不同')
    require(all(master.border_event_mean_h.get(k) == v for k, v in expected.border_event_mean_h.items()),
            '共有口岸均值不同')
    scenario = ev.base.ScenarioSet(size=master.size, seed=master.seed, stochastic=True,
        travel_multiplier={k: master.travel_multiplier[k].copy() for k in keys},
        border_delay_h={k: master.border_delay_h[k].copy() for k in expected.border_delay_h},
        arc_border_event=expected.arc_border_event, border_event_mean_h=expected.border_event_mean_h)
    require(all(np.array_equal(a, master.travel_multiplier[k]) for k, a in scenario.travel_multiplier.items())
            and all(np.array_equal(a, master.border_delay_h[k]) for k, a in scenario.border_delay_h.items()),
            '大小网络共有抽样数组不一致')
    return scenario


def old_reports(roots, plans, inventory, entries, files):
    """Pin exact old reports for a numerical round trip, without using their means."""
    lookup, bounds = {}, {}
    for table, root in roots.items():
        folder = root / ('results_union' if table == 'Table6' else 'results')
        bp = folder / 'INSTANCE_BOUNDS.json'
        pp = folder / ('PER_RUN.csv' if table == 'Table7' else 'PER_PAIR.csv')
        retain(bp, files); retain(pp, files)
        bounds[table] = read(bp)
        if table != 'Table7':
            rp = folder / 'REPORT.json'
            retain(rp, files)
            report = read(rp)
            require(report['completed'] and report['plan_sha256'] == plans[table]['sha256']
                    and report['hv_already_divided'] and report['hv_divisor'] == DIVISOR
                    and report.get('hv_reference', report.get('reference')) == list(REF), '旧报告版本或HV单位不符')
        with pp.open(encoding='utf-8-sig', newline='') as f:
            for r in csv.DictReader(f):
                if int(r['evaluation']) != 5000:
                    continue
                if table == 'Table7':
                    key = (table, r['configuration'], r['network'], r['method'], int(r['seed']))
                    value = float(r['hv'])
                else:
                    if r['reference_method'] != 'BASE':
                        continue
                    key = (table, r['instance'], r['method'], int(r['seed']))
                    value = float(r['own'])
                require(key not in lookup, '旧HV记录重复')
                lookup[key] = value
    for r in inventory:
        if r['table'] == 'Table7':
            key = (r['table'], r['configuration'], r['network'], r['method'], r['seed'])
            bkey = r['configuration']
        else:
            key = (r['table'], r['instance'], r['method'], r['seed'])
            bkey = r['instance']
        require(key in lookup, '旧报告缺少该次运行HV: ' + str(key))
        r['old_hv'], r['old_bounds'] = lookup[key], bounds[r['table']][bkey]


def prepare_instances(out, inv, entries, files, pilot, alpha):
    """Deduplicate full decisions within each instance, preserving every membership."""
    pilot_inputs = read(pilot / 'PILOT_INPUTS.json')
    require(read(pilot / 'STATUS.json')['completed'] and pilot_inputs['alpha'] == alpha,
            '试评未完成或alpha不同')
    require(canonical({k: v for k, v in pilot_inputs.items() if k != 'sha256'}) == pilot_inputs['sha256'],
            '试评输入摘要不同')
    retain(pilot / 'PILOT_INPUTS.json', files)
    retain(pilot / 'STATUS.json', files)
    pilot_name = pilot_inputs['instance']['instance_id']
    groups, units = [], {}
    known_seeds = {int(e['ccp100_seed']) for es in entries.values() for e in es.values()}
    for table, es in entries.items():
        for name in sorted(es):
            e = es[name]
            unit = table + '/' + (e['pair_id'] if table == 'Table7' else name)
            if unit not in units:
                index = len(units)
                units[unit] = {str(n): 381000000 + 10 * index + j for j, n in enumerate((100, 1000, 5000))}
            seeds = units[unit]
            source = [r for r in inv if r['table'] == table and r['instance'] == name]
            configs = [read(Path(r['run_path']).parent / 'configuration.json') for r in source]
            require(configs and all(c['uncertainty'] == configs[0]['uncertainty'] for c in configs), '实例场景配置不一致')
            solutions, memberships = {}, []
            for r in source:
                for row in read(r['front_path']):
                    key = exact_id(row)
                    if key in solutions:
                        require(all(math.isclose(a, b, rel_tol=RTOL, abs_tol=ATOL)
                                    for a, b in zip(vector(solutions[key]), vector(row))), '同一决策原目标不一致')
                    else:
                        solutions[key] = dict(row, exact_decision_id=key)
                    memberships.append(dict(method=r['method'], seed=r['seed'],
                        exact_decision_id=key, source_solution_id=row['source_solution_id']))
            folder = out / 'instances' / table / name
            packet = dict(table=table, instance=e, source_config=configs[0], source_runs=source,
                decisions=[solutions[k] for k in sorted(solutions)], memberships=memberships,
                seeds=seeds, bounds_unit=unit, alpha=alpha, workbook=str(e.get('workbook', DATA)))
            if table == 'Table5' and name == pilot_name:
                require(e == pilot_inputs['instance'] and seeds == pilot_inputs['new_seeds'], 'S0实例或新场景种子与试评不符')
                for tag in ('optimization_100', 'fresh_100', 'fresh_1000', 'fresh_5000'):
                    for ext in ('.json', '.npz'):
                        retain(pilot / 'scenarios' / (tag + ext), files)
                packet['pilot_scenarios'] = str(pilot / 'scenarios')
            if table == 'Table7':
                for filename, field in (('SCENARIOS.json', 'scenario_metadata_sha256'), ('SCENARIOS.npz', 'scenarios_sha256')):
                    retain(Path(e['input_folder']) / filename, files, e[field])
            packet['sha256'] = canonical(packet)
            save(folder / 'INPUTS.json', packet)
            save(folder / 'instance.json', e)
            groups.append(dict(table=table, instance=name, network=e.get('network', 'expanded'),
                               folder=str(folder), bounds_unit=unit, inputs_sha256=packet['sha256'],
                               unique_decisions=len(solutions)))
    require(any('pilot_scenarios' in read(Path(g['folder']) / 'INPUTS.json') for g in groups), '试评S0未包含在全量面板')
    all_seeds = [s for mapping in units.values() for s in mapping.values()]
    require(len(all_seeds) == len(set(all_seeds)) and not set(all_seeds).intersection(known_seeds), '评估随机种子重复或与优化重叠')
    # Coupled networks also require the same uncertainty parameters and cargo identity.
    for unit in {g['bounds_unit'] for g in groups if g['table'] == 'Table7'}:
        parts = [read(Path(g['folder']) / 'INPUTS.json') for g in groups if g['bounds_unit'] == unit]
        # Actual sampling diagnostics differ with the numbers of arcs/events;
        # they are observations, not sampler parameters.
        profile_keys = ('mode_time_cv','mode_time_max_factor','border_delay_cv',
                        'border_delay_max_factor','border_mean_source','border_include_customs')
        require(len(parts) == 2 and parts[0]['instance']['batch_signature'] == parts[1]['instance']['batch_signature']
                and all(parts[0]['source_config']['uncertainty'][k] == parts[1]['source_config']['uncertainty'][k]
                        for k in profile_keys),
                'Table7共有批次或不确定性配置不一致')
    return groups
