#!/usr/bin/env python3
"""Read-only Section 6.8 gate. No search, objective evaluation, or training.

This first-stage command deliberately cannot launch the pilot or formal runs.
The source manifests, validation instances and existing OOS packs must first
be inspected on the machine that holds the immutable experiment records.
"""
import argparse
import ast
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import socket
import subprocess
import sys

sys.dont_write_bytecode = True
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['MPLBACKEND'] = 'Agg'
HERE = Path(__file__).resolve().parent
EXP = HERE.parent
DIAG = EXP / 'transformer_ppo_runs/diagnostics'
ENGINE = EXP / 'formal_icm_s0_v1/src'
DATA = EXP / 'formal_icm_s0_v1/assets/data/data_expanded.xlsx'
SEEDS = [993001, 993002, 993003]
TARGETS = {'S6': (30, .5), 'S8': (30, 2.)}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def canonical(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def save(path, obj):
    # A fresh audit directory is mandatory: never replace an earlier report.
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(obj, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def command(args):
    p = subprocess.run(args, cwd=EXP, capture_output=True, text=True, check=False)
    return {'returncode': p.returncode, 'stdout': p.stdout.strip(), 'stderr': p.stderr.strip()}


def literal_assignments(path):
    out = {}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign):
            try:
                value = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                continue
            for target in node.targets:
                if isinstance(target, ast.Name):
                    out[target.id] = value
    return out


def main(args):
    out = args.out.resolve()
    sources = [args.table5.resolve(), args.ppo.resolve()]
    if out.exists() or any(out == p or p in out.parents or out in p.parents for p in sources):
        raise ValueError('必须使用全新的、与原实验分离的输出目录：' + str(out))
    out.mkdir(parents=True)
    os.environ['MPLCONFIGDIR'] = str(out/'matplotlib_cache')
    files, checks, selected, inventory = {}, [], [], []
    report = dict(protocol='section68-preflight-v1', completed=False,
                  ready_for_pilot=False, new_searches=0, new_objective_evaluations=0,
                  training_updates=0, selected=selected, checks=checks,
                  output=str(out), sources=[str(p) for p in sources])

    def check(name, ok, detail):
        checks.append(dict(check=name, passed=bool(ok), detail=detail))
        print(('PASS ' if ok else 'STOP ') + name + ': ' + str(detail), flush=True)
        if not ok:
            raise ValueError(name + ': ' + str(detail))

    def pin(path, expected=None):
        path = Path(path).resolve()
        got = sha(path)
        if expected is not None and got != expected:
            raise ValueError('冻结源文件校验失败：' + str(path))
        files[str(path)] = got
        return read(path) if path.suffix == '.json' else got

    def plan(path):
        q = pin(path)
        check('计划完整性 ' + str(path),
              canonical({k: v for k, v in q.items() if k != 'sha256'}) == q['sha256'], q['sha256'])
        return q

    try:
        environment = dict(hostname=socket.gethostname(), python=platform.python_version(),
            platform=platform.platform(), executable=sys.executable,
            cpu_count=os.cpu_count(), affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
            threads={k: os.environ[k] for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')},
            maximum_search_workers=2, commit=command(['git', 'rev-parse', 'HEAD']),
            worktree=command(['git', 'status', '--short']))
        if sys.platform.startswith('linux'):
            environment['lscpu'] = command(['lscpu', '-J'])
            environment['meminfo'] = Path('/proc/meminfo').read_text()
        save(out/'ENVIRONMENT.json', environment)
        check('计时节点', socket.gethostname().split('.')[0] == 'lnx-grid-41', socket.gethostname())
        check('Table 5 已完成', pin(args.table5/'STATUS.json').get('completed'), str(args.table5))
        t5 = plan(args.table5/'PLAN.json')
        ppo = plan(args.ppo/'PLAN.json')
        training_root = Path(ppo['training_source'])
        training = plan(training_root/'PLAN.json')
        s = t5['settings']
        check('Table 5 搜索设置',
              (s['population'], s['budget'], s['scenario_count'], s['confidence'], s['screen_seeds'])
              == (250, 5000, 100, .9, SEEDS),
              {k: s[k] for k in ('population', 'budget', 'scenario_count', 'confidence', 'screen_seeds')})
        runtime = list(ENGINE.rglob('*.py')) + [ENGINE/'data/border_crossing_events.csv', DATA]
        for name in ('component_joint_b5000_v1', 'joint_icm_multiscale_terminal_v1',
                     'joint_icm_multiscale_v1', 'component_screen_b20_v1'):
            runtime.extend((EXP/name).glob('*.py'))
        for path in sorted(set(runtime)):
            expected = t5['files'].get(str(path.resolve()))
            if expected is None:
                check('运行源已由 Table 5 固定', False, str(path))
            pin(path, expected)
        check('冻结求解器、控制器和数据', True, str(len(set(runtime)))+' 个文件哈希与 Table 5 相同')
        constants = literal_assignments(ENGINE/'baseline_uncertainty.py')
        check('交叉与变异概率', (constants['CROSSOVER_RATE'], constants['MUTATION_RATE']) == (.90, .15),
              {k: constants[k] for k in ('CROSSOVER_RATE', 'MUTATION_RATE')})
        base_rows = []
        for name in t5['source_records']['BASE']:
            r = pin(name, t5['files'][name])
            c = pin(Path(name).parent/'configuration.json')
            valid = (r['completed'] and r['arm'] == 'BASE' and not r['stochastic']
                  and r['enabled_components'] == [] and r.get('model_bundle_sha256') is None
                  and all(r['selection'][comp]['calls'] == 0 for comp in 'ICM')
                  and c['population'] == 250 and c['evaluation_budget'] == 5000
                  and c['training_size'] == 100 and c['alpha'] == .9)
            if not valid:
                check('Base 控制器关闭', False, name)
            base_rows.append((name, r, c))
        check('Table 5 Base 记录', len(base_rows) == 54
              and len({(r['instance'], r['seed']) for _, r, _ in base_rows}) == 54
              and all(r.get('hostname', '').split('.')[0] == 'lnx-grid-41' for _, r, _ in base_rows),
              dict(runs=len(base_rows), I_C_M_calls=0, model_loaded=False,
                   recorded_evaluation_counts=sorted({r['real_objective_evaluation_count'] for _, r, _ in base_rows})))
        check('Base 共同种子', sorted({r['seed'] for _, r, _ in base_rows}) == SEEDS, SEEDS)

        # Import only the same frozen engine; importing never starts a search.
        sys.path.insert(0, str(ENGINE/'support')); sys.path.insert(0, str(ENGINE))
        import numpy as np
        import baseline_uncertainty as base
        from transformer_ppo.episode import resolve_uncertainty_profile
        from transformer_ppo.instance_catalog import read_instance
        from run_ev_ccp_oos_pilot import scenario_digest
        check('冻结评价器', Path(base.__file__).resolve() == ENGINE/'baseline_uncertainty.py', base.__file__)
        u = resolve_uncertainty_profile('ccp', 'transport_border_saa_v1', 100)
        check('不确定性设置', all(c['uncertainty'] == u for _, _, c in base_rows), u)
        network = base.load_network_from_extended(str(DATA))
        arcs = network[5]
        border = base.load_border_delay_map_from_workbook(DATA, include_customs=False)
        definitions = base.load_border_event_definitions(base.DEFAULT_BORDER_EVENT_DATA_FILE)
        ev = base.build_expected_value_scenario_set(arcs, border, seed=0,
            border_event_definitions=definitions, border_mean_source='node_border')
        check('EV 平均输入', ev.size == 1 and not ev.stochastic
              and all(np.array_equal(v, np.ones(1)) for v in ev.travel_multiplier.values())
              and all(np.array_equal(v, np.array([ev.border_event_mean_h[k]])) for k, v in ev.border_delay_h.items())
              and bool([v for k, v in border.items() if k[1] == 'rail'])
              and all(v == 40.6 for k, v in border.items() if k[1] == 'rail')
              and all(v == border.get((k[0],k[2]),0.) + border.get((k[1],k[2]),0.)
                      for k,v in ev.border_event_mean_h.items()),
              dict(transport_multiplier=1,
                   rail_node_means=sorted({v for k,v in border.items() if k[1]=='rail'}),
                   rail_crossing_event_means=sorted({v for k,v in ev.border_event_mean_h.items() if k[2]=='rail'}),
                   convention='沿用 Table 5：跨境弧的事件均值为两端节点均值之和；40.6+40.6=81.2，不是每个过境事件40.6'))
        a = np.arange(1, 101, dtype=float)
        check('CCP 和最坏情况次序统计量', base.empirical_ccp_quantile(a, .9) == 90
              and base.empirical_ccp_quantile(a, 1.) == 100
              and base.empirical_ccp_quantile(np.arange(1, 5001), .9) == 4500,
              '100: 第90/100小；5000: P90为第4500小，无插值')
        for cfg, (k, alpha) in TARGETS.items():
            candidates = sorted((e for e in training['entries']
                                 if e['configuration'] == cfg and e['development_role'] == 'validation'),
                                key=lambda e: e['instance_id'])
            check('验证实例存在 ' + cfg, bool(candidates), [e['instance_id'] for e in candidates])
            e = candidates[0]
            check('实例规格 ' + cfg, (e['K'], e['deadline_window_alpha'], len(e['batches'])) == (k, alpha, k),
                  dict(instance=e['instance_id'], batches=e['K'], window=e['deadline_window_alpha']))
            entry_path = training_root/'instances'/(e['instance_id']+'.json')
            check('验证实例文件 ' + cfg, pin(entry_path, training['files'].get(str(entry_path))) == e, str(entry_path))
            actual, batches = read_instance(entry_path)
            check('实例网络哈希 ' + cfg, actual['source_data_sha256'] == sha(DATA), actual['source_data_sha256'])
            scenario = base.build_scenario_set(arcs, border, 100, e['ccp100_seed'], stochastic=True,
                border_event_definitions=definitions, mode_time_cv=u['mode_time_cv'],
                mode_time_max_factor=u['mode_time_max_factor'], border_delay_cv=u['border_delay_cv'],
                border_delay_max_factor=u['border_delay_max_factor'], border_mean_source=u['border_mean_source'])
            matches = [dict(seed=r['seed'], record=name, front=str(Path(name).parent/'final_feasible_nondominated.json'))
                       for name, r, _ in base_rows if r['instance'] == e['instance_id']]
            for name, r, _ in base_rows:
                if r['instance'] == e['instance_id']:
                    check('Table 5 同实例场景 ' + cfg, r['scenario_digest'] == scenario_digest(scenario), r['scenario_digest'])
            # Paths and feasibility checks do not evaluate a complete plan.
            # The original Table 5 engine uses a nominal option set for CCP.
            tt = base.build_timetable_dict(network[6])
            lookup = base.build_arc_lookup(arcs)
            random.seed(e['path_seed']); np.random.seed(e['path_seed'] % (2**32-1))
            paths = base.build_path_library(network[0], network[1], arcs, batches, tt, lookup)
            options_ev = base.build_reliable_path_options(batches, paths, tt, network[13], border, ev, mode='deterministic')
            options_ccp = base.build_reliable_path_options(batches, paths, tt, network[13], border, ev, mode='ev')

            def option_description(options):
                return [dict(batch=list(key), options=[dict(nodes=o.path.nodes, modes=o.path.modes,
                        on_time_probability=o.on_time_probability, max_lateness_h=o.max_lateness_h,
                        resources=sorted([(repr(k), v) for k, v in o.resources.items()])) for o in items])
                        for key, items in sorted(options.items())]

            a_options, b_options = option_description(options_ev), option_description(options_ccp)
            check('相同名义可行路径集 ' + cfg, a_options == b_options
                  and len(options_ev) == k and all(options_ev.values()),
                  dict(batches=len(options_ev), min_paths=min(map(len, options_ev.values()), default=0),
                       max_paths=max(map(len, options_ev.values()), default=0), sha256=canonical(a_options)))
            save(out/(cfg+'_NOMINAL_PATH_OPTIONS.json'), a_options)
            selected.append(dict(configuration=cfg, instance=e['instance_id'], instance_file=str(entry_path),
                K=k, deadline_window_alpha=alpha, role=e['development_role'],
                instance_digest=e['instance_digest'], path_seed=e['path_seed'],
                optimization_seed=e['ccp100_seed'], optimization_scenario_digest=scenario_digest(scenario),
                existing_oos_seed=e.get('oos5000_seed'), search_seeds=SEEDS,
                exact_table5_reproduction_sources=matches,
                reproduction_note='可逐点核对' if matches else 'Table 5 无同实例 Base；按用户条件本项不适用'))
            save(out/(cfg+'_INSTANCE.json'), e)
        # Inventory inputs rather than interpreting objective-sample packs as
        # network scenarios. A reference_1000 result pack is NOT 5000 scenarios.
        wanted = {e['instance'] for e in selected}
        for folder, dirs, names in os.walk(DIAG):
            dirs[:] = [d for d in dirs if d not in ('backup', 'backups', '__pycache__', '.git')]
            here = Path(folder)
            if here == out or out in here.parents:
                dirs[:] = []; continue
            for name in names:
                path = here/name
                lowered = str(path).lower()
                if name in ('ENVIRONMENT.json', 'PLAN.json', 'FORMAL_EXPERIMENT_PLAN.json') or (
                    path.suffix.lower() in ('.json', '.npz', '.pkl', '.pickle')
                    and ('5000' in name.lower() or 'scenario' in name.lower() or 'oos' in name.lower())
                    and ('scenario' in lowered or 'oos' in lowered)):
                    inventory.append(dict(path=str(path), bytes=path.stat().st_size,
                        selected_instance_in_path=any(i in str(path) for i in wanted),
                        kind='environment' if name=='ENVIRONMENT.json' else
                             'experiment_manifest' if name in ('PLAN.json','FORMAL_EXPERIMENT_PLAN.json') else 'candidate_scenario_file'))
        save(out/'EXISTING_SCENARIO_AND_ENVIRONMENT_FILES.json', inventory)
        report.update(completed=True, status='awaiting_source_review', uncertainty=u,
            required_next_checks=[
                '核对已有5000独立场景的元数据、实例和来源；不能把目标值样本包当成场景输入',
                '若无可复用场景，记录新的固定独立种子并保存实际场景；三方法三种子共用',
                '将ENVIRONMENT与论文6.1的原记录比对；当前主机信息不等于已核对历史资源分配',
                '固定同一名义路径/可行集后再进行三次完整试跑；此入口不会启动试跑'])
    except Exception as exc:
        report.update(status='stopped', error=type(exc).__name__ + ': ' + str(exc))
        print('[SECTION68] STOP:', report['error'], flush=True)
    finally:
        save(out/'SOURCE_HASHES.json', files)
        save(out/'PREFLIGHT.json', report)
        if selected:
            with (out/'SELECTED_INSTANCES.csv').open('x', encoding='utf-8-sig', newline='') as stream:
                columns = ('configuration','instance','K','deadline_window_alpha','role','path_seed','optimization_seed','existing_oos_seed')
                w = csv.DictWriter(stream, fieldnames=columns, extrasaction='ignore'); w.writeheader(); w.writerows(selected)
        print(json.dumps({k: report[k] for k in ('completed','ready_for_pilot','status','selected',
                                               'new_searches','new_objective_evaluations','training_updates')},
                         ensure_ascii=False, indent=2), flush=True)
        print('[SECTION68] Report:', out, flush=True)
    return 0 if report.get('completed') else 2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--table5', type=Path, default=DIAG/'ccp_shared_sl_ablations_b5000_v2')
    parser.add_argument('--ppo', type=Path, default=DIAG/'ccp_single_ppo_s0_s8_b5000_v1')
    parser.add_argument('--out', type=Path, default=DIAG/('section68_preflight_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')))
    raise SystemExit(main(parser.parse_args()))
