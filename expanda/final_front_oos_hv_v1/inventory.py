"""Read-only inventory of the current Table 5/6/7 final decision fronts."""
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
DIAG = EXP / 'transformer_ppo_runs/diagnostics'
ENGINE = EXP / 'formal_icm_s0_v1/src'
DATA = EXP / 'formal_icm_s0_v1/assets/data/data_expanded.xlsx'
OBJECTIVES = ('cost', 'emission', 'makespan')
SHARED, PPO = 'ICM_SHARED_SUPERVISED', 'ICM_TERMINAL_PPO'
RANDOM, ZERO = 'ICM_RANDOM_RATE_CONTROLLED', 'ICM_SHARED_ZERO_THRESHOLD'
ARMS = ('BASE', 'I_RULE', 'I_LEARN', 'C_RULE', 'C_LEARN', 'M_RULE', 'M_LEARN',
        'ICM_HEURISTIC', 'ICM_INDEPENDENT', SHARED, PPO, RANDOM, ZERO)
ROOT_NAMES = dict(Table5='ccp_shared_sl_ablations_b5000_v2',
                  Table6='ccp_single_ppo_s9_s11_refresh_v1',
                  Table7='ccp_network_transfer_single_b5000_v1')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def csv_write(path, rows, fields=None):
    with Path(path).open('x', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def retain(path, files, expected=None):
    path = Path(path).resolve()
    actual = digest(path)
    require(expected is None or actual == expected, '文件哈希不同: ' + str(path))
    require(str(path) not in files or files[str(path)] == actual, '读取期间文件变化: ' + str(path))
    files[str(path)] = actual
    return actual


def verify(files):
    for name, expected in files.items():
        require(digest(name) == expected, '输入文件变化: ' + name)


def vector(row):
    return [float(row['optimisation_objectives'][o]) for o in OBJECTIVES]


def exact_id(row):
    blocks = []
    for block in row['decision']:
        allocations = [dict(nodes=a['nodes'], modes=a['modes'], share=a['share'])
                       for a in block['allocations']]
        blocks.append(dict(origin=block['origin'], destination=block['destination'],
                           batch_id=block['batch_id'], allocations=sorted(allocations, key=canonical)))
    return canonical(sorted(blocks, key=canonical))


def saved_fingerprint(row):
    """Same canonical representation as frozen decision_signature, without loading a solver."""
    blocks = []
    for block in row['decision']:
        key = (block['origin'], block['destination'], int(block['batch_id']))
        allocations = []
        for a in block['allocations']:
            share = round(float(a['share']), 12)
            if share == 0:
                share = 0.0
            allocations.append((tuple(a['nodes']), tuple(a['modes']), f'{share:.12f}'))
        blocks.append((key, tuple(sorted(allocations))))
    data = json.dumps(sorted(blocks), ensure_ascii=False, separators=(',', ':'))
    return hashlib.sha256(data.encode('utf-8')).hexdigest()


def validate_decision(row, entry):
    expected = {(b['origin'], b['destination'], int(b['batch_id'])) for b in entry['batches']}
    found = [(b['origin'], b['destination'], int(b['batch_id'])) for b in row['decision']]
    require(len(found) == len(expected) and set(found) == expected, '决策中批次缺失、重复或不符')
    require(row['training_feasible'] is True and all(math.isfinite(x) for x in vector(row)),
            '原最终解不是有限目标的可行解')
    require(saved_fingerprint(row) == row['decision_fingerprint'], '保存的决策与指纹不符')
    for block in row['decision']:
        alloc = block['allocations']
        require(bool(alloc), '批次没有路径分配')
        shares = [float(a['share']) for a in alloc]
        require(all(math.isfinite(s) and s >= 0 for s in shares)
                and abs(sum(shares) - 1) <= 1e-8, '路径份额不合法')
        for a in alloc:
            require(len(a['nodes']) >= 2 and len(a['modes']) == len(a['nodes']) - 1
                    and a['nodes'][0] == block['origin'] and a['nodes'][-1] == block['destination'],
                    '路径节点、运输方式或起讫点不完整')


def source_plan(table, root, files, alpha):
    path = root / 'PLAN.json'
    retain(path, files)
    plan = read(path)
    require(canonical({k: v for k, v in plan.items() if k != 'sha256'}) == plan['sha256'],
            '计划校验不通过: ' + str(path))
    status = read(root / 'STATUS.json')
    require(status.get('completed') is True, table + ' 尚未完成')
    settings = plan['settings']
    require((settings['population'], settings['budget'], settings['scenario_count'], settings['confidence'])
            == (250, 5000, 100, alpha), table + f' 不是 alpha={alpha}/pop250/B5000/100场景记录')
    entries = plan['entries']
    require(len({e['instance_id'] for e in entries}) == len(entries), '实例重复')
    if table == 'Table5':
        require(Counter(e['configuration'] for e in entries) == {f'S{i}': 2 for i in range(9)},
                'Table5 应为 S0–S8 每配置两个实例')
        seeds = settings['screen_seeds']
    elif table == 'Table6':
        require(Counter(e['configuration'] for e in entries) == {f'S{i}': 3 for i in range(9, 12)},
                'Table6 应为 S9–S11 每配置三个实例')
        seeds = plan['protocol']['algorithm_seeds']
    else:
        require(Counter((e['K'], e['network']) for e in entries)
                == {(k, n): 1 for k in (10, 20, 30) for n in ('small', 'large')}, 'Table7 实例面板不符')
        seeds = plan['algorithm_seeds']
    require(len(seeds) == len(set(seeds)) == 3, '必须有三个共同搜索种子')
    if table != 'Table6':
        require(sorted(seeds) == [993001, 993002, 993003], '搜索种子与原表不符')
    return plan, sorted(seeds)


def inspect_run(table, root, plan, entry, method, seed, path, files, alpha):
    for f in (path, path.parent / 'configuration.json', path.parent / 'final_feasible_nondominated.json'):
        retain(f, files, plan['files'].get(str(f)))
    run = read(path)
    config = read(path.parent / 'configuration.json')
    front_path = path.parent / 'final_feasible_nondominated.json'
    front = read(front_path)
    expected_arm = method
    # The historical zero-update PPO alias is allowed only if the source plan says so.
    if method == PPO and plan.get('selected_ppo_update', plan.get('selected_update')) == 0:
        expected_arm = SHARED
    require(run.get('completed') and run['instance'] == entry['instance_id']
            and run['configuration'] == entry['configuration'] and run['seed'] == seed
            and run['arm'] == expected_arm and run['instance_digest'] == entry['instance_digest']
            and run['evaluation_budget'] == 5000
            and run['real_objective_evaluation_count'] in (4999, 5000), '运行身份/完成状态/预算不符')
    require(config['evaluation_mode'] == 'ccp' and config['alpha'] == alpha
            and config['population'] == 250 and config['evaluation_budget'] == 5000
            and config['training_size'] == 100 and config['instance_id'] == entry['instance_id']
            and config['instance_digest'] == entry['instance_digest']
            and config['seeds']['algorithm'] == seed
            and config['seeds']['training'] == entry['ccp100_seed']
            and config['scenario_digest'] == run['scenario_digest'], '原评价配置或场景不符')
    if 'final_solutions_sha256' in run:
        require(files[str(front_path.resolve())] == run['final_solutions_sha256'], '最终解文件哈希不符')
    require(isinstance(front, list) and bool(front), '最终解集缺失或为空')
    require(sorted(vector(r) for r in front) == sorted(run['final_front_raw']), '决策文件与最终前沿不一致')
    endpoint = [s for s in run['budget_fronts'] if s['evaluation'] == 5000]
    require(len(endpoint) == 1 and sorted(endpoint[0]['front_raw']) == sorted(run['final_front_raw']),
            '5000记录点不是保存的实际最终前沿')
    for row in front:
        validate_decision(row, entry)
    return dict(table=table, configuration=entry['configuration'], instance=entry['instance_id'],
                network=entry.get('network', 'expanded'), method=method, seed=seed, status='saved',
                final_solutions=len(front), actual_evaluations=run['real_objective_evaluation_count'],
                optimization_scenario_seed=entry['ccp100_seed'], scenario_digest=run['scenario_digest'],
                run_path=str(path), front_path=str(front_path), reason='')


def inventory(roots, out, alpha=.9):
    files, plans, rows, errors, entries_by_table = {}, {}, [], [], {}
    for table, root in roots.items():
        print('[FINAL-OOS] inventory ' + table, flush=True)
        try:
            plan, seeds = source_plan(table, root, files, alpha)
            plans[table] = plan
            entries_by_table[table] = {e['instance_id']: e for e in plan['entries']}
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors.append(dict(table=table, method='', instance='', seed='', path=str(root), reason=str(exc)))
            continue
        sources = {}
        for method, paths in plan.get('source_records', {}).items():
            # Old Simple/old PPO in Table6 are historical normalization-only sources.
            wanted = (table == 'Table5' and method in ARMS and method != RANDOM
                      or table == 'Table6' and method in ('BASE', SHARED))
            if not wanted:
                continue
            for name in paths:
                path = Path(name)
                # Folder names are used so missing RUN_COMPLETE files can still be listed.
                try:
                    key = (method, path.parent.parent.parent.name, int(path.parent.parent.name.removeprefix('seed_')))
                    require(key not in sources, '来源列表中运行重复')
                    sources[key] = path
                except (ValueError, IndexError) as exc:
                    errors.append(dict(table=table, method=method, instance='', seed='', path=name, reason=str(exc)))
        expected_source_keys = set()
        for entry in plan['entries']:
            name = entry['instance_id']
            methods = ARMS if table == 'Table5' else (('BASE', SHARED, PPO) if table == 'Table6'
                         else (('BASE', SHARED) if entry['network'] == 'small' else (SHARED,)))
            entry_error = None
            try:
                require(canonical({k: v for k, v in entry.items() if k != 'instance_digest'})
                        == entry['instance_digest'], '实例摘要不符')
                ip = root / 'instances' / (name + '.json')
                retain(ip, files, plan['files'].get(str(ip)))
                require(read(ip) == entry, '实例副本与计划不同')
                workbook = Path(entry.get('workbook', DATA))
                retain(workbook, files, entry['source_data_sha256'])
            except (OSError, ValueError, KeyError) as exc:
                entry_error = str(exc)
            for method in methods:
                for seed in seeds:
                    from_source = (table == 'Table5' and method != RANDOM
                                   or table == 'Table6' and method in ('BASE', SHARED))
                    key = (method, name, seed)
                    if from_source:
                        expected_source_keys.add(key)
                    path = sources.get(key) if from_source else root / 'runs' / name / f'seed_{seed}' / method / 'RUN_COMPLETE.json'
                    try:
                        require(entry_error is None, entry_error)
                        require(path is not None, '来源计划缺少该运行路径')
                        record = inspect_run(table, root, plan, entry, method, seed, path, files, alpha)
                    except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
                        reason = str(exc)
                        errors.append(dict(table=table, method=method, instance=name, seed=seed,
                                           path=str(path or ''), reason=reason))
                        record = dict(table=table, configuration=entry['configuration'], instance=name,
                                      network=entry.get('network', 'expanded'), method=method, seed=seed,
                                      status='missing_or_invalid', final_solutions='', actual_evaluations='',
                                      optimization_scenario_seed=entry['ccp100_seed'], scenario_digest='',
                                      run_path=str(path or ''), front_path='', reason=reason)
                    rows.append(record)
        for key in set(sources) - expected_source_keys:
            errors.append(dict(table=table, method=key[0], instance=key[1], seed=key[2],
                               path=str(sources[key]), reason='来源列表含未安排的运行'))
    # All methods/seeds within one instance must use identical optimization draws/profile.
    grouped = defaultdict(list)
    for row in rows:
        if row['status'] == 'saved':
            grouped[row['table'], row['instance']].append(row)
    for (table, name), group in grouped.items():
        configs = [read(Path(r['run_path']).parent / 'configuration.json') for r in group]
        if len({r['scenario_digest'] for r in group}) != 1 or any(c['uncertainty'] != configs[0]['uncertainty'] for c in configs):
            errors.append(dict(table=table, method='ALL', instance=name, seed='', path='',
                               reason='同实例各方法/种子的优化场景或不确定性配置不一致'))
    # Verify the evaluator and input files against historical pinned manifests, without loading models.
    runtime = [p for p in ENGINE.rglob('*') if p.is_file() and p.suffix in ('.py', '.csv')]
    runtime += list((EXP / 'fixed_q90_s0_s11_v1').glob('*.py'))
    for path in runtime:
        try:
            actual = retain(path, files)
            for table, plan in plans.items():
                expected = plan['files'].get(str(path))
                require(expected is None or expected == actual, table + ' 冻结评价代码不同: ' + str(path))
        except (OSError, ValueError) as exc:
            errors.append(dict(table='runtime', method='', instance='', seed='', path=str(path), reason=str(exc)))
    for table, expected_count in dict(Table5=702, Table6=81, Table7=27).items():
        actual = sum(r['table'] == table for r in rows)
        if actual != expected_count:
            errors.append(dict(table=table, method='ALL', instance='', seed='', path=str(roots[table]),
                               reason=f'预期 {expected_count} 次运行，列出 {actual} 次'))
    csv_write(out / 'INVENTORY.csv', rows, list(rows[0]) if rows else ['table', 'status'])
    csv_write(out / 'MISSING_OR_INVALID.csv', errors, ['table', 'method', 'instance', 'seed', 'path', 'reason'])
    counts = Counter((r['table'], r['method'], r['status']) for r in rows)
    csv_write(out / 'INVENTORY_SUMMARY.csv', [dict(table=t, method=m, status=s, runs=n)
                for (t, m, s), n in sorted(counts.items())], ['table', 'method', 'status', 'runs'])
    save(out / 'SOURCE_FILES.json', files)
    save(out / 'INVENTORY_RESULT.json', dict(passed=not errors, alpha=alpha, expected_runs=810,
         listed_runs=len(rows), saved_runs=sum(r['status'] == 'saved' for r in rows), errors=len(errors),
         roots={k: str(v) for k, v in roots.items()}, final_search_budget=5000,
         accepted_actual_counts=[4999, 5000], new_searches=0, new_objective_evaluations=0,
         scope='Current displayed methods only: Table5=13; Table6=Base/shared/current S0-S8 PPO; Table7=small Base/shared and large shared.'))
    print('[FINAL-OOS] inventory:', len(rows), 'runs;', len(errors), 'issues', flush=True)
    if errors:
        print('STOP: 查看 MISSING_OR_INVALID.csv；未开始场景生成或重新评价。', flush=True)
        raise ValueError('最终解或来源核对不通过；必须先报告，不能继续试评')
    verify(files)
    return rows, plans, entries_by_table, files
