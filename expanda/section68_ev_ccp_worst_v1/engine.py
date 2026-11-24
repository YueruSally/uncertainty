"""Isolated adapters; the Table 5 solver and evaluator files stay unchanged."""
import csv
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np

from preflight import ENGINE, DATA, EXP, canonical, read, sha, save, option_description

OBJECTIVES = ('cost', 'emission', 'makespan')
sys.path.insert(0, str(ENGINE / 'support'))
sys.path.insert(0, str(ENGINE))
import baseline_uncertainty as base
from transformer_ppo.instance_catalog import read_instance
from run_ccp_candidate_pool import decision_signature
from run_ev_ccp_oos_pilot import scenario_digest

if Path(base.__file__).resolve() != ENGINE / 'baseline_uncertainty.py':
    raise RuntimeError('Wrong evaluator loaded')


def csv_write(path, rows):
    rows = list(rows)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('x', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(rows)


def verify(files):
    for name, expected in files.items():
        if sha(name) != expected:
            raise ValueError('Source file changed: ' + name)


def decision_id(row):
    blocks = []
    for b in row['decision']:
        allocations = sorted([(tuple(a['nodes']), tuple(a['modes']), float(a['share']))
                              for a in b['allocations']])
        blocks.append((b['origin'], b['destination'], int(b['batch_id']), allocations))
    return canonical(sorted(blocks))


def objectives(row):
    return np.asarray([row['optimisation_objectives'][k] for k in OBJECTIVES], dtype=float)


def select_plans(front):
    if not front or any(not r['training_feasible'] for r in front):
        raise ValueError('Missing or infeasible final front')
    values = np.asarray([objectives(r) for r in front])
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite own-model objectives')
    # Own-model objectives only. IDs break residual exact ties deterministically.
    cost = min(range(len(front)), key=lambda i: (values[i, 0], values[i, 2],
                                                values[i, 1], decision_id(front[i])))
    lo, hi = values.min(0), values.max(0)
    scaled = (values - lo) / np.where(hi > lo, hi - lo, 1.)
    distances = np.linalg.norm(scaled, axis=1)
    compromise = min(range(len(front)), key=lambda i: (distances[i], values[i, 0],
                            values[i, 2], values[i, 1], decision_id(front[i])))
    return {'minimum_cost': cost, 'compromise': compromise}, dict(
        minimum=lo.tolist(), maximum=hi.tolist(), equal_objective_weights=True,
        compromise_distance=distances.tolist(), zero_span_contribution=0,
        residual_tie_break='cost, makespan, emission, exact decision ID')


def save_scenarios(stem, scenario, identity):
    """Save actual network draws, not plan objective samples or only a seed."""
    travel_keys = sorted(scenario.travel_multiplier)
    border_keys = sorted(scenario.border_delay_h)
    path = stem.with_suffix('.npz')
    with path.open('xb') as f:
        np.savez_compressed(f,
            travel=np.asarray([scenario.travel_multiplier[k] for k in travel_keys]).reshape(-1, scenario.size),
            border=np.asarray([scenario.border_delay_h[k] for k in border_keys]).reshape(-1, scenario.size))
    meta = dict(format='section68-network-scenarios-v1', identity=identity,
        size=scenario.size, seed=scenario.seed, stochastic=scenario.stochastic,
        travel_keys=travel_keys, border_keys=border_keys,
        arc_border_event=[[k, v] for k, v in sorted(scenario.arc_border_event.items())],
        border_event_mean_h=[[k, v] for k, v in sorted(scenario.border_event_mean_h.items())],
        data_sha256=sha(path), scenario_digest=scenario_digest(scenario))
    save(stem.with_suffix('.json'), meta)
    return meta


def load_scenarios(stem):
    m = read(stem.with_suffix('.json'))
    if m['format'] != 'section68-network-scenarios-v1' or sha(stem.with_suffix('.npz')) != m['data_sha256']:
        raise ValueError('Scenario pack changed: ' + str(stem))
    with np.load(stem.with_suffix('.npz'), allow_pickle=False) as p:
        travel, border = p['travel'].copy(), p['border'].copy()
    if travel.shape != (len(m['travel_keys']), m['size']) or border.shape != (len(m['border_keys']), m['size']):
        raise ValueError('Scenario pack shape mismatch')
    if not np.isfinite(travel).all() or not np.isfinite(border).all():
        raise ValueError('Nonfinite network realizations')
    s = base.ScenarioSet(size=m['size'], seed=m['seed'], stochastic=m['stochastic'],
        travel_multiplier={tuple(k): a for k, a in zip(m['travel_keys'], travel)},
        border_delay_h={tuple(k): a for k, a in zip(m['border_keys'], border)},
        arc_border_event={tuple(k): tuple(v) for k, v in m['arc_border_event']},
        border_event_mean_h={tuple(k): v for k, v in m['border_event_mean_h']})
    if scenario_digest(s) != m['scenario_digest']:
        raise ValueError('Scenario digest mismatch')
    for a in (*s.travel_multiplier.values(), *s.border_delay_h.values()):
        a.flags.writeable = False
    return s, m


class Evaluator:
    def __init__(self, instance_path, uncertainty):
        self.entry, self.batches = read_instance(Path(instance_path))
        self.u = uncertainty
        base.BORDER_EVENT_DEFINITIONS = base.load_border_event_definitions(base.DEFAULT_BORDER_EVENT_DATA_FILE)
        (_, _, self.hold, self.proc, self.trans_cost, self.arcs, timetables, _,
         self.wait_cost, self.wait_emission, self.carbon, _, _, self.trans_map,
         _, self.theta, _) = base.load_network_from_extended(str(DATA))
        self.border = base.load_border_delay_map_from_workbook(DATA, include_customs=uncertainty['border_include_customs'])
        self.tt = base.build_timetable_dict(timetables)
        self.lookup = base.build_arc_lookup(self.arcs)
        self.paths = {}
        for b in self.batches:
            b.penalty_per_teu_h = base.DEFAULT_LATE_PENALTY_USD_PER_TEU_H
        base.ACTIVE_MUTATION_LOGGER = None

    def make_scenarios(self, count, seed):
        return base.build_scenario_set(self.arcs, self.border, count, seed, stochastic=True,
            border_event_definitions=base.BORDER_EVENT_DEFINITIONS,
            **{k: self.u[k] for k in ('mode_time_cv', 'mode_time_max_factor',
               'border_delay_cv', 'border_delay_max_factor', 'border_mean_source')})

    def ev_scenarios(self):
        return base.build_expected_value_scenario_set(self.arcs, self.border, seed=0,
            border_event_definitions=base.BORDER_EVENT_DEFINITIONS,
            border_mean_source=self.u['border_mean_source'])

    def activate(self, scenario, model='CCP'):
        self.scenario = scenario
        base.ACTIVE_SCENARIO_SET = scenario
        base._PATH_SCENARIO_CACHE = {}
        base.RISK_METRIC = 'deterministic' if model == 'EV' else 'ccp'
        base.CONFIDENCE_COST = base.CONFIDENCE_EMISSION = base.CONFIDENCE_TIME = (1. if model == 'WORST' else .9)

    def restore(self, row):
        ind = base.Individual()
        for block in row['decision']:
            key = (block['origin'], block['destination'], int(block['batch_id']))
            allocs = []
            for a in block['allocations']:
                pk = (tuple(a['nodes']), tuple(a['modes']))
                if pk not in self.paths:
                    p = base.rebuild_path_from_nodes_modes(block['origin'], block['destination'],
                        a['nodes'], a['modes'], self.tt, self.lookup, allow_road_fallback=False)
                    if p is None or (tuple(p.nodes), tuple(p.modes)) != pk:
                        raise ValueError('Cannot restore exact path')
                    self.paths[pk] = p
                allocs.append(base.PathAllocation(self.paths[pk], float(a['share'])))
            ind.od_allocations[key] = allocs
        if decision_signature(ind) != row['decision_fingerprint']:
            raise ValueError('Restored decision changed')
        return ind

    def evaluate(self, row, arrivals=False):
        if len(base._PATH_SCENARIO_CACHE) > 256:
            base._PATH_SCENARIO_CACHE = {}
        ind = self.restore(row)
        base.evaluate_individual(ind, self.batches, self.arcs, self.tt, self.wait_cost, self.wait_emission,
            node_hold_cost=self.hold, node_proc_cost=self.proc, carbon_tax_map=self.carbon,
            trans_map=self.trans_map, border_delay_map=self.border, theta_rm=self.theta,
            node_trans_cost=self.trans_cost)
        if decision_signature(ind) != row['decision_fingerprint'] or not ind.feasible:
            raise ValueError('Re-evaluation changed the decision or nominal feasibility')
        values = np.stack((ind.cost_s, ind.emission_s, ind.makespan_s))
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite scenario objectives')
        arrival_values = None
        if arrivals:
            arrival_values = np.empty((len(self.batches), self.scenario.size))
            for i, b in enumerate(self.batches):
                allocs = [a for a in ind.od_allocations[(b.origin, b.destination, b.batch_id)] if a.share > 1e-12]
                a = np.full(self.scenario.size, b.ET, dtype=float)
                for alloc in allocs:
                    r = base.simulate_path_over_scenarios(alloc.path, b, self.tt, self.trans_map, self.border, self.scenario)
                    if r.missing_timetable:
                        raise ValueError('Missing timetable during arrival export')
                    a = np.maximum(a, r.arrival_h)
                arrival_values[i] = a
                if float(np.mean(a <= b.LT)) != ind.batch_on_time_prob[b.batch_id]:
                    raise ValueError('Whole-batch punctuality mismatch')
            if not np.array_equal(arrival_values.max(0), values[2]):
                raise ValueError('Batch arrivals do not reproduce makespan')
        return ind, values, arrival_values

    def route_metrics(self, row):
        ind = self.restore(row)
        # Visit each rail border node once per allocation, even on a repeated path.
        rail_nodes = sorted({a.from_node for a in self.arcs if a.mode == 'rail' and a.is_border_arc}
                            | {a.to_node for a in self.arcs if a.mode == 'rail' and a.is_border_arc})
        rail = {n: 0. for n in rail_nodes}
        sea = {(a.from_node, a.to_node): 0. for a in self.arcs if a.mode == 'water'}
        modes = {m: 0. for m in ('road', 'rail', 'water')}
        counts = []
        for b in self.batches:
            allocs = [a for a in ind.od_allocations[(b.origin, b.destination, b.batch_id)] if a.share > 1e-12]
            counts.append(len({(tuple(a.path.nodes), tuple(a.path.modes)) for a in allocs}))
            for alloc in allocs:
                flow = float(b.quantity * alloc.share)
                visited = set()
                for arc in alloc.path.arcs:
                    modes[arc.mode] += flow * arc.distance
                    if arc.mode == 'water':
                        sea[(arc.from_node, arc.to_node)] += flow
                    if arc.mode == 'rail' and arc.is_border_arc:
                        visited.update((arc.from_node, arc.to_node))
                for n in visited:
                    rail[n] += flow
        denominator = sum(modes.values())
        rows = [dict(metric='rail_border_TEU', item=k, value=v) for k, v in rail.items()]
        rows += [dict(metric='sea_service_TEU', item=k[0]+' -> '+k[1], value=v) for k, v in sorted(sea.items())]
        rows += [dict(metric='mode_TEU_km_share', item=k, value=v/denominator) for k, v in modes.items()]
        rows += [dict(metric='split_batch_fraction', item='all', value=float(np.mean(np.asarray(counts) >= 2))),
                 dict(metric='mean_paths_per_batch', item='all', value=float(np.mean(counts)))]
        return rows


def search(instance, settings, task, out, plan_sha, model, expected_options_sha):
    """Only explicit run arguments change; all patches are process-local/restored."""
    spec = importlib.util.spec_from_file_location('section68_frozen_search', EXP/'component_joint_b5000_v1/search_b5000.py')
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    original_episode = mod.run_episode
    original_options = base.build_reliable_path_options
    checked = []

    def inspect_options(*args, **kwargs):
        options = original_options(*args, **kwargs)
        description = option_description(options)
        got = canonical(description)
        if got != expected_options_sha:
            diagnostic = out/'NOMINAL_PATH_MISMATCH.json'
            save(diagnostic, dict(expected_sha256=expected_options_sha, actual_sha256=got,
                border_event_definition_count=len(base.BORDER_EVENT_DEFINITIONS),
                actual_options=description))
            raise ValueError('Search nominal path set differs from preflight; see '+str(diagnostic))
        checked.append(got)
        return options

    def episode(*args, **kwargs):
        if model == 'EV':
            kwargs['evaluation_mode'] = 'deterministic'
        elif model == 'WORST':
            kwargs['ccp_confidence'] = 1.
        elif model != 'CCP':
            raise ValueError('Unknown objective model')
        return original_episode(*args, **kwargs)

    mod.run_episode = episode
    base.build_reliable_path_options = inspect_options
    try:
        result = mod.run_search(instance, settings, task, out, plan_sha)
    finally:
        mod.run_episode = original_episode
        base.build_reliable_path_options = original_options
    if not checked:
        raise ValueError('Search did not check nominal options')
    return result, checked
