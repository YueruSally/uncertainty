"""Restore decisions verbatim and evaluate the unchanged CCP evaluator."""
import os
import sys
import time
import json
from pathlib import Path
import numpy as np
from common import (DATA, ENGINE, OBJECTIVES, canonical, digest, read, verify, write)

sys.path.insert(0, str(ENGINE/'support'))
sys.path.insert(0, str(ENGINE))
import baseline_uncertainty as base
from run_ccp_candidate_pool import decision_signature
from run_ev_ccp_oos_pilot import scenario_digest
from transformer_ppo.instance_catalog import read_instance

if Path(base.__file__).resolve() != ENGINE/'baseline_uncertainty.py':
    raise RuntimeError('Wrong evaluator module loaded; frozen engine required')


class Evaluator:
    def __init__(self, entry_path, config):
        self.entry, self.batches = read_instance(entry_path)
        self.uncertainty = config['uncertainty']
        base.BORDER_EVENT_DEFINITIONS = base.load_border_event_definitions(base.DEFAULT_BORDER_EVENT_DATA_FILE)
        (_, _, self.hold, self.proc, self.trans_cost, self.arcs, timetables, _,
         self.wait_cost, self.wait_emission, self.carbon, _, _, self.trans_map,
         border, self.theta, _) = base.load_network_from_extended(str(DATA))
        self.border = (base.load_border_delay_map_from_workbook(DATA,
            include_customs=self.uncertainty['border_include_customs'])
            if self.uncertainty['border_mean_source'] == 'node_border' else border)
        self.tt = base.build_timetable_dict(timetables)
        self.arcs_lookup = base.build_arc_lookup(self.arcs)
        self.paths = {}
        for b in self.batches:
            b.penalty_per_teu_h = base.DEFAULT_LATE_PENALTY_USD_PER_TEU_H
        base.ACTIVE_MUTATION_LOGGER = None
        base.RISK_METRIC = 'ccp'
        base.CONFIDENCE_COST = base.CONFIDENCE_EMISSION = base.CONFIDENCE_TIME = .9

    def scenarios(self, count, seed):
        u = self.uncertainty
        scenario = base.build_scenario_set(self.arcs, self.border, count, seed, stochastic=True,
            border_event_definitions=base.BORDER_EVENT_DEFINITIONS,
            mode_time_cv=u['mode_time_cv'], mode_time_max_factor=u['mode_time_max_factor'],
            border_delay_cv=u['border_delay_cv'], border_delay_max_factor=u['border_delay_max_factor'],
            border_mean_source=u['border_mean_source'])
        base.ACTIVE_SCENARIO_SET = scenario
        base._PATH_SCENARIO_CACHE = {}
        return scenario_digest(scenario)

    def restore(self, row):
        ind = base.Individual()
        for block in row['decision']:
            key = (block['origin'], block['destination'], int(block['batch_id']))
            restored = []
            for a in block['allocations']:
                pk = (tuple(a['nodes']), tuple(a['modes']))
                if pk not in self.paths:
                    path = base.rebuild_path_from_nodes_modes(block['origin'], block['destination'],
                        a['nodes'], a['modes'], self.tt, self.arcs_lookup, allow_road_fallback=False)
                    if path is None or (tuple(path.nodes), tuple(path.modes)) != pk:
                        raise ValueError('Could not restore exact saved path')
                    self.paths[pk] = path
                restored.append(base.PathAllocation(self.paths[pk], float(a['share'])))
            ind.od_allocations[key] = restored
        if decision_signature(ind) != row['decision_fingerprint']:
            raise ValueError('Restored decision changed')
        return ind

    def evaluate(self, row):
        # Bound cache size; eviction changes runtime only, never scenarios.
        if len(base._PATH_SCENARIO_CACHE) > 256:
            base._PATH_SCENARIO_CACHE = {}
        ind = self.restore(row)
        base.evaluate_individual(ind, self.batches, self.arcs, self.tt, self.wait_cost, self.wait_emission,
            node_hold_cost=self.hold, node_proc_cost=self.proc, carbon_tax_map=self.carbon,
            trans_map=self.trans_map, border_delay_map=self.border, theta_rm=self.theta,
            node_trans_cost=self.trans_cost)
        if decision_signature(ind) != row['decision_fingerprint']:
            raise ValueError('Evaluation altered a saved decision')
        samples = np.stack((ind.cost_s, ind.emission_s, ind.makespan_s))
        if not np.isfinite(samples).all():
            raise ValueError('Nonfinite objective realization; no plan will be silently dropped')
        return samples, np.asarray(ind.objectives), bool(ind.feasible), float(ind.normalized_violation)


def pack_specs(inputs):
    return [('original_100', 100, inputs['instance']['ccp100_seed']),
            ('reference_1000', 1000, inputs['new_seeds'][0])] + [
        (f'replicate_{i:02d}_100', 100, seed) for i, seed in enumerate(inputs['new_seeds'][1:], 1)]


def load_pack(folder, tag, identity):
    meta_path = folder/'packs'/(tag+'.json')
    data_path = folder/'packs'/(tag+'.npz')
    if not meta_path.exists():
        return None
    meta = read(meta_path)
    if not meta.get('completed') or meta['identity'] != identity or digest(data_path) != meta['data_sha256']:
        raise ValueError('Completed evaluation pack changed: ' + str(meta_path))
    return meta


def evaluate_packs(folder, inputs, protocol_sha, heartbeat):
    evaluator = None
    metas = []
    total_plans = inputs['unique_plans']
    for pi, (tag, count, seed) in enumerate(pack_specs(inputs)):
        identity = dict(inputs_sha256=inputs['sha256'], audit_plan_sha256=protocol_sha,
                        tag=tag, realizations=count, scenario_seed=seed)
        old = load_pack(folder, tag, identity)
        if old is not None:
            metas.append(old)
            continue
        verify(inputs['files'])
        if evaluator is None:
            evaluator = Evaluator(folder/'instance.json', inputs['source_config'])
        start = time.monotonic()
        scenario_sha = evaluator.scenarios(count, seed)
        if tag == 'original_100' and scenario_sha != inputs['scenario_digest']:
            raise ValueError('Original 100 scenario realizations do not reproduce')
        samples = np.empty((total_plans, 3, count), dtype=np.float64)
        quantiles = np.empty((total_plans, 3), dtype=np.float64)
        feasible = np.empty(total_plans, dtype=bool)
        violation = np.empty(total_plans, dtype=float)
        last = 0.
        for i, row in enumerate(inputs['decisions']):
            samples[i], quantiles[i], feasible[i], violation[i] = evaluator.evaluate(row)
            # Append after every physical evaluation, including repeats of an
            # interrupted pack. Existing source costs are never touched.
            with (folder/'EVALUATION_LEDGER.jsonl').open('a', encoding='utf-8') as ledger:
                ledger.write(json.dumps(dict(pack=tag, decision=row['exact_decision_id'],
                    realizations=count, scenario_seed=seed, evaluated_at=time.time()))+'\n')
            if tag == 'original_100':
                original = [row['optimisation_objectives'][obj] for obj in OBJECTIVES]
                if not feasible[i] or not np.allclose(quantiles[i], original, rtol=1e-10, atol=1e-8):
                    write(folder/'REPRODUCTION_FAILURE.json', dict(decision=row['exact_decision_id'],
                        original=original, recomputed=quantiles[i].tolist(), feasible=bool(feasible[i])))
                    raise ValueError('Original evaluation mismatch: inspect REPRODUCTION_FAILURE.json')
            now = time.monotonic()
            if i == 0 or i+1 == total_plans or now-last >= 20:
                elapsed = now-start
                heartbeat(stage='evaluating', configuration=inputs['configuration'], pack=tag,
                    completed_packs=pi, plans_done=i+1, plans_total=total_plans,
                    pack_elapsed_seconds=round(elapsed, 2),
                    pack_remaining_seconds=round(elapsed/(i+1)*(total_plans-i-1), 1))
                last = now
        verify(inputs['files'])
        dest = folder/'packs'/(tag+'.npz'); dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name+'.writing')
        with tmp.open('wb') as f:
            np.savez_compressed(f, samples=samples, quantiles=quantiles, feasible=feasible, violation=violation)
            f.flush(); os.fsync(f.fileno())
        tmp.replace(dest)
        meta = dict(completed=True, identity=identity, data_sha256=digest(dest), scenario_digest=scenario_sha,
            plans=total_plans, feasible_plans=int(feasible.sum()), seconds=time.monotonic()-start,
            fixed_plan_evaluations=total_plans, plan_realizations=total_plans*count)
        if tag == 'original_100':
            expected = np.asarray([[r['optimisation_objectives'][o] for o in OBJECTIVES] for r in inputs['decisions']])
            meta.update(reproduced=True, maximum_absolute_error=float(np.abs(quantiles-expected).max()),
                        maximum_relative_error=float((np.abs(quantiles-expected)/np.maximum(np.abs(expected),1e-300)).max()))
            write(folder/'REPRODUCTION.json', meta)
        write(folder/'packs'/(tag+'.json'), meta)
        metas.append(meta)
        heartbeat(stage='pack_complete', configuration=inputs['configuration'], pack=tag,
                  completed_packs=pi+1, total_packs=22, pack_seconds=round(meta['seconds'], 2))
    return metas
