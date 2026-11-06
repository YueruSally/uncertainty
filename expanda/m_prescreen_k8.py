"""Frozen eight-child Replace pool for the M prescreen experiment.

Constructing a child never evaluates its CCP objectives. Candidate zero is
the exact Balanced proposal used by the concrete-candidate experiment.
"""
from copy import deepcopy
from dataclasses import asdict
import math
import time

import numpy as np

import baseline_uncertainty as base
from m_candidate_information import (CandidateController, build_proposals,
    private_seed, proposal_features, proposal_randomness)
from m_operator_fixed_mix import FixedMixPolicy
from multi_instance_logging import fingerprint
from multi_instance_policy import Decision
from transformer_ppo.archive import exact_hv_3d
from transformer_ppo.controller import PPOController
from transformer_ppo.observations import joint_balanced_candidate_scores

POOL_SIZE = 8
MAX_STATES = 32


def batch_tokens(child, batches):
    """All batch allocations are represented before any objective evaluation."""
    from m_candidate_information import allocation_metrics
    total_quantity = max(sum(float(b.quantity) for b in batches), 1e-8)
    tokens = []
    for batch in batches:
        key = (batch.origin, batch.destination, batch.batch_id)
        features = allocation_metrics(child.od_allocations.get(key, []), batch.quantity)
        features = np.concatenate((
            [float(batch.quantity)/total_quantity,
             float(batch.ET)/1000., float(batch.LT)/1000.], features))
        if not np.isfinite(features).all():
            raise ValueError('non-finite candidate batch token')
        tokens.append(features.tolist())
    return tokens


def build_replace_pool(controller, parent, batches, path_lib, tt_dict,
                       arc_lookup, reliable_options):
    """Exact Balanced child plus up to seven distinct legal Replace children."""
    operator_candidates = controller._operator_candidates(
        parent, batches, path_lib, tt_dict, arc_lookup, reliable_options)
    balanced = build_proposals(
        controller, parent, batches, path_lib, tt_dict, arc_lookup,
        reliable_options, operator_candidates)[4]
    balanced = {**balanced, 'eligible': bool(operator_candidates[4]['eligible']
                and balanced['ok']), 'source': 'balanced'}
    pool = [balanced]
    seen = {balanced['fingerprint']}
    if balanced['eligible']:
        rows = controller._operator_target_rows(parent, batches, 'replace',
            path_lib, tt_dict, arc_lookup, reliable_options)
        eligible = [bool(row.get('eligible')) for row in rows]
        scores = joint_balanced_candidate_scores(rows, eligible)
        order = sorted((i for i, legal in enumerate(eligible) if legal),
                       key=lambda i: (float(scores[i]),
                           int(rows[i]['target'].batch_id),
                           str(rows[i].get('path_id'))))
        batch_by_id = {b.batch_id: b for b in batches}
        for index in order:
            if len(pool) == POOL_SIZE:
                break
            row = rows[index]
            batch = batch_by_id[row['target'].batch_id]
            child = deepcopy(parent)
            seed = private_seed('prescreen-k8-v1', controller.run_id,
                controller.seeds['algorithm'], controller.generation,
                controller._active_mixture_event_index, index)
            with proposal_randomness(seed):
                ok = bool(base.apply_mutation_op(child, 'replace', batch,
                    path_lib, tt_dict, arc_lookup,
                    reliable_options=reliable_options, target=row['target']))
                if ok:
                    base.repair_missing_allocations(child, batches, path_lib)
            identity = fingerprint(child)
            if not ok or identity in seen:
                continue
            seen.add(identity)
            pool.append({'operator': 'replace', 'child': child, 'batch': batch,
                'target': row['target'], 'ok': ok, 'metadata': {'selector': 'prescreen-k8'},
                'seed': seed, 'fingerprint': identity, 'eligible': True,
                'source': 'alternative',
                'features': proposal_features(parent, child, batch, row['target'],
                    batches, ok)})
    return pool


class CollectBalancedPolicy(FixedMixPolicy):
    """The collector searches with Balanced; labels never alter decisions."""
    def __init__(self, instance, seed):
        super().__init__(instance, seed, 0.)
        self.name = 'prescreen-k8-collect-balanced'
        self.collect = True

    def choose(self, candidates, context=None, **kwargs):
        if not candidates or len(candidates) > POOL_SIZE:
            raise ValueError('invalid Replace pool size')
        chosen = 0 if candidates[0]['eligible'] else None
        probabilities = [float(i == chosen) for i in range(len(candidates))]
        return Decision(chosen, probabilities, [None]*len(candidates),
            {'decision_kind': 'prescreen_collect_balanced', 'inference_seconds': 0.})


class CollectK8Controller(CandidateController):
    """Evaluate alternatives only for saved training labels, outside B400."""
    def _choose_operator(self, ind, batches, path_lib, tt_dict, arc_lookup,
                         reliable_options):
        started = time.perf_counter()
        self.proposal_parent = fingerprint(ind)
        self.proposals = build_replace_pool(self, ind, batches, path_lib,
            tt_dict, arc_lookup, reliable_options)
        candidates = [{'candidate_operator': 'replace',
            'eligible': bool(row['eligible'])} for row in self.proposals]
        self.proposal_seconds += time.perf_counter()-started
        decision = self.trainer.choose(candidates)
        self.selected_proposal = decision.chosen_index
        self.feedback_pending = None
        if (decision.chosen_index == 0 and len(self.proposals) > 1
                and len(self.feedback_rows) < MAX_STATES):
            points = [list(self.archive.normalizer.transform(p.objectives))
                for p in self.current_population if p.feasible]
            self.feedback_pending = {
                'instance': self.run_id, 'algorithm_seed': self.seeds['algorithm'],
                'generation': self.generation,
                'evaluation': self.evaluations,
                'event': self._active_mixture_event_index,
                'parent_fingerprint': self.proposal_parent,
                'population_points': points,
                'candidate_features': [p['features'].tolist() for p in self.proposals],
                'candidate_batch_tokens': [batch_tokens(p['child'], batches)
                    for p in self.proposals],
                'candidate_fingerprints': [p['fingerprint'] for p in self.proposals],
                'candidate_sources': [p['source'] for p in self.proposals],
            }
        return decision, candidates

    def _execute_operator_action(self, ind, operator, batches, path_lib,
                                 tt_dict, arc_lookup, reliable_options,
                                 target_rng=None, trace=None):
        if operator != 'replace' or fingerprint(ind) != self.proposal_parent:
            raise RuntimeError('Replace pool execution contract changed')
        proposal = self.proposals[self.selected_proposal]
        ind.od_allocations = deepcopy(proposal['child'].od_allocations)
        if fingerprint(ind) != proposal['fingerprint']:
            raise RuntimeError('selected K8 child was not executed exactly')
        self.execution_checks += 1
        return (proposal['ok'], proposal['batch'], proposal['target'],
            {**proposal['metadata'], 'proposal_seed': proposal['seed'],
             'proposal_exact': True})

    def _mutate_operator_only(self, ind, batches, path_lib, tt_dict, arc_lookup,
                              arcs, waiting_cost, waiting_emission,
                              reliable_options, mixture_event_index,
                              evaluation_kwargs):
        result = PPOController._mutate_operator_only(self, ind, batches,
            path_lib, tt_dict, arc_lookup, arcs, waiting_cost,
            waiting_emission, reliable_options, mixture_event_index,
            evaluation_kwargs)
        row = self.feedback_pending
        if row is None:
            return result
        hv_before = exact_hv_3d(row['population_points'],
            self.archive.normalizer.reference)
        labels = []
        for index, proposal in enumerate(self.proposals):
            if index == 0:
                child = ind
            else:
                child = deepcopy(proposal['child'])
                base._evaluate_individual_impl(child, batches, arcs,
                    tt_dict, waiting_cost, waiting_emission,
                    **evaluation_kwargs)
                self.feedback_evaluations += 1
            point = ([self.archive.normalizer.transform(child.objectives)]
                     if child.feasible else [])
            objectives = [float(x) for x in child.objectives]
            if not all(math.isfinite(x) for x in objectives):
                if child.feasible:
                    raise ValueError('feasible K8 child has non-finite objectives')
                objectives = None
            labels.append({'index': index, 'objectives': objectives,
                'feasible': bool(child.feasible),
                'hv_contribution': exact_hv_3d(row['population_points']+point,
                    self.archive.normalizer.reference)-hv_before,
                'fingerprint': proposal['fingerprint'],
                'source': proposal['source'],
                'target': (asdict(proposal['target'])
                    if proposal['target'] is not None else None)})
        row['labels'] = labels
        row['additional_objective_evaluations'] = len(labels)-1
        self.feedback_rows.append(row)
        self.feedback_pending = None
        print(f'[M-K8-DATA] {self.run_id} states={len(self.feedback_rows)}/{MAX_STATES} '
              f'extra_evaluations={self.feedback_evaluations}', flush=True)
        return result

    def summary(self):
        return {**super().summary(), 'candidate_contract': 'prescreen-k8-v1'}
