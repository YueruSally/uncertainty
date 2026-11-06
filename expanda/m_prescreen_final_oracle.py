"""Final-HV audit policies for the frozen K8 Replace candidate pool.

The oracle is deliberately diagnostic: at each supported mutation event it
evaluates every constructed child with the frozen CCP100 scenarios, selects
the largest one-step population-HV contribution, and charges the selected
evaluation to B400 while reporting all other evaluations separately.  The
simple scorer never evaluates an unselected child.
"""
from collections import Counter
from copy import deepcopy
import time

import numpy as np

import baseline_uncertainty as base
from m_candidate_information import SPECIFIC_NAMES
from m_prescreen_k8 import (CollectBalancedPolicy, CollectK8Controller,
                            build_replace_pool)
from multi_instance_logging import fingerprint
from multi_instance_policy import Decision
from transformer_ppo.archive import exact_hv_3d
from transformer_ppo.controller import PPOController


METHODS = ("BALANCED", "K8_SIMPLE", "K8_ORACLE")


def simple_nominal_score(proposal):
    """Evaluation-free equal-weight score for local nominal objective deltas."""
    features = np.asarray(proposal["features"], dtype=float)
    names = ("delta_cost", "delta_emission", "delta_nominal_time")
    indices = [SPECIFIC_NAMES.index(name) for name in names]
    if features.shape != (len(SPECIFIC_NAMES),) or not np.isfinite(features).all():
        raise ValueError("invalid K8 proposal features")
    # Features use one common pre-action scale per objective.  Lower is better.
    return -float(np.mean(features[indices]))


class FinalOraclePolicy(CollectBalancedPolicy):
    """Frozen audit method identity; selection is implemented by the controller."""
    def __init__(self, instance, seed, method):
        if method not in METHODS:
            raise ValueError(f"unknown K8 final-HV audit method: {method}")
        super().__init__(instance, seed)
        self.audit_method = method
        self.collect = False
        self.name = f"prescreen-k8-final-hv-{method.lower()}"
        self.action_counts = Counter()
        self.switches = 0


class FinalOracleController(CollectK8Controller):
    """Execute Balanced, a cheap nominal scorer, or the greedy true-HV oracle."""
    def __init__(self, *args, **kwargs):
        self.audit_method = kwargs["trainer"].audit_method
        self.pool_events = 0
        self.unsupported_events = 0
        self.pool_candidates = 0
        self.oracle_pool_objective_evaluations = 0
        self.oracle_additional_objective_evaluations = 0
        self.oracle_additional_evaluation_seconds = 0.0
        self.oracle_selected_evaluation_seconds = 0.0
        self.selection_counts = Counter()
        self._audit_evaluation_context = None
        self._oracle_evaluated = None
        self._oracle_selected_index = None
        self._oracle_selected_seconds_current = 0.0
        super().__init__(*args, **kwargs)

    def _mutate_operator_only(
            self, ind, batches, path_lib, tt_dict, arc_lookup, arcs,
            waiting_cost, waiting_emission, reliable_options,
            mixture_event_index, evaluation_kwargs):
        self._audit_evaluation_context = (batches, arcs, tt_dict,
            waiting_cost, waiting_emission, evaluation_kwargs)
        try:
            # Skip CollectK8Controller's label collector.  This audit chooses
            # the search child itself and never saves a training panel.
            return PPOController._mutate_operator_only(
                self, ind, batches, path_lib, tt_dict, arc_lookup, arcs,
                waiting_cost, waiting_emission, reliable_options,
                mixture_event_index, evaluation_kwargs)
        finally:
            self._audit_evaluation_context = None
            self._oracle_evaluated = None
            self._oracle_selected_index = None

    def _evaluate_oracle_pool(self, population_points):
        if self._audit_evaluation_context is None:
            raise RuntimeError("oracle pool requested outside mutation")
        batches, arcs, tt_dict, waiting_cost, waiting_emission, kwargs = \
            self._audit_evaluation_context
        hv_before = exact_hv_3d(population_points,
                                self.archive.normalizer.reference)
        evaluated, scores, seconds = [], [], []
        for proposal in self.proposals:
            child = deepcopy(proposal["child"])
            started = time.perf_counter()
            base._evaluate_individual_impl(
                child, batches, arcs, tt_dict, waiting_cost,
                waiting_emission, **kwargs)
            seconds.append(time.perf_counter() - started)
            point = ([self.archive.normalizer.transform(child.objectives)]
                     if child.feasible else [])
            scores.append(exact_hv_3d(
                population_points + point,
                self.archive.normalizer.reference) - hv_before)
            evaluated.append(child)
        self.oracle_pool_objective_evaluations += len(evaluated)
        return evaluated, scores, seconds

    def _choose_operator(self, ind, batches, path_lib, tt_dict, arc_lookup,
                         reliable_options):
        started = time.perf_counter()
        self.proposal_parent = fingerprint(ind)
        self.proposals = build_replace_pool(
            self, ind, batches, path_lib, tt_dict, arc_lookup,
            reliable_options)
        if not self.proposals or not self.proposals[0]["eligible"]:
            self.unsupported_events += 1
            self.selected_proposal = None
            candidates = [{"candidate_operator": "replace",
                "eligible": False}]
            return Decision(None, [0.0], [0.0], {
                "decision_kind": "k8_final_hv_audit_unsupported",
                "audit_method": self.audit_method,
                "pool_size": 0,
                "switched": False,
                "inference_seconds": 0.0,
            }), candidates
        self.pool_events += 1
        self.pool_candidates += len(self.proposals)
        method = self.audit_method
        scores = [0.0] * len(self.proposals)
        if method == "BALANCED":
            selected = 0
        elif method == "K8_SIMPLE":
            scores = [simple_nominal_score(row) for row in self.proposals]
            selected = max(range(len(scores)), key=lambda i: (scores[i], -i))
        else:
            points = [list(self.archive.normalizer.transform(p.objectives))
                      for p in self.current_population if p.feasible]
            evaluated, scores, seconds = self._evaluate_oracle_pool(points)
            selected = max(range(len(scores)), key=lambda i: (scores[i], -i))
            self._oracle_evaluated = evaluated
            self._oracle_selected_index = selected
            # The selected result is the one B400 evaluation.  The remaining
            # pool members are diagnostic evaluations outside that budget.
            self.oracle_selected_evaluation_seconds += seconds[selected]
            self._oracle_selected_seconds_current = seconds[selected]
            self.oracle_additional_objective_evaluations += len(scores) - 1
            self.oracle_additional_evaluation_seconds += (
                sum(seconds) - seconds[selected])
        self.selected_proposal = selected
        self.selection_counts[str(selected)] += 1
        self.trainer.action_counts[str(selected)] += 1
        self.trainer.switches += int(selected != 0)
        self.proposal_seconds += time.perf_counter() - started
        candidates = [{"candidate_operator": "replace",
            "eligible": bool(row["eligible"])} for row in self.proposals]
        probabilities = [float(i == selected) for i in range(len(scores))]
        return Decision(selected, probabilities, list(scores), {
            "decision_kind": "k8_final_hv_audit",
            "audit_method": method,
            "pool_size": len(scores),
            "switched": selected != 0,
            "inference_seconds": 0.0,
        }), candidates

    def _execute_operator_action(self, ind, operator, batches, path_lib,
                                 tt_dict, arc_lookup, reliable_options,
                                 target_rng=None, trace=None):
        result = super()._execute_operator_action(
            ind, operator, batches, path_lib, tt_dict, arc_lookup,
            reliable_options, target_rng=target_rng, trace=trace)
        if self.audit_method != "K8_ORACLE":
            return result
        if self._oracle_evaluated is None:
            raise RuntimeError("oracle selected without evaluated K8 pool")
        if not self.has_budget(1):
            raise RuntimeError("objective evaluation budget exhausted")
        evaluated = self._oracle_evaluated[self._oracle_selected_index]
        # Adopt both the frozen decision and its already-computed CCP100 result.
        # The following base.evaluate_individual call becomes a verified cache
        # hit; this manual increment charges exactly one selected child to B400.
        for name, value in evaluated.__dict__.items():
            setattr(ind, name, deepcopy(value))
        identity = fingerprint(ind)
        self.evaluations += 1
        self.objective_evaluation_seconds += self._oracle_selected_seconds_current
        ind._learning_eval_key = (self.scenario_id, identity)
        ind._learning_eval_id = self.evaluations
        return result

    def summary(self):
        selected = sum(self.selection_counts.values())
        return {**super().summary(),
            "k8_final_hv_audit": {
                "method": self.audit_method,
                "pool_events": self.pool_events,
                "unsupported_events": self.unsupported_events,
                "pool_candidates": self.pool_candidates,
                "mean_pool_size": (self.pool_candidates / self.pool_events
                    if self.pool_events else 0.0),
                "selection_counts": dict(self.selection_counts),
                "switches": selected - self.selection_counts.get("0", 0),
                "oracle_pool_objective_evaluations":
                    self.oracle_pool_objective_evaluations,
                "oracle_additional_objective_evaluations":
                    self.oracle_additional_objective_evaluations,
                "oracle_additional_evaluation_seconds":
                    self.oracle_additional_evaluation_seconds,
                "selected_objective_evaluations_charged_to_b400":
                    self.evaluations,
                "total_physical_objective_evaluations":
                    self.evaluations +
                    self.oracle_additional_objective_evaluations,
            }}
