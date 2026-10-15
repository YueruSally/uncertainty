"""Counterfactual position probes using a frozen pre-action child and archive."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import json
import random
import time

import numpy as np

import baseline_uncertainty as base
from learning_mutation import repair_after_mutation
from multi_instance_logging import fingerprint
from multi_instance_mutation import candidates_for
from multi_instance_policy import RulePolicy

from .archive import RewardArchive, mutation_reward
from .controller import PPOController
from .rf_reward import FEATURE_NAMES, feature_values


class DiagnosticRuleTrainer:
    name = "rule-position-diagnostic"
    training = False
    updates = 0
    checkpoint_provenance = None

    def __init__(self, policy_seed):
        self.policy = RulePolicy(random.Random(int(policy_seed)))
        self.observation_provider = None

    def choose(self, candidates, context=None):
        return self.policy.choose(candidates, context)

    def observe_outcome(self, reward, done=False):
        return None

    def finish(self):
        return None


class PositionDiagnosticController(PPOController):
    def __init__(self, *args, diagnostic_seed=0, diagnostic_repeats=3,
                 diagnostic_max_events=8, diagnostic_max_candidates=8, **kwargs):
        super().__init__(*args, **kwargs)
        if diagnostic_repeats < 1 or diagnostic_max_events < 1 or diagnostic_max_candidates < 2:
            raise ValueError("diagnostic repeats/events must be positive and candidates >=2")
        self.diagnostic_seed = int(diagnostic_seed)
        self.diagnostic_repeats = int(diagnostic_repeats)
        self.diagnostic_max_events = int(diagnostic_max_events)
        self.diagnostic_max_candidates = int(diagnostic_max_candidates)
        self.diagnostic_events = 0
        self.diagnostic_evaluations = 0
        self.diagnostic_rows = 0
        self.diagnostic_handle = (self.out / "position_diagnostics.jsonl").open(
            "x", encoding="utf-8")

    def _archive_fork(self):
        archive = RewardArchive(self.archive.normalizer)
        archive._points = list(self.archive._points)
        archive._fingerprints = set(self.archive._fingerprints)
        return archive

    def _context(self, before, operator, candidate_count):
        return {
            "operator": operator,
            "generation": self.generation,
            "phase": self.phase,
            "K": self.instance.get("K"),
            "deadline_window_alpha": self.instance.get("deadline_window_alpha"),
            "feasible_before": bool(before.feasible),
            "violation_before": before.normalized_violation,
            "violation_breakdown_before": before.vio_breakdown,
            "candidate_count": candidate_count,
            "q90_cost_before": before.objectives[0],
            "q90_emission_before": before.objectives[1],
            "q90_makespan_before": before.objectives[2],
        }

    def _probe(self, ind, operator, candidates, batches, path_lib, tt_dict,
               arc_lookup, arcs, waiting_cost, waiting_emission,
               reliable_options, eval_kwargs):
        eligible = [i for i, candidate in enumerate(candidates) if candidate["eligible"]]
        if len(eligible) < 2 or self.diagnostic_events >= self.diagnostic_max_events:
            return
        event_number = self.diagnostic_events + 1
        chooser = random.Random(self.diagnostic_seed + event_number * 1_000_003)
        selected = (eligible if len(eligible) <= self.diagnostic_max_candidates else
                    sorted(chooser.sample(eligible, self.diagnostic_max_candidates)))
        inclusion_probability = min(1.0, self.diagnostic_max_candidates / len(eligible))
        context = self._context(ind, operator, len(candidates))
        self.current_individual = ind
        try:
            observation = self._observation(candidates, context)
        finally:
            self.current_individual = None
        event_id = f"{self.run_id}:diagnostic:{event_number}"
        before_hash = fingerprint(ind)
        for repeat in range(self.diagnostic_repeats):
            new_value_seed = self.diagnostic_seed + event_number * 100_000 + repeat
            for candidate_index in selected:
                random.seed(new_value_seed)
                np.random.seed(new_value_seed % (2**32 - 1))
                trial_before = deepcopy(ind)
                trial = deepcopy(ind)
                candidate = candidates[candidate_index]
                target = candidate["target"]
                batch = next(b for b in batches if b.batch_id == target.batch_id)
                started = time.perf_counter()
                ok = base.apply_mutation_op(
                    trial, operator, batch, path_lib, tt_dict, arc_lookup,
                    reliable_options=reliable_options, target=target)
                raw_hash = fingerprint(trial)
                repair = repair_after_mutation(
                    trial, trial_before, batches, path_lib, tt_dict, arc_lookup)
                after_hash = fingerprint(trial)
                base._evaluate_individual_impl(
                    trial, batches, arcs, tt_dict, waiting_cost, waiting_emission,
                    **eval_kwargs)
                self.diagnostic_evaluations += 1
                fork = self._archive_fork()
                hv_before = fork.hypervolume
                fork.observe(trial.objectives, bool(trial.feasible), after_hash)
                delta_hv = max(0.0, fork.hypervolume - hv_before)
                outcome = {
                    **repair,
                    "decision_changed": after_hash != before_hash,
                    "feasible_after": bool(trial.feasible),
                    "violation_before": float(ind.normalized_violation),
                    "violation_after": float(trial.normalized_violation),
                }
                reward = mutation_reward(outcome, delta_hv, self.reward_config)
                row = {
                    "schema_version": 1,
                    "event_id": event_id,
                    "instance_id": self.instance.get("instance_id"),
                    "instance_digest": self.instance.get("instance_digest"),
                    "operator": operator,
                    "generation": self.generation,
                    "phase": self.phase,
                    "candidate_index": candidate_index,
                    "eligible_candidate_count": len(eligible),
                    "diagnosed_candidate_count": len(selected),
                    "diagnostic_inclusion_probability": inclusion_probability,
                    "repeat": repeat,
                    "new_value_seed": new_value_seed,
                    "target": asdict(target),
                    "feature_names": FEATURE_NAMES,
                    "feature_values": feature_values(observation, candidate_index),
                    "p_rule": float(candidate["p_rule"]),
                    "mutation_success": bool(ok),
                    "raw_mutation_changed": raw_hash != before_hash,
                    "decision_changed": after_hash != before_hash,
                    "effective_mutation": bool(ok and raw_hash != before_hash
                                               and after_hash != before_hash),
                    "repair_success": bool(repair["repair_success"]),
                    "mutation_reverted": bool(repair["mutation_reverted"]),
                    "feasible_before": bool(ind.feasible),
                    "feasible_after": bool(trial.feasible),
                    "reward": float(reward),
                    "reward_hv_delta": float(delta_hv),
                    "reward_archive_hv_before": float(hv_before),
                    "reward_archive_hv_after": float(fork.hypervolume),
                    "evaluation_seconds": time.perf_counter() - started,
                }
                self.diagnostic_handle.write(
                    json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
                self.diagnostic_handle.flush()
                self.diagnostic_rows += 1
        self.diagnostic_events += 1

    def mutate(self, ind, batches, path_lib, tt_dict, arc_lookup,
               arcs, waiting_cost, waiting_emission, reliable_options=None, **kwargs):
        # Evaluate and archive the post-crossover child exactly as the normal
        # controller does, while keeping it out of pending Q until mutation ends.
        self._inside_mutation = True
        try:
            base.evaluate_individual(
                ind, batches, arcs, tt_dict, waiting_cost, waiting_emission, **kwargs)
        finally:
            self._inside_mutation = False
        operator = base.sample_operator()
        candidates = candidates_for(
            ind, batches, operator, path_lib, tt_dict, arc_lookup, reliable_options)
        python_state, numpy_state = random.getstate(), np.random.get_state()
        try:
            self._probe(ind, operator, candidates, batches, path_lib, tt_dict,
                        arc_lookup, arcs, waiting_cost, waiting_emission,
                        reliable_options, kwargs)
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
        original = base.sample_operator
        base.sample_operator = lambda: operator
        try:
            return super().mutate(
                ind, batches, path_lib, tt_dict, arc_lookup, arcs,
                waiting_cost, waiting_emission,
                reliable_options=reliable_options, **kwargs)
        finally:
            base.sample_operator = original

    def summary(self):
        return {**super().summary(),
                "diagnostic_events": self.diagnostic_events,
                "diagnostic_rows": self.diagnostic_rows,
                "diagnostic_evaluation_count": self.diagnostic_evaluations}

    def close(self):
        try:
            super().close()
        finally:
            if not self.diagnostic_handle.closed:
                self.diagnostic_handle.close()
