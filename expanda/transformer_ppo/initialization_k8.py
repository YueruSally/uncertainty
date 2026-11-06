"""K8 complete-individual prescreening for initialization diagnostics."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import math
import random
import time

import baseline_uncertainty as base
from multi_instance_logging import fingerprint

from .archive import exact_hv_3d
from .comparison_controllers import BaseBudgetController
from .initialization_rule import _directed_path_ranking
from .v2_controls import private_seed


METHODS = ("balanced", "simple", "oracle")
PROFILE_CANDIDATES = (
    "base", "joint_balanced", "cost", "emission", "time_reliability")


def _path_signature(path):
    return tuple(path.nodes), tuple(path.modes)


def _scale_columns(rows, names):
    for name in names:
        values = [float(row[name]) for row in rows]
        low, high = min(values), max(values)
        span = high - low
        for row in rows:
            row[f"scaled_{name}"] = (
                0.0 if span <= 1e-15 else (float(row[name]) - low) / span)


def deterministic_proxy_rows(candidates, batches, arcs):
    """Score complete individuals without CCP/scenario objective evaluation."""
    capacity = {
        (arc.from_node, arc.to_node, arc.mode): max(0.0, float(arc.capacity))
        for arc in arcs
    }
    rows = []
    for index, individual in enumerate(candidates):
        total_cost = total_emission = capacity_excess = lateness = 0.0
        makespan = 0.0; missing = 0; flows = Counter()
        for batch in batches:
            key = (batch.origin, batch.destination, batch.batch_id)
            allocations = [
                allocation for allocation in individual.od_allocations.get(key, ())
                if float(allocation.share) > 1e-12
            ]
            if not allocations:
                missing += 1
                continue
            completion = float(batch.ET)
            for allocation in allocations:
                share = float(allocation.share)
                flow = share * float(batch.quantity)
                path = allocation.path
                total_cost += flow * float(path.base_cost_per_teu)
                total_emission += flow * float(path.base_emission_per_teu)
                completion = max(
                    completion, float(batch.ET) + float(path.base_travel_time_h))
                for arc in path.arcs:
                    flows[(arc.from_node, arc.to_node, arc.mode)] += flow
            makespan = max(makespan, completion)
            lateness += max(0.0, completion - float(batch.LT)) * float(batch.quantity)
        for key, flow in flows.items():
            cap = capacity.get(key, 0.0)
            if cap <= 0.0:
                capacity_excess += flow
            else:
                capacity_excess += max(0.0, flow - cap) / cap
        rows.append({
            "candidate_index": index,
            "cost": total_cost,
            "emission": total_emission,
            "time": makespan,
            "capacity_excess": capacity_excess,
            "lateness": lateness,
            "missing": missing,
        })
    names = ("cost", "emission", "time", "capacity_excess", "lateness")
    _scale_columns(rows, names)
    for row in rows:
        # Fixed before the screen: average deterministic objective pressure,
        # plus strong capacity/missing penalties and a lateness penalty.
        row["score"] = (
            (row["scaled_cost"] + row["scaled_emission"]
             + row["scaled_time"]) / 3.0
            + 2.0 * row["scaled_capacity_excess"]
            + row["scaled_lateness"]
            + 10.0 * float(row["missing"])
        )
    return rows


class InitializationK8Controller(BaseBudgetController):
    """Prescreen eight complete initial individuals at each Base-random slot."""

    controls_initialization = True
    controls_crossover = False
    controls_mutation = False

    def __init__(self, *args, selection_mode, reliable_options, batches, arcs,
                 pool_size=8, **kwargs):
        if selection_mode not in METHODS:
            raise ValueError(f"selection_mode must be one of {METHODS}")
        if int(pool_size) != 8:
            raise ValueError("the frozen initialization screen requires K=8")
        self.selection_mode = selection_mode
        self.reliable_options = dict(reliable_options)
        self.batches_for_proxy = tuple(batches)
        self.arcs_for_proxy = tuple(arcs)
        self.pool_size = int(pool_size)
        self.current_population = []
        self._active = None
        self.pool_events = 0
        self.candidate_objective_evaluations = 0
        self.hidden_candidate_objective_evaluations = 0
        self.candidate_construction_objective_evaluations = 0
        self.selection_counts = Counter()
        self.oracle_gain_total = 0.0
        self.proxy_selected_gain_total = 0.0
        super().__init__(*args, log_names=(
            "initialization_k8_candidates", "generation_summary"), **kwargs)

    def _ranked_paths(self, paths, batch, profile, event_index):
        ranking_profile = profile
        ranking_index = event_index
        if profile in ("cost", "emission", "time_reliability"):
            ranking_profile = "diverse_cycle"
            ranking_index = {
                "cost": 1, "emission": 2, "time_reliability": 3,
            }[profile]
        rows, _ = _directed_path_ranking(
            paths, batch,
            self.reliable_options.get(
                (batch.origin, batch.destination, batch.batch_id), ()),
            ranking_profile, ranking_index)
        return [row["path"] for row in rows]

    def initial_individual_from_base(
            self, base_individual, batches, path_lib, max_paths=3,
            slot_index=None):
        """Build K8 around the exact Base candidate without advancing Base RNG."""
        if self._active is not None:
            raise RuntimeError("previous K8 initialization event was not evaluated")
        self.pool_events += 1
        event_index = self.pool_events - 1
        if slot_index is not None and int(slot_index) != event_index:
            raise RuntimeError("initialization slot order is inconsistent")
        candidates = [deepcopy(base_individual)
                      for _ in range(self.pool_size)]
        for batch in batches:
            key = (batch.origin, batch.destination, batch.batch_id)
            paths = list(path_lib.get((batch.origin, batch.destination), ()))
            if not paths:
                continue
            base_allocations = list(base_individual.od_allocations.get(key, ()))
            if base_allocations:
                k = min(len(base_allocations), int(max_paths), len(paths))
                raw_shares = [float(item.share)
                              for item in base_allocations[:k]]
            else:
                k = 1
                raw_shares = [1.0]
            selected_by_candidate = [[
                item.path for item in base_allocations[:k]
            ]]
            if not selected_by_candidate[0]:
                selected_by_candidate[0] = [paths[0]]
            for profile in PROFILE_CANDIDATES[1:]:
                selected_by_candidate.append(
                    self._ranked_paths(paths, batch, profile, event_index)[:k])
            for candidate_index in range(len(PROFILE_CANDIDATES), self.pool_size):
                generator = random.Random(private_seed(
                    "I-K8-private", self.run_id, event_index,
                    batch.batch_id, candidate_index))
                selected_by_candidate.append(generator.sample(paths, k))
            # Candidate zero is the untouched Base individual.  Only replace
            # allocations in the seven alternatives.
            for candidate, selected in zip(
                    candidates[1:], selected_by_candidate[1:]):
                candidate.od_allocations[key] = base.merge_and_normalize([
                    base.PathAllocation(path=path, share=raw_shares[position])
                    for position, path in enumerate(selected)
                ])
        self._active = {
            "event_index": event_index,
            "placeholder_id": id(candidates[0]),
            "candidates": candidates,
            "fingerprints": [fingerprint(candidate) for candidate in candidates],
        }
        return candidates[0]

    def initial_individual(self, batches, path_lib, max_paths=3):
        """Compatibility entry point for callers without the all-slot hook."""
        base_individual = base.random_initial_individual(
            batches, path_lib, max_paths=max_paths)
        return self.initial_individual_from_base(
            base_individual, batches, path_lib, max_paths=max_paths)

    def _population_hv(self, extra=None):
        individuals = list(self.current_population)
        if extra is not None:
            individuals.append(extra)
        points = [
            self.normalizer.transform(individual.objectives)
            for individual in individuals if individual.feasible
        ]
        return exact_hv_3d(points, self.normalizer.reference)

    @staticmethod
    def _copy_individual(target, source):
        if target is source:
            return
        target.__dict__.clear()
        target.__dict__.update(deepcopy(source.__dict__))

    def _record(self, active, proxy_rows, selected_index, gains):
        for index, row in enumerate(proxy_rows):
            candidate = active["candidates"][index]
            evaluated = all(math.isfinite(float(value))
                            for value in candidate.objectives)
            self.append("initialization_k8_candidates", {
                "event_index": active["event_index"],
                "candidate_index": index,
                "candidate_role": (
                    PROFILE_CANDIDATES[index]
                    if index < len(PROFILE_CANDIDATES)
                    else f"private_random_{index-len(PROFILE_CANDIDATES)+1}"),
                "fingerprint": active["fingerprints"][index],
                "proxy": row,
                "objectives": (
                    [float(value) for value in candidate.objectives]
                    if evaluated else None),
                "feasible": bool(candidate.feasible) if evaluated else None,
                "oracle_hv_gain": None if gains is None else gains[index],
                "selected": index == selected_index,
                "selection_mode": self.selection_mode,
            })

    def evaluate(self, function, ind, *args, **kwargs):
        active = self._active
        if active is None or id(ind) != active["placeholder_id"]:
            before = self.evaluations
            result = super().evaluate(function, ind, *args, **kwargs)
            if self.generation < 0 and self.evaluations > before:
                self.current_population.append(ind)
            return result

        candidates = active["candidates"]
        proxy_rows = deterministic_proxy_rows(
            candidates, self.batches_for_proxy, self.arcs_for_proxy)
        gains = None
        if self.selection_mode == "balanced":
            selected_index = 1
            self._copy_individual(ind, candidates[selected_index])
            result = super().evaluate(function, ind, *args, **kwargs)
            self.candidate_objective_evaluations += 1
        elif self.selection_mode == "simple":
            selected_index = min(
                range(len(candidates)),
                key=lambda index: (proxy_rows[index]["score"],
                                   index != 1, index))
            self._copy_individual(ind, candidates[selected_index])
            result = super().evaluate(function, ind, *args, **kwargs)
            self.candidate_objective_evaluations += 1
            base_hv = self._population_hv()
            selected_gain = self._population_hv(ind) - base_hv
            self.proxy_selected_gain_total += selected_gain
        else:
            started = time.perf_counter()
            for candidate in candidates:
                function(candidate, *args, **kwargs)
            self.objective_evaluation_seconds += time.perf_counter() - started
            self.candidate_objective_evaluations += len(candidates)
            self.hidden_candidate_objective_evaluations += len(candidates) - 1
            base_hv = self._population_hv()
            gains = [self._population_hv(candidate) - base_hv
                     for candidate in candidates]
            selected_index = max(
                range(len(candidates)),
                key=lambda index: (
                    gains[index], bool(candidates[index].feasible),
                    index == 1, -proxy_rows[index]["score"], -index))
            self.oracle_gain_total += gains[selected_index]
            self._copy_individual(ind, candidates[selected_index])
            if not self.has_budget(1):
                raise RuntimeError("logical objective evaluation budget exhausted")
            self.evaluations += 1
            ind._learning_eval_key = (self.scenario_id, fingerprint(ind))
            ind._learning_eval_id = self.evaluations
            self._archive_pending.append(
                (ind.objectives, bool(ind.feasible), fingerprint(ind)))
            result = None

        self.total_attempts += 1
        self.total_effective += int(
            active["fingerprints"][selected_index] != active["fingerprints"][0])
        role = (PROFILE_CANDIDATES[selected_index]
                if selected_index < len(PROFILE_CANDIDATES)
                else "private_random")
        self.selection_counts[role] += 1
        self._record(active, proxy_rows, selected_index, gains)
        self.current_population.append(ind)
        self._active = None
        return result

    def finish_episode(self):
        if self._active is not None:
            raise RuntimeError("episode ended with an unevaluated K8 candidate pool")
        super().finish_episode()

    def summary(self):
        result = super().summary()
        result.update({
            "initialization_k8_selection_mode": self.selection_mode,
            "initialization_k8_pool_size": self.pool_size,
            "initialization_k8_pool_events": self.pool_events,
            "initialization_k8_selection_counts": dict(self.selection_counts),
            "initialization_k8_candidate_objective_evaluations": (
                self.candidate_objective_evaluations),
            "initialization_k8_hidden_candidate_objective_evaluations": (
                self.hidden_candidate_objective_evaluations),
            "initialization_k8_candidate_construction_objective_evaluations": (
                self.candidate_construction_objective_evaluations),
            "initialization_k8_oracle_gain_total": self.oracle_gain_total,
            "initialization_k8_proxy_selected_gain_total": (
                self.proxy_selected_gain_total),
            "initialization_k8_physical_objective_evaluations": (
                self.evaluations + self.hidden_candidate_objective_evaluations),
        })
        return result
