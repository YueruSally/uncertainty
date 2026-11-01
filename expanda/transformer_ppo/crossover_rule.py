"""Deterministic directed-balanced teacher for common-node crossover."""
from __future__ import annotations

import math

import numpy as np

from .comparison_controllers import BaseBudgetController
from .crossover import apply_cut, finalize_children, prepare_common_node_crossover


CROSSOVER_RULE_PROFILES = ("joint_balanced",)


def _finite(value, default=0.0):
    value = float(value)
    return value if math.isfinite(value) else float(default)


def _directed_cut_ranking(candidates, profile="joint_balanced"):
    """Rank legal cut nodes without any real objective evaluation.

    New crossover paths are not members of the precomputed reliable-path
    library, so the final two risk terms are deterministic zero-cost proxies:
    deadline pressure and modal/border uncertainty exposure.
    """
    if profile not in CROSSOVER_RULE_PROFILES:
        raise ValueError(f"invalid directed C profile: {profile}")
    rows = []
    mode_cv = {"road": 0.15, "rail": 0.10, "water": 0.20}
    for index, source in enumerate(candidates):
        candidate = dict(source)
        deadline = max(1.0, _finite(candidate.get("deadline_window_h"), 1.0))
        child_time = _finite(candidate.get("child_combined_time"))
        modal_cv = 0.5 * sum(
            mode_cv[mode] * (
                _finite(candidate.get(f"child_1_{mode}_fraction"))
                + _finite(candidate.get(f"child_2_{mode}_fraction")))
            for mode in mode_cv)
        deadline_pressure = max(0.0, child_time - deadline) / deadline
        uncertainty_exposure = (
            child_time * modal_cv / deadline
            + 0.5 * _finite(candidate.get("child_border_fraction")))
        raw = [
            _finite(candidate.get("child_cost_delta")),
            _finite(candidate.get("child_emission_delta")),
            _finite(candidate.get("child_time_delta")),
            deadline_pressure,
            uncertainty_exposure,
        ]
        rows.append({
            "candidate": candidate,
            "candidate_index": index,
            "raw": raw,
            "deadline_pressure": deadline_pressure,
            "uncertainty_exposure": uncertainty_exposure,
        })
    if not rows:
        return []
    values = np.asarray([row["raw"] for row in rows], dtype=float)
    minima = values.min(axis=0)
    spans = values.max(axis=0) - minima
    spans[spans <= 1e-15] = 1.0
    scaled = (values - minima) / spans
    for row, vector in zip(rows, scaled):
        row["scaled"] = vector.tolist()
        row["score"] = float(
            vector[0] + vector[1]
            + (vector[2] + vector[3] + vector[4]) / 3.0)
    rows.sort(key=lambda row: (
        row["score"],
        int(row["candidate_index"]),
        str(row["candidate"].get("join_node"))))
    return rows


class CrossoverRuleController(BaseBudgetController):
    """Apply the best balanced legal cut for every common-node decision."""

    controls_crossover = True

    def __init__(self, *args, profile="joint_balanced", min_generation=0,
                 **kwargs):
        if profile not in CROSSOVER_RULE_PROFILES:
            raise ValueError(f"invalid directed C profile: {profile}")
        self.profile = str(profile)
        self.min_generation = int(min_generation)
        if self.min_generation < 0:
            raise ValueError("directed C minimum generation cannot be negative")
        self.directed_events = 0
        self.directed_cut_count = 0
        self.overridden_cut_count = 0
        self._active_crossover = None
        kwargs["log_names"] = (
            "crossover_rule_events", "generation_summary")
        super().__init__(*args, **kwargs)

    def crossover_common_node(
            self, parent1, parent2, batches, tt_dict, arc_lookup):
        child1, child2, decisions = prepare_common_node_crossover(
            parent1, parent2, batches, tt_dict=tt_dict)
        self.total_attempts += 1
        directed = self.generation >= self.min_generation
        self.directed_events += int(directed and bool(decisions))
        selected = []
        changed = False
        for cut_index, cut in enumerate(decisions):
            if directed and len(cut.candidates) >= 2:
                ranked = _directed_cut_ranking(cut.candidates, self.profile)
                candidate_index = int(ranked[0]["candidate_index"])
                metadata = ranked[0]
                self.directed_cut_count += 1
            else:
                candidate_index = int(cut.shadow_base_index)
                metadata = None
            candidate = cut.candidates[candidate_index]
            changed = apply_cut(
                child1, child2, cut, candidate, tt_dict, arc_lookup) or changed
            overridden = candidate_index != int(cut.shadow_base_index)
            self.overridden_cut_count += int(directed and overridden)
            selected.append({
                "cut_index": cut_index,
                "batch_id": int(cut.batch.batch_id),
                "candidate_count": len(cut.candidates),
                "shadow_base_index": int(cut.shadow_base_index),
                "chosen_index": candidate_index,
                "join_node": candidate.get("join_node"),
                "different_from_shadow_base": overridden,
                "score": None if metadata is None else metadata["score"],
                "raw_proxy_vector": (
                    None if metadata is None else metadata["raw"]),
                "scaled_proxy_vector": (
                    None if metadata is None else metadata["scaled"]),
            })
        finalize_children(child1, child2)
        self.total_effective += int(changed)
        self._active_crossover = {
            "generation": int(self.generation),
            "directed": bool(directed),
            "decision_count": len(decisions),
            "changed": bool(changed),
            "selected_cuts": selected,
        }
        if not decisions:
            self.no_candidate += 1
        return child1, child2

    def settle_crossover(self, child1, child2):
        event = self._active_crossover
        if event is None:
            # Structural crossover does not pass through this controller.
            return
        self.append("crossover_rule_events", {
            **event,
            "child_objectives": [
                list(map(float, child1.objectives)),
                list(map(float, child2.objectives)),
            ],
            "child_feasible": [bool(child1.feasible), bool(child2.feasible)],
        })
        self._active_crossover = None

    def summary(self):
        return {
            **super().summary(),
            "crossover_rule_profile": self.profile,
            "crossover_rule_min_generation": self.min_generation,
            "directed_crossover_events": self.directed_events,
            "directed_cut_count": self.directed_cut_count,
            "overridden_cut_count": self.overridden_cut_count,
        }
