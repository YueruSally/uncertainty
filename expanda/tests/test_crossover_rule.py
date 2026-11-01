import random
import tempfile
import unittest
from pathlib import Path

import baseline_uncertainty as base
from transformer_ppo.crossover_rule import (
    CrossoverRuleController,
    _directed_cut_ranking,
    _joint_single_override_plans,
    _safe_balanced_override,
)


def candidate(cost, emission, time, deadline, road=0.0, rail=1.0,
              water=0.0, border=0.0, join="J"):
    return {
        "child_cost_delta": cost,
        "child_emission_delta": emission,
        "child_time_delta": time,
        "child_combined_time": time + 10.0,
        "deadline_window_h": deadline,
        "child_1_road_fraction": road,
        "child_2_road_fraction": road,
        "child_1_rail_fraction": rail,
        "child_2_rail_fraction": rail,
        "child_1_water_fraction": water,
        "child_2_water_fraction": water,
        "child_border_fraction": border,
        "join_node": join,
    }


def path(path_id, nodes, costs):
    arcs = [
        base.Arc(a, b, "rail", 1.0, 100.0, float(cost), 1.0, 1.0)
        for (a, b), cost in zip(zip(nodes[:-1], nodes[1:]), costs)
    ]
    return base.Path(
        path_id, nodes[0], nodes[-1], list(nodes),
        ["rail"] * len(arcs), arcs,
        sum(costs), float(len(arcs)), float(len(arcs)))


class CrossoverRuleTests(unittest.TestCase):
    def test_balanced_ranking_uses_cost_emission_time_and_risk(self):
        risky_cheap = candidate(
            -10.0, -10.0, 20.0, 10.0, water=1.0, rail=0.0,
            border=1.0, join="RISKY")
        balanced = candidate(
            -5.0, -5.0, -5.0, 20.0, rail=1.0, join="BALANCED")
        dirty_fast = candidate(
            10.0, 10.0, -10.0, 20.0, road=1.0, rail=0.0,
            join="DIRTY")
        ranked = _directed_cut_ranking(
            [risky_cheap, balanced, dirty_fast])
        self.assertEqual(ranked[0]["candidate"]["join_node"], "BALANCED")
        self.assertEqual(len(ranked[0]["raw"]), 5)
        self.assertEqual(len(ranked[0]["scaled"]), 5)

    def test_safe_override_requires_five_signal_pareto_improvement(self):
        shadow = candidate(0.0, 0.0, 0.0, 20.0, join="BASE")
        tradeoff = candidate(-2.0, 1.0, -1.0, 20.0, join="TRADEOFF")
        safe = candidate(-1.0, -1.0, -1.0, 20.0, join="SAFE")
        ranked = _directed_cut_ranking([shadow, tradeoff, safe])
        chosen, reference = _safe_balanced_override(ranked, 0)
        self.assertEqual(reference["candidate"]["join_node"], "BASE")
        self.assertEqual(chosen["candidate"]["join_node"], "SAFE")

    def test_joint_single_uses_one_event_wide_candidate_pool(self):
        class Cut:
            def __init__(self, candidates, shadow_base_index=0):
                self.candidates = tuple(candidates)
                self.shadow_base_index = shadow_base_index

        cuts = [
            Cut([
                candidate(0.0, 0.0, 0.0, 20.0, join="BASE-A"),
                candidate(-1.0, -1.0, -1.0, 20.0, join="GOOD-A"),
            ]),
            Cut([
                candidate(0.0, 0.0, 0.0, 20.0, join="BASE-B"),
                candidate(-3.0, -3.0, -3.0, 20.0, join="BEST-B"),
            ]),
        ]
        plans, chosen = _joint_single_override_plans(cuts, safe=False)
        self.assertEqual(chosen["candidate"]["join_node"], "BEST-B")
        self.assertEqual(
            sum(
                plan["candidate_index"] != cut.shadow_base_index
                for plan, cut in zip(plans, cuts)),
            1)

    def test_joint_safe_single_falls_back_on_tradeoff_only_pool(self):
        class Cut:
            def __init__(self, candidates, shadow_base_index=0):
                self.candidates = tuple(candidates)
                self.shadow_base_index = shadow_base_index

        cuts = [Cut([
            candidate(0.0, 0.0, 0.0, 20.0, join="BASE"),
            candidate(-2.0, 1.0, -1.0, 20.0, join="TRADEOFF"),
        ])]
        plans, chosen = _joint_single_override_plans(cuts, safe=True)
        self.assertTrue(chosen["is_base"])
        self.assertEqual(plans[0]["candidate_index"], 0)

    def test_rule_preserves_base_rng_and_overrides_all_learnable_cuts(self):
        batch = base.Batch(7, "A", "D", 10.0, 0.0, 24.0)
        path1 = path(1, ["A", "X", "J", "D"], [9.0, 9.0, 9.0])
        path2 = path(2, ["A", "X", "K", "J", "D"], [1.0, 1.0, 1.0, 1.0])
        key = ("A", "D", 7)
        parent1 = base.Individual({
            key: [base.PathAllocation(path1, 1.0)]})
        parent2 = base.Individual({
            key: [base.PathAllocation(path2, 1.0)]})
        lookup = {
            (arc.from_node, arc.to_node, arc.mode): arc
            for item in (path1, path2) for arc in item.arcs
        }
        random.seed(41)
        base.crossover_common_node(
            parent1, parent2, [batch], {}, lookup)
        expected_state = random.getstate()
        with tempfile.TemporaryDirectory() as directory:
            controller = CrossoverRuleController(
                Path(directory), "run", "scenario", 20,
                normalizer=None, profile="joint_balanced", min_generation=0,
                method="rule-c", instance={}, seeds={})
            try:
                controller.generation = 0
                random.seed(41)
                controller.crossover_common_node(
                    parent1, parent2, [batch], {}, lookup)
                self.assertEqual(random.getstate(), expected_state)
                self.assertEqual(controller.directed_cut_count, 1)
                self.assertEqual(controller.total_attempts, 1)
                self.assertEqual(controller.evaluations, 0)
            finally:
                controller.close()

    def test_pre_schedule_rule_replays_exact_base_children(self):
        batch = base.Batch(7, "A", "D", 10.0, 0.0, 24.0)
        path1 = path(1, ["A", "X", "J", "D"], [9.0, 9.0, 9.0])
        path2 = path(2, ["A", "X", "K", "J", "D"], [1.0, 1.0, 1.0, 1.0])
        key = ("A", "D", 7)
        parent1 = base.Individual({
            key: [base.PathAllocation(path1, 1.0)]})
        parent2 = base.Individual({
            key: [base.PathAllocation(path2, 1.0)]})
        lookup = {
            (arc.from_node, arc.to_node, arc.mode): arc
            for item in (path1, path2) for arc in item.arcs
        }
        random.seed(73)
        expected1, expected2 = base.crossover_common_node(
            parent1, parent2, [batch], {}, lookup)
        expected_state = random.getstate()
        with tempfile.TemporaryDirectory() as directory:
            controller = CrossoverRuleController(
                Path(directory), "run", "scenario", 20,
                normalizer=None, profile="joint_balanced", min_generation=10,
                method="rule-c", instance={}, seeds={})
            try:
                controller.generation = 0
                random.seed(73)
                actual1, actual2 = controller.crossover_common_node(
                    parent1, parent2, [batch], {}, lookup)
                self.assertEqual(actual1.od_allocations, expected1.od_allocations)
                self.assertEqual(actual2.od_allocations, expected2.od_allocations)
                self.assertEqual(random.getstate(), expected_state)
                self.assertEqual(controller.directed_cut_count, 0)
            finally:
                controller.close()


if __name__ == "__main__":
    unittest.main()
