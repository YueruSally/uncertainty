import tempfile
import unittest
from pathlib import Path
from unittest import mock

import baseline_uncertainty as base
from experiments.run_c_k8_final_oracle_v1 import (
    BASE, DIRECTED, METHODS, ORACLE, SIMPLE, summarize)
from transformer_ppo.archive import ObjectiveNormalizer
from transformer_ppo.crossover_k8 import CrossoverK8Controller


def path(path_id, nodes, costs):
    arcs = [
        base.Arc(a, b, "rail", 1.0, 100.0, float(cost), 1.0, 1.0)
        for (a, b), cost in zip(zip(nodes[:-1], nodes[1:]), costs)
    ]
    return base.Path(path_id, nodes[0], nodes[-1], list(nodes),
                     ["rail"] * len(arcs), arcs,
                     sum(costs), float(len(arcs)), float(len(arcs)))


def fixture():
    batch = base.Batch(7, "A", "D", 10.0, 0.0, 24.0)
    path1 = path(1, ["A", "X", "J", "D"], [9.0, 9.0, 9.0])
    path2 = path(2, ["A", "X", "K", "J", "D"], [1.0, 1.0, 1.0, 1.0])
    key = ("A", "D", 7)
    parent1 = base.Individual({key: [base.PathAllocation(path1, 1.0)]})
    parent2 = base.Individual({key: [base.PathAllocation(path2, 1.0)]})
    lookup = {(arc.from_node, arc.to_node, arc.mode): arc
              for item in (path1, path2) for arc in item.arcs}
    arcs = list({id(arc): arc for item in (path1, path2)
                 for arc in item.arcs}.values())
    return batch, parent1, parent2, lookup, arcs


class CK8Tests(unittest.TestCase):
    def test_boost_selects_before_evaluation_and_uses_boost_mutation_rate(self):
        for mode in ("simple", "transformer"):
            for boost_rate in (0.0, 1.0):
                with self.subTest(mode=mode, boost_rate=boost_rate), \
                        tempfile.TemporaryDirectory() as folder:
                    policy = mock.Mock()
                    policy.select.return_value = 3
                    controller, batch, p1, p2, lookup, arcs = self.controller(
                        folder, mode, policy if mode == "transformer" else None)
                    controller.phase, controller.generation = "boost", 0
                    path_lib = {("A", "D"): [
                        parent.od_allocations[("A", "D", 7)][0].path
                        for parent in (p1, p2)]}
                    evaluated = []

                    def evaluate(child, *_args, **_kwargs):
                        if mode == "transformer":
                            self.assertEqual(policy.select.call_count, 1)
                        evaluated.append(child)
                        child.objectives = (10., 10., 10.)
                        child.feasible = child.feasible_hard = True

                    try:
                        with mock.patch.object(base, "ACTIVE_MUTATION_LOGGER", controller), \
                                mock.patch.object(base, "capacity_aware_initial_individual",
                                                  return_value=(None, 0.)), \
                                mock.patch.object(base, "crossover_hybrid",
                                                  side_effect=controller.crossover_common_node), \
                                mock.patch.object(base, "_evaluate_individual_impl",
                                                  side_effect=evaluate), \
                                mock.patch.object(base, "MUTATION_RATE", 1. - boost_rate), \
                                mock.patch.object(controller, "_mutate_without_evaluation",
                                                  return_value=("replace", True)) as mutate:
                            base.feasibility_boost(
                                [p1, p2], [batch], path_lib, {}, lookup,
                                arcs, 0., 0., {}, {}, {}, boost_rounds=2,
                                boost_mutation_rate=boost_rate)
                        self.assertEqual(len(evaluated), 2)
                        self.assertEqual(controller.evaluations, 2)
                        self.assertEqual(controller.candidate_objective_evaluations, 2)
                        self.assertEqual(controller.hidden_candidate_objective_evaluations, 0)
                        self.assertEqual(mutate.call_count, 16 if boost_rate else 0)
                        self.assertIsNone(controller._active)
                        controller.finish_episode()
                    finally:
                        controller.close()

    def controller(self, folder, mode, policy=None):
        batch, parent1, parent2, lookup, arcs = fixture()
        normalizer = ObjectiveNormalizer(
            (0.0, 0.0, 0.0), (100.0, 100.0, 100.0),
            reference=(1.1, 1.1, 1.1))
        controller = CrossoverK8Controller(
            Path(folder), "run", "scenario", 400,
            selection_mode=mode, reliable_options={}, batches=[batch],
            arcs=arcs, pool_size=8, normalizer=normalizer,
            selection_policy=policy,
            method=f"C_K8_{mode.upper()}", instance={}, seeds={})
        return controller, batch, parent1, parent2, lookup, arcs

    def test_pool_contains_base_and_eight_complete_pairs(self):
        with tempfile.TemporaryDirectory() as folder:
            controller, batch, p1, p2, lookup, _ = self.controller(
                folder, "simple")
            try:
                c1, c2 = controller.crossover_common_node(
                    p1, p2, [batch], {}, lookup)
                self.assertIs(c1, controller._active["pairs"][0][0])
                self.assertIs(c2, controller._active["pairs"][0][1])
                self.assertEqual(len(controller._active["pairs"]), 8)
                self.assertGreaterEqual(controller._active["unique_count"], 2)
                self.assertEqual(controller._active["roles"][0], "base")
                self.assertEqual(
                    controller.candidate_construction_objective_evaluations, 0)
            finally:
                controller.close()

    def _process(self, controller, batch, p1, p2, lookup, arcs):
        for parent in (p1, p2):
            parent.objectives = (80.0, 80.0, 80.0)
            parent.feasible = parent.feasible_hard = True
        controller.current_population = [p1, p2]
        c1, c2 = controller.crossover_common_node(
            p1, p2, [batch], {}, lookup)
        pairs = controller._active["pairs"]
        by_id = {id(child): index for index, pair in enumerate(pairs)
                 for child in pair}

        def evaluate(child, *_args, **_kwargs):
            index = by_id[id(child)]
            value = 10.0 if index == 3 else 40.0 + index
            child.objectives = (value, value, value)
            child.feasible = child.feasible_hard = True

        with mock.patch.object(base, "MUTATION_RATE", 0.0):
            chosen = controller.process_crossover_pair(
                evaluate, c1, c2, [batch], {}, {}, lookup, arcs, 0.0, 0.0,
                reliable_options={})
        return chosen

    def test_simple_evaluates_only_selected_pair(self):
        with tempfile.TemporaryDirectory() as folder:
            controller, batch, p1, p2, lookup, arcs = self.controller(
                folder, "simple")
            try:
                chosen = self._process(
                    controller, batch, p1, p2, lookup, arcs)
                self.assertEqual(len(chosen), 2)
                self.assertEqual(controller.evaluations, 2)
                self.assertEqual(controller.candidate_objective_evaluations, 2)
                self.assertEqual(
                    controller.hidden_candidate_objective_evaluations, 0)
            finally:
                controller.close()

    def test_oracle_evaluates_sixteen_but_commits_two(self):
        with tempfile.TemporaryDirectory() as folder:
            controller, batch, p1, p2, lookup, arcs = self.controller(
                folder, "oracle")
            try:
                chosen = self._process(
                    controller, batch, p1, p2, lookup, arcs)
                self.assertEqual(chosen[0].objectives, (10.0, 10.0, 10.0))
                self.assertEqual(controller.evaluations, 2)
                self.assertEqual(controller.candidate_objective_evaluations, 16)
                self.assertEqual(
                    controller.hidden_candidate_objective_evaluations, 14)
            finally:
                controller.close()

    def test_collection_records_eight_exact_pair_labels(self):
        with tempfile.TemporaryDirectory() as folder:
            controller, batch, p1, p2, lookup, arcs = self.controller(
                folder, "collect_simple")
            try:
                self._process(controller, batch, p1, p2, lookup, arcs)
                self.assertEqual(len(controller.feedback_rows), 1)
                row = controller.feedback_rows[0]
                self.assertEqual(len(row["labels"]), 8)
                self.assertEqual(len(row["candidate_batch_tokens"]), 8)
                self.assertEqual(row["additional_objective_evaluations"], 14)
                self.assertEqual(controller.evaluations, 2)
            finally:
                controller.close()

    def test_transformer_selects_without_hidden_evaluation(self):
        class Policy:
            def select(self, row):
                self.row = row
                return 3

        policy = Policy()
        with tempfile.TemporaryDirectory() as folder:
            controller, batch, p1, p2, lookup, arcs = self.controller(
                folder, "transformer", policy)
            try:
                chosen = self._process(
                    controller, batch, p1, p2, lookup, arcs)
                self.assertEqual(chosen[0].objectives, (10.0, 10.0, 10.0))
                self.assertEqual(len(policy.row["candidate_features"]), 8)
                self.assertEqual(controller.candidate_objective_evaluations, 2)
                self.assertEqual(
                    controller.hidden_candidate_objective_evaluations, 0)
            finally:
                controller.close()

    def test_deferred_transformer_selects_pre_mutation_pair_for_m_stage(self):
        class Policy:
            def select(self, row):
                self.row = row
                return 3

        policy = Policy()
        with tempfile.TemporaryDirectory() as folder:
            batch, p1, p2, lookup, arcs = fixture()
            normalizer = ObjectiveNormalizer(
                (0.0, 0.0, 0.0), (100.0, 100.0, 100.0),
                reference=(1.1, 1.1, 1.1))
            controller = CrossoverK8Controller(
                Path(folder), "run", "scenario", 400,
                selection_mode="transformer", selection_policy=policy,
                reliable_options={}, batches=[batch], arcs=arcs, pool_size=8,
                defer_selected_pair_mutation=True, normalizer=normalizer,
                method="ICM", instance={}, seeds={})
            try:
                for parent in (p1, p2):
                    parent.objectives = (80.0, 80.0, 80.0)
                    parent.feasible = parent.feasible_hard = True
                controller.current_population = [p1, p2]
                c1, c2 = controller.crossover_common_node(
                    p1, p2, [batch], {}, lookup)
                with mock.patch.object(base, "MUTATION_RATE", 0.0):
                    result = controller.process_crossover_pair(
                        lambda *_args, **_kwargs: self.fail("must not evaluate"),
                        c1, c2, [batch], {}, {}, lookup, arcs, 0.0, 0.0,
                        reliable_options={})
                self.assertIsNone(result)
                self.assertEqual(controller._active["selected_index"], 3)
                self.assertTrue(
                    controller._active["deferred_selected_pair_mutation"])
                self.assertEqual(controller.evaluations, 0)
                self.assertEqual(controller.candidate_objective_evaluations, 0)
            finally:
                controller.close()

    def test_summary_uses_base_as_primary_reference(self):
        rows = []
        for name in ("A", "B", "C"):
            for seed in (1, 2):
                values = {BASE: 1.0, DIRECTED: 1.05,
                          SIMPLE: 1.02, ORACLE: 1.12}
                for method in METHODS:
                    rows.append({
                        "instance": name, "algorithm_seed": seed,
                        "strategy": method, "scenario_digest": "same",
                        "final_hypervolume": values[method],
                        "optimisation_seconds": 1.0,
                        "logical_search_objective_evaluations": 400,
                        "hidden_candidate_objective_evaluations": (
                            14 if method == ORACLE else 0),
                        "physical_objective_evaluations": (
                            414 if method == ORACLE else 400),
                    })
        result = summarize(rows, ["A", "B", "C"], [1, 2])
        self.assertTrue(result["oracle_final_hv_screen_pass"])
        self.assertEqual(result["reference_method"], BASE)
        self.assertGreater(result["methods"][ORACLE]["vs_base"]["mean"], 1.0)


if __name__ == "__main__":
    unittest.main()
