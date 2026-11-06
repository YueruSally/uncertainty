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
    def controller(self, folder, mode):
        batch, parent1, parent2, lookup, arcs = fixture()
        normalizer = ObjectiveNormalizer(
            (0.0, 0.0, 0.0), (100.0, 100.0, 100.0),
            reference=(1.1, 1.1, 1.1))
        controller = CrossoverK8Controller(
            Path(folder), "run", "scenario", 400,
            selection_mode=mode, reliable_options={}, batches=[batch],
            arcs=arcs, pool_size=8, normalizer=normalizer,
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
