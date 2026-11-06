import random
import tempfile
import unittest
from pathlib import Path

import baseline_uncertainty as base
from experiments.run_i_k8_final_oracle_v1 import (
    BALANCED, BASE, METHODS, ORACLE, SIMPLE, summarize)
from multi_instance_logging import fingerprint
from transformer_ppo.archive import ObjectiveNormalizer
from transformer_ppo.initialization_k8 import InitializationK8Controller


def fixture():
    arcs = [
        base.Arc("A", "X", "rail", 10.0, 100.0, 1.0, 1.0, 10.0),
        base.Arc("X", "B", "rail", 10.0, 100.0, 1.0, 1.0, 10.0),
        base.Arc("A", "Y", "road", 8.0, 100.0, 1.0, 1.0, 10.0),
        base.Arc("Y", "B", "road", 8.0, 100.0, 1.0, 1.0, 10.0),
        base.Arc("A", "Z", "water", 12.0, 100.0, 1.0, 1.0, 10.0),
        base.Arc("Z", "B", "water", 12.0, 100.0, 1.0, 1.0, 10.0),
    ]
    paths = [
        base.Path(1, "A", "B", ["A", "X", "B"], ["rail", "rail"],
                  arcs[:2], 5.0, 8.0, 4.0),
        base.Path(2, "A", "B", ["A", "Y", "B"], ["road", "road"],
                  arcs[2:4], 8.0, 5.0, 3.0),
        base.Path(3, "A", "B", ["A", "Z", "B"], ["water", "water"],
                  arcs[4:], 4.0, 4.0, 7.0),
    ]
    batch = base.Batch(0, "A", "B", 10.0, 0.0, 10.0)
    options = {("A", "B", 0): [
        base.ReliablePathOption(paths[0], .8, 1.0, {}),
        base.ReliablePathOption(paths[1], .9, .5, {}),
        base.ReliablePathOption(paths[2], .7, 2.0, {}),
    ]}
    return batch, paths, options, arcs


class InitializationK8Tests(unittest.TestCase):
    def controller(self, folder, mode):
        batch, paths, options, arcs = fixture()
        normalizer = ObjectiveNormalizer(
            (0.0, 0.0, 0.0), (10.0, 10.0, 10.0))
        controller = InitializationK8Controller(
            Path(folder), "run", "scenario", 400,
            selection_mode=mode, reliable_options=options,
            batches=[batch], arcs=arcs, pool_size=8,
            normalizer=normalizer, method=f"I_K8_{mode.upper()}",
            instance={}, seeds={})
        return controller, batch, paths

    def test_candidate_construction_is_k8_and_preserves_base_rng(self):
        with tempfile.TemporaryDirectory() as folder:
            controller, batch, paths = self.controller(folder, "simple")
            try:
                random.seed(41)
                base_individual = base.random_initial_individual(
                    [batch], {("A", "B"): paths}, max_paths=3)
                before = random.getstate()
                controller.initial_individual_from_base(
                    base_individual, [batch], {("A", "B"): paths},
                    max_paths=3, slot_index=0)
                after = random.getstate()
                self.assertEqual(after, before)
                self.assertEqual(len(controller._active["candidates"]), 8)
                self.assertEqual(
                    controller._active["fingerprints"][0],
                    fingerprint(base_individual))
                self.assertEqual(
                    controller.candidate_construction_objective_evaluations, 0)
            finally:
                controller.close()

    def test_simple_evaluates_only_selected_candidate(self):
        with tempfile.TemporaryDirectory() as folder:
            controller, batch, paths = self.controller(folder, "simple")
            calls = []
            try:
                random.seed(43)
                individual = controller.initial_individual(
                    [batch], {("A", "B"): paths}, max_paths=3)

                def evaluate(candidate, *_args, **_kwargs):
                    calls.append(id(candidate))
                    candidate.objectives = (2.0, 2.0, 2.0)
                    candidate.feasible = candidate.feasible_hard = True

                controller.evaluate(evaluate, individual)
                self.assertEqual(len(calls), 1)
                self.assertEqual(controller.evaluations, 1)
                self.assertEqual(controller.candidate_objective_evaluations, 1)
                self.assertEqual(
                    controller.hidden_candidate_objective_evaluations, 0)
            finally:
                controller.close()

    def test_oracle_evaluates_eight_but_advances_logical_budget_once(self):
        with tempfile.TemporaryDirectory() as folder:
            controller, batch, paths = self.controller(folder, "oracle")
            try:
                random.seed(47)
                individual = controller.initial_individual(
                    [batch], {("A", "B"): paths}, max_paths=3)
                candidates = controller._active["candidates"]
                index_by_id = {id(candidate): index
                               for index, candidate in enumerate(candidates)}

                def evaluate(candidate, *_args, **_kwargs):
                    index = index_by_id[id(candidate)]
                    value = 1.0 if index == 3 else 5.0 + index
                    candidate.objectives = (value, value, value)
                    candidate.feasible = candidate.feasible_hard = True

                controller.evaluate(evaluate, individual)
                self.assertEqual(individual.objectives, (1.0, 1.0, 1.0))
                self.assertEqual(controller.evaluations, 1)
                self.assertEqual(controller.candidate_objective_evaluations, 8)
                self.assertEqual(
                    controller.hidden_candidate_objective_evaluations, 7)
                self.assertEqual(
                    controller.summary()[
                        "initialization_k8_physical_objective_evaluations"], 8)
            finally:
                controller.close()

    def test_final_hv_summary_requires_oracle_to_beat_simple(self):
        rows = []
        instances = ["i-1", "i-2", "i-3"]
        seeds = [1, 2]
        factors = {BASE: 1.0, BALANCED: 1.05, SIMPLE: 1.06, ORACLE: 1.10}
        for instance in instances:
            for seed in seeds:
                for method in METHODS:
                    rows.append({
                        "instance": instance, "algorithm_seed": seed,
                        "strategy": method, "final_hypervolume": factors[method],
                        "scenario_digest": "same", "optimisation_seconds": 1.0,
                        "logical_search_objective_evaluations": 400,
                        "hidden_candidate_objective_evaluations": (
                            140 if method == ORACLE else 0),
                        "physical_objective_evaluations": (
                            540 if method == ORACLE else 400),
                    })
        result = summarize(rows, instances, seeds)
        self.assertTrue(result["oracle_final_hv_screen_pass"])
        self.assertGreater(
            result["methods"][ORACLE]["vs_balanced"]["mean"],
            result["methods"][SIMPLE]["vs_balanced"]["mean"])
        self.assertEqual(
            result["methods"][ORACLE][
                "hidden_candidate_objective_evaluations"], 840)


if __name__ == "__main__":
    unittest.main()
