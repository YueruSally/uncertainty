from pathlib import Path
import json
import tempfile
import unittest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import baseline_uncertainty as base
from transformer_ppo.archive import ObjectiveNormalizer, RewardConfig
from transformer_ppo.comparison_controllers import (
    BaseBudgetController, CombinedPPOController,
)
from transformer_ppo.episode import _optimisation_vector


class _FrozenTrainer:
    name = "frozen"
    training = False
    updates = 0

    def __init__(self, component):
        self.component = component
        self.observation_provider = None
        self.pending = None
        self.sequence_depth = 0
        self.sequence_policy_actions = 0

    def finish(self):
        if self.pending or self.sequence_depth or self.sequence_policy_actions:
            raise RuntimeError("unfinished frozen trainer")


class ComparisonControllerTests(unittest.TestCase):
    def setUp(self):
        self.normalizer = ObjectiveNormalizer(
            (0.0, 0.0, 0.0), (10.0, 10.0, 10.0),
            reference=(1.1, 1.1, 1.1))
        self.batch = base.Batch(1, "A", "D", 10.0, 0.0, 24.0)

    def test_base_budget_controller_does_not_replace_components(self):
        with tempfile.TemporaryDirectory() as directory:
            controller = BaseBudgetController(
                Path(directory), "run", "scenario", 20,
                method="base-nsga2", instance={}, seeds={})
            try:
                self.assertFalse(controller.controls_initialization)
                self.assertFalse(controller.controls_crossover)
                self.assertFalse(controller.controls_mutation)
                self.assertEqual(controller.budget_reserve("offspring_pair"), 2)
            finally:
                controller.close()

    def test_base_records_initial_population_hv_and_evaluation_time(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            controller = BaseBudgetController(
                root, "run", "scenario", 20,
                method="base-nsga2", instance={"instance_id": "toy"}, seeds={},
                normalizer=self.normalizer)
            individual = base.Individual()
            individual.rank = 0
            try:
                def evaluate(item):
                    item.objectives = (2.0, 3.0, 4.0)
                    item.feasible = True

                controller.evaluate(evaluate, individual)
                self.assertEqual(controller.archive.points, ())
                controller.log_initial_population([individual])
                self.assertEqual(len(controller.archive.points), 1)
            finally:
                controller.close()
            row = json.loads(
                (root / "search_progress.jsonl").read_text().strip())
            self.assertEqual(row["checkpoint"], "initial_population")
            self.assertEqual(row["real_objective_evaluation_count"], 1)
            self.assertGreater(row["population_hv"], 0.0)
            self.assertEqual(row["population_hv"], row["archive_hv"])
            self.assertGreaterEqual(
                row["objective_evaluation_seconds_cumulative"], 0.0)
            initial = json.loads(
                (root / "initial_feasible_objectives.json").read_text())
            self.assertEqual(len(initial), 1)
            self.assertEqual(initial[0]["objectives"], [2.0, 3.0, 4.0])

    def test_final_objectives_use_cost_emission_makespan_axis_order(self):
        row = {"optimisation_objectives": {
            "makespan": 30.0, "cost": 10.0, "emission": 20.0,
        }}
        self.assertEqual(_optimisation_vector(row), (10.0, 20.0, 30.0))

    def test_combined_evaluates_once_and_updates_three_archives(self):
        trainers = {name: _FrozenTrainer(name) for name in ("I", "C", "M")}
        with tempfile.TemporaryDirectory() as directory:
            controller = CombinedPPOController(
                Path(directory), "run", "scenario", 20,
                trainers=trainers, normalizer=self.normalizer,
                reward_config=RewardConfig(), batches=[self.batch],
                generations=2, pop_size=2, graph_observation=None,
                method="combined", instance={"instance_id": "toy"}, seeds={})
            calls = []
            individual = base.Individual()
            try:
                def evaluate(item):
                    calls.append(item)
                    item.objectives = (2.0, 3.0, 4.0)
                    item.feasible = True

                controller.evaluate(evaluate, individual)
                controller.evaluate(evaluate, individual)
                self.assertEqual(len(calls), 1)
                self.assertEqual(controller.evaluations, 1)
                self.assertEqual(controller.cache_hits, 1)
                self.assertTrue(all(
                    len(child.archive.points) == 1
                    for child in controller.controllers.values()))
                self.assertTrue(all(
                    child.evaluations == 1
                    for child in controller.controllers.values()))
                summary = controller.summary()
                self.assertEqual(
                    summary["ppo_updates"], {"I": 0, "C": 0, "M": 0})
                self.assertEqual(
                    summary["ppo_updates_before_run"],
                    {"I": 0, "C": 0, "M": 0})
                self.assertEqual(
                    summary["ppo_updates_total"],
                    {"I": 0, "C": 0, "M": 0})
            finally:
                controller.close()

    def test_frozen_summary_separates_checkpoint_and_run_updates(self):
        trainers = {name: _FrozenTrainer(name) for name in ("I", "C", "M")}
        for trainer, updates in zip(trainers.values(), (1, 2, 3)):
            trainer.updates = updates
        with tempfile.TemporaryDirectory() as directory:
            controller = CombinedPPOController(
                Path(directory), "run", "scenario", 20,
                trainers=trainers, normalizer=self.normalizer,
                reward_config=RewardConfig(), batches=[self.batch],
                generations=2, pop_size=2, graph_observation=None,
                method="combined", instance={"instance_id": "toy"}, seeds={})
            try:
                summary = controller.summary()
                self.assertEqual(
                    summary["ppo_updates"], {"I": 0, "C": 0, "M": 0})
                self.assertEqual(
                    summary["ppo_updates_before_run"],
                    {"I": 1, "C": 2, "M": 3})
                self.assertEqual(
                    summary["ppo_updates_total"],
                    {"I": 1, "C": 2, "M": 3})
            finally:
                controller.close()

    def test_combined_withholds_each_active_component_reward_candidate(self):
        trainers = {name: _FrozenTrainer(name) for name in ("I", "C", "M")}
        with tempfile.TemporaryDirectory() as directory:
            controller = CombinedPPOController(
                Path(directory), "run", "scenario", 20,
                trainers=trainers, normalizer=self.normalizer,
                reward_config=RewardConfig(), batches=[self.batch],
                generations=2, pop_size=2, graph_observation=None,
                method="combined", instance={"instance_id": "toy"}, seeds={})
            individual = base.Individual()
            initial = controller.controllers["I"]
            crossover = controller.controllers["C"]
            mutation = controller.controllers["M"]
            initial._active_event = {"individual_id": id(individual)}
            crossover._pending_child_ids = {id(individual)}
            mutation._suppress_archive = True
            try:
                controller.evaluate(
                    lambda item: (
                        setattr(item, "objectives", (2.0, 3.0, 4.0)),
                        setattr(item, "feasible", True)),
                    individual)
                self.assertTrue(all(
                    len(child.archive.points) == 0
                    for child in controller.controllers.values()))
            finally:
                initial._active_event = None
                crossover._pending_child_ids.clear()
                mutation._suppress_archive = False
                controller.close()


if __name__ == "__main__":
    unittest.main()
