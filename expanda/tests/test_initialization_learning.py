from pathlib import Path
import random
import sys
import tempfile
import unittest
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import baseline_uncertainty as base
from multi_instance_policy import Decision
from transformer_ppo.archive import ObjectiveNormalizer, RewardArchive, RewardConfig
from transformer_ppo.initialization_controller import InitializationPPOController
from transformer_ppo.initialization_observations import INITIALIZATION_MODEL_SPEC


def _path(path_id, nodes):
    arcs = [base.Arc(a, b, "road", 1.0, 100.0, 1.0, 1.0, 1.0)
            for a, b in zip(nodes[:-1], nodes[1:])]
    return base.Path(
        path_id, nodes[0], nodes[-1], list(nodes), ["road"] * len(arcs), arcs,
        float(len(arcs)), float(len(arcs)), float(len(arcs)))


class _FakeTrainer:
    name = "fake-ppo-i-masked"
    component = "I"
    updates = 0

    def __init__(self, fail_on_choose=False):
        self.fail_on_choose = fail_on_choose
        self.observation_provider = None
        self.pending = None
        self.sequence_depth = 0
        self.sequence_policy_actions = 0
        self.rewards = []
        self.done = []

    def begin_sequence(self):
        self.sequence_depth = 1
        self.sequence_policy_actions = 0

    def choose(self, candidates, context=None):
        if self.fail_on_choose:
            raise AssertionError("forced path must not invoke the model")
        observation = self.observation_provider(candidates, context or {})
        self.last_observation = observation
        self.pending = True
        self.sequence_policy_actions += 1
        chosen = len(candidates) - 1
        probabilities = [0.0] * len(candidates)
        probabilities[chosen] = 1.0
        return Decision(chosen, probabilities, [None] * len(candidates), {
            "decision_kind": "model_choice", "inference_seconds": 0.001,
        })

    def observe_outcome(self, reward, done=False):
        if self.pending:
            self.rewards.append(float(reward))
            self.done.append(bool(done))
        self.pending = None

    def end_sequence(self, reward, done=False):
        self.observe_outcome(reward, done=done)
        self.sequence_depth = 0
        self.sequence_policy_actions = 0

    def finish(self):
        if self.pending or self.sequence_depth or self.sequence_policy_actions:
            raise RuntimeError("unfinished fake policy sequence")


class InitializationControllerTests(unittest.TestCase):
    def setUp(self):
        self.batch = base.Batch(4, "A", "D", 12.0, 1.0, 25.0)
        self.paths = [
            _path(1, ["A", "X", "D"]),
            _path(2, ["A", "Y", "D"]),
            _path(3, ["A", "Z", "D"]),
        ]
        self.path_lib = {("A", "D"): self.paths}
        self.normalizer = ObjectiveNormalizer(
            (0.0, 0.0, 0.0), (10.0, 10.0, 10.0), reference=(1.1, 1.1, 1.1))

    def _controller(self, directory, trainer):
        return InitializationPPOController(
            Path(directory), "run", "scenario", 50, trainer=trainer,
            archive=RewardArchive(self.normalizer), reward_config=RewardConfig(),
            batches=[self.batch], generations=2, pop_size=4,
            method=trainer.name,
            instance={"instance_id": "toy", "K": 1,
                      "deadline_window_alpha": 1.0}, seeds={})

    def test_model_changes_only_path_identity_and_preserves_base_rng_stream(self):
        random.seed(42)
        base_individual = base.random_initial_individual(
            [self.batch], self.path_lib, max_paths=3)
        expected_state = random.getstate()
        base_shares = sorted(
            round(item.share, 12)
            for item in base_individual.od_allocations[("A", "D", 4)])

        trainer = _FakeTrainer()
        with tempfile.TemporaryDirectory() as directory:
            controller = self._controller(directory, trainer)
            try:
                random.seed(42)
                learned = controller.initial_individual(
                    [self.batch], self.path_lib, max_paths=3)
                self.assertEqual(random.getstate(), expected_state)
                allocations = learned.od_allocations[("A", "D", 4)]
                self.assertEqual(len({_path_signature(item.path) for item in allocations}),
                                 len(allocations))
                self.assertEqual(
                    sorted(round(item.share, 12) for item in allocations), base_shares)
                self.assertEqual(
                    len(trainer.last_observation["parent"][0]),
                    INITIALIZATION_MODEL_SPEC["parent_dim"])
                self.assertEqual(
                    len(trainer.last_observation["candidates"][0]),
                    INITIALIZATION_MODEL_SPEC["candidate_dim"])
                self.assertEqual(
                    len(trainer.last_observation["global"]),
                    INITIALIZATION_MODEL_SPEC["global_dim"])
            finally:
                controller.close()

    def test_complete_individual_receives_one_terminal_reward(self):
        trainer = _FakeTrainer()
        with tempfile.TemporaryDirectory() as directory:
            controller = self._controller(directory, trainer)
            try:
                random.seed(42)
                individual = controller.initial_individual(
                    [self.batch], self.path_lib, max_paths=3)
                individual.objectives, individual.feasible = (2.0, 3.0, 4.0), True
                controller.settle_initialization(individual)
                self.assertTrue(all(value == 0.0 for value in trainer.rewards[:-1]))
                self.assertGreater(trainer.rewards[-1], 0.0)
                self.assertTrue(all(done is False for done in trainer.done[:-1]))
                self.assertTrue(trainer.done[-1])
                self.assertEqual(controller.positive_reward_actions, 1)
                self.assertEqual(len(controller.archive.points), 1)
            finally:
                controller.close()

    def test_v3_policy_controls_path_count_without_consuming_base_rng(self):
        random.seed(9)
        base.random_initial_individual(
            [self.batch], self.path_lib, max_paths=3)
        expected_state = random.getstate()

        trainer = _FakeTrainer()
        trainer.config = SimpleNamespace(candidate_representation="path_v3")
        with tempfile.TemporaryDirectory() as directory:
            controller = self._controller(directory, trainer)
            try:
                random.seed(9)
                individual = controller.initial_individual(
                    [self.batch], self.path_lib, max_paths=3)
                allocations = individual.od_allocations[("A", "D", 4)]
                self.assertEqual(random.getstate(), expected_state)
                self.assertEqual(len(allocations), 3)
                self.assertAlmostEqual(sum(item.share for item in allocations), 1.0)
                count_log = Path(directory) / "initialization_count_candidates.jsonl"
                self.assertEqual(len(count_log.read_text().splitlines()), 3)
            finally:
                controller.close()

    def test_v4_path_control_keeps_base_count_with_v3_features(self):
        random.seed(9)
        base_individual = base.random_initial_individual(
            [self.batch], self.path_lib, max_paths=3)
        expected_state = random.getstate()
        expected_count = len(
            base_individual.od_allocations[("A", "D", 4)])

        trainer = _FakeTrainer()
        trainer.config = SimpleNamespace(
            candidate_representation="path_v3",
            initialization_control="path")
        with tempfile.TemporaryDirectory() as directory:
            controller = self._controller(directory, trainer)
            try:
                random.seed(9)
                individual = controller.initial_individual(
                    [self.batch], self.path_lib, max_paths=3)
                allocations = individual.od_allocations[("A", "D", 4)]
                self.assertEqual(random.getstate(), expected_state)
                self.assertEqual(len(allocations), expected_count)
                count_log = Path(directory) / "initialization_count_candidates.jsonl"
                self.assertEqual(count_log.read_text(), "")
                self.assertEqual(
                    len(trainer.last_observation["candidates"][0]), 50)
            finally:
                controller.close()

    def test_one_path_is_direct_and_creates_no_policy_reward(self):
        trainer = _FakeTrainer(fail_on_choose=True)
        with tempfile.TemporaryDirectory() as directory:
            controller = self._controller(directory, trainer)
            try:
                individual = controller.initial_individual(
                    [self.batch], {("A", "D"): self.paths[:1]}, max_paths=3)
                individual.objectives, individual.feasible = (2.0, 3.0, 4.0), True
                controller.settle_initialization(individual)
                self.assertEqual(trainer.rewards, [])
                self.assertEqual(controller.decision_counts["single_candidate"], 1)
            finally:
                controller.close()

    def test_learned_individual_enters_archive_only_at_terminal_settlement(self):
        trainer = _FakeTrainer()
        with tempfile.TemporaryDirectory() as directory:
            controller = self._controller(directory, trainer)
            try:
                fixed = base.Individual()
                controller.evaluate(
                    lambda ind: (
                        setattr(ind, "objectives", (4.0, 4.0, 4.0)),
                        setattr(ind, "feasible", True)),
                    fixed)
                hv_before = controller.archive.hypervolume
                random.seed(42)
                learned = controller.initial_individual(
                    [self.batch], self.path_lib, max_paths=3)
                controller.evaluate(
                    lambda ind: (
                        setattr(ind, "objectives", (2.0, 3.0, 4.0)),
                        setattr(ind, "feasible", True)),
                    learned)
                self.assertEqual(controller.archive.hypervolume, hv_before)
                controller.settle_initialization(learned)
                self.assertGreater(controller.archive.hypervolume, hv_before)
            finally:
                controller.close()

    def test_v2_rule_prefilter_limits_model_candidates_without_changing_base_rng(self):
        for path, value in zip(self.paths, (9.0, 1.0, 5.0)):
            path.base_cost_per_teu = value
            path.base_emission_per_teu = value
            path.base_travel_time_h = value
        rule = {
            "source": "training-only",
            "configuration": "S0",
            "features": [
                "base_cost_per_teu", "base_emission_per_teu",
                "base_travel_time_h",
            ],
            "minimum": [0.0, 0.0, 0.0],
            "maximum": [10.0, 10.0, 10.0],
            "test_instances_used": False,
            "normalization_digest": "toy-rule",
        }
        random.seed(42)
        base.random_initial_individual([self.batch], self.path_lib, max_paths=2)
        expected_state = random.getstate()

        trainer = _FakeTrainer()
        with tempfile.TemporaryDirectory() as directory:
            controller = InitializationPPOController(
                Path(directory), "run", "scenario", 50, trainer=trainer,
                archive=RewardArchive(self.normalizer),
                reward_config=RewardConfig(), batches=[self.batch],
                generations=2, pop_size=4, method=trainer.name,
                instance={"instance_id": "toy", "K": 1,
                          "deadline_window_alpha": 1.0}, seeds={},
                rule_normalization=rule, rule_top_k=2)
            try:
                random.seed(42)
                learned = controller.initial_individual(
                    [self.batch], self.path_lib, max_paths=2)
                self.assertEqual(random.getstate(), expected_state)
                self.assertLessEqual(
                    len(trainer.last_observation["candidates"]), 2)
                selected_ids = {
                    item.path.path_id
                    for item in learned.od_allocations[("A", "D", 4)]
                }
                self.assertNotIn(1, selected_ids)
                self.assertTrue(controller.summary()["rule_prefilter_enabled"])
            finally:
                controller.close()


def _path_signature(path):
    return tuple(path.nodes), tuple(path.modes)


if __name__ == "__main__":
    unittest.main()
