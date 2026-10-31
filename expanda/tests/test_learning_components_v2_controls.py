from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import random
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import baseline_uncertainty as base
import numpy as np
from multi_instance_policy import Decision
from transformer_ppo.archive import ObjectiveNormalizer, RewardArchive, RewardConfig
from transformer_ppo.controller import PPOController
from transformer_ppo.v2_controls import (
    private_seed,
    validate_model_choice_probability,
)


class _Trainer:
    name = "fake-ppo-m-masked"
    component = "M"
    updates = 0
    pending = None
    feature_construction_seconds = 0.0
    ppo_update_seconds = 0.0
    update_records = ()

    def __init__(self):
        self.observation_provider = None
        self.choose_calls = 0

    def choose(self, candidates, context=None):
        self.choose_calls += 1
        raise AssertionError("zero model probability must not call PPO")

    def finish(self):
        return None


class V2ControlTests(unittest.TestCase):
    def test_probability_validation_and_private_seed(self):
        self.assertEqual(validate_model_choice_probability(0.4), 0.4)
        self.assertEqual(private_seed("M", "run", 7), private_seed("M", "run", 7))
        self.assertNotEqual(private_seed("M", "run", 7), private_seed("M", "run", 8))
        with self.assertRaises(ValueError):
            validate_model_choice_probability(-0.1)
        with self.assertRaises(ValueError):
            validate_model_choice_probability(1.1)

    def test_m_zero_probability_delegates_to_original_base_mutation(self):
        trainer = _Trainer()
        normalizer = ObjectiveNormalizer(
            (0.0, 0.0, 0.0), (10.0, 10.0, 10.0),
            reference=(1.1, 1.1, 1.1))
        batch = base.Batch(1, "A", "D", 10.0, 0.0, 24.0)
        individual = base.Individual()
        with tempfile.TemporaryDirectory() as directory:
            controller = PPOController(
                Path(directory), "run", "scenario", 20,
                trainer=trainer, archive=RewardArchive(normalizer),
                reward_config=RewardConfig(), batches=[batch],
                generations=2, pop_size=2, method=trainer.name,
                instance={"instance_id": "toy"}, seeds={"algorithm": 7},
                model_choice_probability=0.0)
            try:
                with patch(
                        "transformer_ppo.controller.base.mutate_fixed",
                        return_value=("add", False)) as original:
                    result = controller.mutate(
                        individual, [batch], {}, {}, {}, [], 0.0, 0.0)
                self.assertEqual(result, ("add", False))
                self.assertEqual(original.call_count, 1)
                self.assertEqual(trainer.choose_calls, 0)
                self.assertEqual(
                    controller.decision_counts["base_mixture_choice"], 1)
                self.assertTrue(controller.controls_mutation)
            finally:
                controller.close()

    def test_m_continuation_seed_restarts_python_and_numpy_after_target(self):
        trainer = _Trainer()
        normalizer = ObjectiveNormalizer(
            (0.0, 0.0, 0.0), (10.0, 10.0, 10.0),
            reference=(1.1, 1.1, 1.1))
        batch = base.Batch(1, "A", "D", 10.0, 0.0, 24.0)
        individual = base.Individual()
        with tempfile.TemporaryDirectory() as directory:
            controller = PPOController(
                Path(directory), "run", "scenario", 20,
                trainer=trainer, archive=RewardArchive(normalizer),
                reward_config=RewardConfig(), batches=[batch],
                generations=2, pop_size=2, method=trainer.name,
                instance={"instance_id": "toy"}, seeds={"algorithm": 7},
                model_choice_probability=0.0,
                continuation_event=1, continuation_seed=31001)
            try:
                with patch(
                        "transformer_ppo.controller.base.mutate_fixed",
                        return_value=("add", False)):
                    controller.mutate(
                        individual, [batch], {}, {}, {}, [], 0.0, 0.0)
                self.assertAlmostEqual(
                    random.random(), random.Random(31001).random())
                self.assertAlmostEqual(
                    np.random.random(),
                    np.random.RandomState(31001).random_sample())
                self.assertEqual(controller.continuation_reseed_count, 1)
                summary = controller.summary()
                self.assertEqual(summary["continuation_event"], 1)
                self.assertEqual(summary["continuation_seed"], 31001)
            finally:
                controller.close()

    def test_m_continuation_event_and_seed_are_atomic(self):
        trainer = _Trainer()
        normalizer = ObjectiveNormalizer(
            (0.0, 0.0, 0.0), (10.0, 10.0, 10.0),
            reference=(1.1, 1.1, 1.1))
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "supplied together"):
                PPOController(
                    Path(directory), "run", "scenario", 20,
                    trainer=trainer, archive=RewardArchive(normalizer),
                    reward_config=RewardConfig(), batches=[], generations=2,
                    pop_size=2, method=trainer.name,
                    instance={"instance_id": "toy"},
                    seeds={"algorithm": 7}, continuation_event=1)

    def test_forced_m_operator_keeps_base_batch_and_internal_target(self):
        trainer = _Trainer()
        normalizer = ObjectiveNormalizer(
            (0.0, 0.0, 0.0), (10.0, 10.0, 10.0),
            reference=(1.1, 1.1, 1.1))
        batch = base.Batch(1, "A", "D", 10.0, 0.0, 24.0)
        individual = base.Individual()

        def apply(_ind, operator, selected_batch, *_args, target=None,
                  **_kwargs):
            self.assertEqual(operator, "replace")
            self.assertIs(selected_batch, batch)
            self.assertIsNone(target)
            return True

        operator_target = {
            "target": {}, "eligible": True,
            "selection_probability": 1.0,
        }
        with tempfile.TemporaryDirectory() as directory:
            controller = PPOController(
                Path(directory), "run", "scenario", 20,
                trainer=trainer, archive=RewardArchive(normalizer),
                reward_config=RewardConfig(), batches=[batch],
                generations=2, pop_size=2, method=trainer.name,
                instance={"instance_id": "toy"}, seeds={"algorithm": 7},
                model_choice_event=1, forced_operator="replace",
                continuation_event=1, continuation_seed=34001)
            try:
                with patch(
                        "transformer_ppo.controller.candidates_for",
                        return_value=[operator_target]), patch(
                        "transformer_ppo.controller.base.sample_operator",
                        return_value="replace"), patch(
                        "transformer_ppo.controller.base.apply_mutation_op",
                        side_effect=apply) as apply_mock, patch(
                        "transformer_ppo.controller.base.repair_missing_allocations"
                        ) as repair_mock, patch(
                        "transformer_ppo.controller.base.evaluate_individual"
                        ) as evaluate_mock:
                    result = controller.mutate(
                        individual, [batch], {}, {}, {}, [], 0.0, 0.0)
                self.assertEqual(result, ("replace", True))
                self.assertEqual(apply_mock.call_count, 1)
                self.assertEqual(repair_mock.call_count, 1)
                self.assertEqual(evaluate_mock.call_count, 1)
                self.assertEqual(trainer.choose_calls, 0)
                self.assertEqual(controller.continuation_reseed_count, 1)
                self.assertEqual(controller.summary()["forced_operator"], "replace")
            finally:
                controller.close()

    def test_forced_m_operator_generation_schedule(self):
        trainer = _Trainer()
        normalizer = ObjectiveNormalizer(
            (0.0, 0.0, 0.0), (10.0, 10.0, 10.0),
            reference=(1.1, 1.1, 1.1))
        with tempfile.TemporaryDirectory() as directory:
            controller = PPOController(
                Path(directory), "run", "scenario", 20,
                trainer=trainer, archive=RewardArchive(normalizer),
                reward_config=RewardConfig(), batches=[], generations=20,
                pop_size=2, method=trainer.name,
                instance={"instance_id": "toy"}, seeds={"algorithm": 7},
                forced_operator="replace",
                forced_operator_min_generation=15,
                forced_operator_probability=1.0)
            try:
                controller.generation = 14
                self.assertEqual(controller._next_control_branch(), (False, 1))
                controller.generation = 15
                self.assertEqual(controller._next_control_branch(), (True, 2))
                summary = controller.summary()
                self.assertEqual(
                    summary["forced_operator_min_generation"], 15)
                self.assertEqual(summary["forced_operator_probability"], 1.0)
            finally:
                controller.close()

    def test_forced_m_operator_positive_probe_gates_later_actions(self):
        trainer = _Trainer()
        normalizer = ObjectiveNormalizer(
            (0.0, 0.0, 0.0), (10.0, 10.0, 10.0),
            reference=(1.1, 1.1, 1.1))
        with tempfile.TemporaryDirectory() as directory:
            controller = PPOController(
                Path(directory), "run", "scenario", 20,
                trainer=trainer, archive=RewardArchive(normalizer),
                reward_config=RewardConfig(), batches=[], generations=20,
                pop_size=2, method=trainer.name,
                instance={"instance_id": "toy"}, seeds={"algorithm": 7},
                forced_operator="replace",
                forced_operator_min_generation=15,
                forced_operator_probability=1.0,
                forced_operator_positive_probe=True)
            try:
                controller.generation = 15
                self.assertEqual(controller._next_control_branch(), (True, 1))
                controller.forced_operator_probe_completed = True
                controller.forced_operator_probe_positive = False
                self.assertEqual(controller._next_control_branch(), (False, 2))
                controller.forced_operator_probe_positive = True
                self.assertEqual(controller._next_control_branch(), (True, 3))
            finally:
                controller.close()

    def test_reversible_m_operator_probe_falls_back_to_base(self):
        trainer = _Trainer()
        normalizer = ObjectiveNormalizer(
            (0.0, 0.0, 0.0), (10.0, 10.0, 10.0),
            reference=(1.1, 1.1, 1.1))
        batch = base.Batch(1, "A", "D", 10.0, 0.0, 24.0)
        individual = base.Individual()
        path = base.Path(1, "A", "D", ["A", "D"], ["road"], [],
                         1.0, 1.0, 1.0)
        operator_target = {
            "target": {}, "eligible": True,
            "selection_probability": 1.0,
        }
        with tempfile.TemporaryDirectory() as directory:
            controller = PPOController(
                Path(directory), "run", "scenario", 20,
                trainer=trainer, archive=RewardArchive(normalizer),
                reward_config=RewardConfig(), batches=[batch], generations=20,
                pop_size=2, method=trainer.name,
                instance={"instance_id": "toy"}, seeds={"algorithm": 7},
                forced_operator="replace",
                forced_operator_min_generation=15,
                forced_operator_probability=1.0,
                forced_operator_positive_probe=True,
                forced_operator_reversible_probe=True)
            controller.archive.observe((1.0, 1.0, 1.0), True, "baseline")
            controller.generation = 15

            def apply(probe, *_args, **_kwargs):
                probe.od_allocations[("A", "D", 1)] = [
                    base.PathAllocation(path, 1.0)]
                return True

            def evaluate(probe, *_args, **_kwargs):
                probe.objectives = (2.0, 2.0, 2.0)
                probe.feasible = True

            def fallback(replayed, *_args, **_kwargs):
                self.assertEqual(replayed.od_allocations, {})
                self.assertEqual(
                    replayed.objectives,
                    (float("inf"), float("inf"), float("inf")))
                return "replace", False

            try:
                with patch(
                        "transformer_ppo.controller.candidates_for",
                        return_value=[operator_target]), patch(
                        "transformer_ppo.controller.base.sample_operator",
                        return_value="replace"), patch(
                        "transformer_ppo.controller.base.apply_mutation_op",
                        side_effect=apply), patch(
                        "transformer_ppo.controller.base.repair_missing_allocations"
                        ), patch(
                        "transformer_ppo.controller.base.evaluate_individual",
                        side_effect=evaluate), patch(
                        "transformer_ppo.controller.base.mutate_fixed",
                        side_effect=fallback) as fallback_mock:
                    result = controller.mutate(
                        individual, [batch], {}, {}, {}, [], 0.0, 0.0)
                self.assertEqual(result, ("replace", False))
                self.assertEqual(fallback_mock.call_count, 1)
                self.assertTrue(controller.forced_operator_probe_completed)
                self.assertFalse(controller.forced_operator_probe_positive)
                self.assertEqual(controller.forced_operator_probe_delta_hv, 0.0)
                self.assertEqual(controller.forced_operator_probe_fallbacks, 1)
                self.assertEqual(individual.od_allocations, {})
            finally:
                controller.close()

    def test_reversible_m_operator_probe_retry_schedule(self):
        trainer = _Trainer()
        normalizer = ObjectiveNormalizer(
            (0.0, 0.0, 0.0), (10.0, 10.0, 10.0),
            reference=(1.1, 1.1, 1.1))
        with tempfile.TemporaryDirectory() as directory:
            controller = PPOController(
                Path(directory), "run", "scenario", 20,
                trainer=trainer, archive=RewardArchive(normalizer),
                reward_config=RewardConfig(), batches=[], generations=20,
                pop_size=2, method=trainer.name,
                instance={"instance_id": "toy"}, seeds={"algorithm": 7},
                forced_operator="replace",
                forced_operator_min_generation=15,
                forced_operator_probability=1.0,
                forced_operator_positive_probe=True,
                forced_operator_reversible_probe=True,
                forced_operator_probe_max_attempts=2,
                forced_operator_probe_retry_events=5)
            controller.generation = 15
            try:
                self.assertEqual(controller._next_control_branch(), (True, 1))
                controller.forced_operator_probe_attempts = 1
                controller.forced_operator_probe_completed = True
                controller.forced_operator_probe_positive = False
                controller.forced_operator_probe_next_event = 7
                for event in range(2, 7):
                    self.assertEqual(
                        controller._next_control_branch(), (False, event))
                self.assertEqual(controller._next_control_branch(), (True, 7))
                controller.forced_operator_probe_attempts = 2
                self.assertEqual(controller._next_control_branch(), (False, 8))
                controller.forced_operator_probe_positive = True
                self.assertEqual(controller._next_control_branch(), (True, 9))
            finally:
                controller.close()

    def test_replace_path_nonreplace_gate_uses_exact_base_passthrough(self):
        class ReplacePathTrainer(_Trainer):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(
                    mutation_control="replace_path",
                    credit_assignment="operation_terminal")

            def choose(self, candidates, context=None):
                self.choose_calls += 1
                self.asserted_candidates = candidates
                return Decision(
                    0, [1.0], [None],
                    {"decision_kind": "single_candidate",
                     "fallback": None, "inference_seconds": 0.0})

        trainer = ReplacePathTrainer()
        normalizer = ObjectiveNormalizer(
            (0.0, 0.0, 0.0), (10.0, 10.0, 10.0),
            reference=(1.1, 1.1, 1.1))
        batch = base.Batch(1, "A", "D", 10.0, 0.0, 24.0)
        individual = base.Individual()

        def exact_base_apply(_ind, op, selected_batch, *_args, target=None,
                             **_kwargs):
            self.assertEqual(op, "add")
            self.assertIs(selected_batch, batch)
            self.assertIsNone(target)
            return True

        with tempfile.TemporaryDirectory() as directory:
            controller = PPOController(
                Path(directory), "run", "scenario", 20,
                trainer=trainer, archive=RewardArchive(normalizer),
                reward_config=RewardConfig(), batches=[batch],
                generations=2, pop_size=2, method=trainer.name,
                instance={"instance_id": "toy"}, seeds={"algorithm": 7})
            try:
                with patch(
                        "multi_instance_logging.base.sample_operator",
                        return_value="add"), patch(
                        "multi_instance_logging.base.apply_mutation_op",
                        side_effect=exact_base_apply), patch(
                        "multi_instance_logging.base.repair_missing_allocations"
                        ) as base_repair, patch(
                        "multi_instance_logging.base.evaluate_individual"
                        ) as base_evaluate, patch(
                        "multi_instance_logging.repair_after_mutation",
                        side_effect=AssertionError(
                            "learned repair must not run for Base passthrough")):
                    result = controller.mutate(
                        individual, [batch], {("A", "D"): []}, {}, {}, [],
                        0.0, 0.0)
                self.assertEqual(result, ("add", True))
                self.assertEqual(trainer.choose_calls, 1)
                self.assertTrue(trainer.asserted_candidates[0]["base_passthrough"])
                self.assertEqual(base_repair.call_count, 1)
                self.assertEqual(base_evaluate.call_count, 1)
                self.assertIsNone(controller.pending[-1][1]["evaluation_after"])
                self.assertEqual(controller.base_passthrough_events, 1)
                self.assertEqual(controller.positive_reward_actions, 0)
                self.assertEqual(controller.reward_total, 0.0)
            finally:
                controller.close()


if __name__ == "__main__":
    unittest.main()
