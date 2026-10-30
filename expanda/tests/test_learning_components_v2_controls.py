from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import baseline_uncertainty as base
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
                self.assertEqual(controller.base_passthrough_events, 1)
                self.assertEqual(controller.positive_reward_actions, 0)
                self.assertEqual(controller.reward_total, 0.0)
            finally:
                controller.close()


if __name__ == "__main__":
    unittest.main()
