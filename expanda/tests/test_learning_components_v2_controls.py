from pathlib import Path
from unittest.mock import patch
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import baseline_uncertainty as base
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


if __name__ == "__main__":
    unittest.main()
