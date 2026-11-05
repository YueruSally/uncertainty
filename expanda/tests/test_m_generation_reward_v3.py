from pathlib import Path
from types import SimpleNamespace
import unittest

from transformer_ppo.archive import ObjectiveNormalizer, RewardConfig
from transformer_ppo.config import PPOConfig
from transformer_ppo.controller import PPOController


class _Trainer:
    def __init__(self):
        self.sequence_depth = 0
        self.sequence_policy_actions = 0
        self.settled = []

    def begin_sequence(self):
        if self.sequence_depth:
            raise RuntimeError("nested sequence")
        self.sequence_depth = 1
        self.sequence_policy_actions = 0

    def end_sequence(self, reward, done=False):
        self.settled.append((float(reward), bool(done)))
        self.sequence_depth = 0
        self.sequence_policy_actions = 0


class MGenerationRewardV3Tests(unittest.TestCase):
    def test_generation_terminal_is_valid_only_for_joint_replace_config(self):
        PPOConfig(
            rollout_steps=16,
            gae_horizon=8,
            actor_policy="base_residual",
            candidate_representation="path_v5",
            credit_assignment="generation_terminal",
            mutation_control="replace_path",
            mutation_action_prior="joint_balanced",
        ).validate()
        with self.assertRaisesRegex(ValueError, "replace_path requires"):
            PPOConfig(
                candidate_representation="path_v5",
                credit_assignment="continuous",
                mutation_control="replace_path",
            ).validate()

    def test_generation_reward_settles_after_selected_population_hv(self):
        controller = object.__new__(PPOController)
        controller.archive = SimpleNamespace(normalizer=ObjectiveNormalizer(
            minimum=(0.0, 0.0, 0.0),
            maximum=(1.0, 1.0, 1.0),
            reference=(1.1, 1.1, 1.1),
        ))
        controller.reward_config = RewardConfig()
        controller.trainer = _Trainer()
        controller.generation = 4
        controller.run_id = "run"
        controller.instance = {"instance_id": "instance"}
        controller._selected_population_hv = 0.700
        controller._generation_reward_sequence_generation = None
        controller.reward_total = 0.0
        controller.positive_reward_actions = 0
        controller.negative_reward_actions = 0
        controller.generation_reward_events = 0
        controller.positive_generation_rewards = 0
        controller.negative_generation_rewards = 0
        rows = []
        controller.append = lambda name, row: rows.append((name, row))

        controller._begin_generation_reward_sequence()
        controller.trainer.sequence_policy_actions = 3
        population = [SimpleNamespace(
            feasible=True, objectives=(0.2, 0.2, 0.2))]
        controller._settle_generation_reward(
            population, "environmental_selection")

        # One normalized point at (0.2, 0.2, 0.2) has HV 0.9^3 = 0.729.
        self.assertAlmostEqual(controller._selected_population_hv, 0.729)
        self.assertAlmostEqual(controller.trainer.settled[0][0], 2.9)
        self.assertTrue(controller.trainer.settled[0][1])
        self.assertEqual(controller.generation_reward_events, 1)
        self.assertEqual(controller.positive_generation_rewards, 1)
        self.assertEqual(rows[0][0], "mutation_generation_rewards")
        self.assertEqual(rows[0][1]["policy_actions"], 3)

    def test_v3_runner_freezes_ccp100_and_equal_test_budget(self):
        script = (
            Path(__file__).resolve().parents[1]
            / "run_m_ccp100_generation_reward_v3.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("--scenario-count 100", script)
        self.assertIn("--ccp-confidence 0.90", script)
        self.assertIn("--credit-assignment generation_terminal", script)
        self.assertGreaterEqual(script.count("--evaluation-budget 800"), 2)
        self.assertGreaterEqual(script.count("--pop 20"), 2)
        self.assertIn("equal_test_parameters: true", script)


if __name__ == "__main__":
    unittest.main()
