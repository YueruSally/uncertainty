from pathlib import Path
from unittest.mock import patch
import tempfile
import unittest

import baseline_uncertainty as base
from learning_mutation import MutationTarget
from multi_instance_policy import Decision
from transformer_ppo.archive import (
    ObjectiveNormalizer, RewardArchive, RewardConfig,
)
from transformer_ppo.config import PPOConfig
from transformer_ppo.controller import PPOController


class _Trainer:
    name = "fake-operator-ppo"
    component = "M"
    updates = 0
    feature_construction_seconds = 0.0
    ppo_update_seconds = 0.0
    update_records = ()
    training = True

    def __init__(self):
        self.config = PPOConfig(
            rollout_steps=8,
            gae_horizon=8,
            actor_policy="base_residual",
            candidate_representation="path_v3",
            credit_assignment="generation_terminal",
            mutation_control="operator",
        )
        self.observation_provider = None
        self.pending = None
        self.sequence_depth = 0
        self.sequence_policy_actions = 0
        self.outcomes = []

    def begin_sequence(self):
        self.sequence_depth = 1
        self.sequence_policy_actions = 0

    def observe_outcome(self, reward, done=False):
        self.outcomes.append((float(reward), bool(done)))
        self.pending = None

    def finish(self):
        return None


class MOperatorSchedulerV6Tests(unittest.TestCase):
    def test_operator_control_has_small_stable_action_space(self):
        PPOConfig(
            rollout_steps=8,
            gae_horizon=8,
            actor_policy="base_residual",
            candidate_representation="path_v3",
            credit_assignment="generation_terminal",
            mutation_control="operator",
        ).validate()
        with self.assertRaisesRegex(ValueError, "operator control requires"):
            PPOConfig(mutation_control="operator").validate()

    def test_operator_policy_executes_balanced_replace_target(self):
        trainer = _Trainer()
        normalizer = ObjectiveNormalizer(
            (0.0, 0.0, 0.0), (10.0, 10.0, 10.0),
            reference=(1.1, 1.1, 1.1))
        batch = base.Batch(1, "A", "D", 10.0, 0.0, 24.0)
        individual = base.Individual()
        target = MutationTarget(1, replacement_path_id="77")

        def choose_operator(*_args, **_kwargs):
            trainer.pending = {"action": 4}
            trainer.sequence_policy_actions += 1
            candidates = [
                {"candidate_operator": name, "eligible": True}
                for name in ("add", "del", "mod", "mode", "replace")
            ]
            return Decision(
                4, [0.1, 0.1, 0.1, 0.1, 0.6], [None] * 5,
                {"decision_kind": "model_choice"}), candidates

        selected = {}

        def apply(_ind, operator, selected_batch, *_args, target=None, **_kwargs):
            selected.update(
                operator=operator, batch=selected_batch, target=target)
            return True

        with tempfile.TemporaryDirectory() as directory:
            controller = PPOController(
                Path(directory), "run", "scenario", 20,
                trainer=trainer, archive=RewardArchive(normalizer),
                reward_config=RewardConfig(), batches=[batch],
                generations=2, pop_size=2, method=trainer.name,
                instance={"instance_id": "toy"},
                seeds={"algorithm": 7})
            try:
                controller.generation = 0
                controller.phase = "offspring"
                with patch.object(
                        controller, "_choose_operator",
                        side_effect=choose_operator), patch.object(
                        controller, "_balanced_replace_target",
                        return_value=(batch, target, {
                            "selector": "joint_balanced"})), patch(
                        "transformer_ppo.controller.base.apply_mutation_op",
                        side_effect=apply), patch(
                        "transformer_ppo.controller.base.repair_missing_allocations"), patch(
                        "transformer_ppo.controller.base.evaluate_individual"):
                    result = controller.mutate(
                        individual, [batch], {}, {}, {}, [], 0.0, 0.0)
                self.assertEqual(result, ("replace", True))
                self.assertEqual(selected["operator"], "replace")
                self.assertIs(selected["target"], target)
                self.assertEqual(trainer.outcomes, [(0.0, False)])
                self.assertEqual(trainer.sequence_depth, 1)
            finally:
                controller.close()


if __name__ == "__main__":
    unittest.main()
