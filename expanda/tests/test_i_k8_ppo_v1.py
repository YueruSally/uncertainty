from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from experiments.run_i_k8_ppo_selection_v1 import (
    ALL_UPDATES, PPO_UPDATES, method, validation_summary)
from i_k8_scale_policy import FrozenIInitializationRanker
from i_k8_surrogate import CandidateGainTransformer, fit_scaler, pack_rows
from m_operator_offline import digest
from train_i_k8_ppo_v1 import ppo_update


class InitializationK8PPOTests(unittest.TestCase):
    @staticmethod
    def rows(count=16):
        result = []
        for state in range(count):
            result.append({
                "instance": f"train-{state % 4}",
                "trajectory_seed": state,
                "event_index": state,
                "slot_fraction": (state % 20) / 19.0,
                "population_points": [[.2, .3, .4], [.3, .2, .5]],
                "candidate_features": [
                    [float(candidate), float(state), 1.0]
                    for candidate in range(8)],
                "candidate_batch_tokens": [[
                    [float(candidate), float(batch), float(state)]
                    for batch in range(3)] for candidate in range(8)],
                "labels": [{
                    "index": candidate,
                    "feasible": True,
                    "objectives": [1.0 + candidate, 2.0, 3.0],
                    "hv_contribution": .001 * candidate * (1 + state / 100),
                } for candidate in range(8)],
                "simple_selected_index": 1,
            })
        return result

    def test_method_identity_and_checkpoint_set_are_frozen(self):
        self.assertEqual(ALL_UPDATES, (0, 2, 4, 8))
        self.assertEqual(PPO_UPDATES, (2, 4, 8))
        self.assertEqual(method(4), "I_K8_TRANSFORMER_PPO_U004")

    def test_validation_selects_best_true_ppo_checkpoint(self):
        records = [{
            "instance_id": f"validation-{index:03d}",
            "algorithm_seed": 95_000_000 + index,
        } for index in range(20)]
        gains = {0: 0.0, 2: .2, 4: 1.0, 8: .7}
        rows = []
        for record in records:
            for update in ALL_UPDATES:
                rows.append({
                    "instance": record["instance_id"],
                    "strategy": method(update),
                    "algorithm_seed": record["algorithm_seed"],
                    "scenario_digest": f"scenario-{record['instance_id']}",
                    "final_hypervolume": .5 * (1 + gains[update] / 100),
                })
        result = validation_summary(rows, records)
        self.assertEqual(result["selected_update"], 4)
        self.assertAlmostEqual(
            result["selected_validation_mean_percent_vs_supervised"], 1.0)
        self.assertEqual(result["selected_validation_positive_instances"], 20)

    def test_validation_does_not_rename_supervised_as_ppo(self):
        records = [{"instance_id": "validation", "algorithm_seed": 7}]
        gains = {0: 0.0, 2: -.1, 4: -.5, 8: -.2}
        rows = [{
            "instance": "validation",
            "strategy": method(update),
            "algorithm_seed": 7,
            "scenario_digest": "same",
            "final_hypervolume": 1 + gains[update] / 100,
        } for update in ALL_UPDATES]
        result = validation_summary(rows, records)
        self.assertEqual(result["selected_update"], 2)
        self.assertLess(
            result["selected_validation_mean_percent_vs_supervised"], 0)

    def test_frozen_ranker_loads_i_ppo_checkpoint_protocol(self):
        rows = self.rows(1)
        scaler = fit_scaler(rows)
        packed = pack_rows(rows, scaler)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            checkpoints, hashes = [], {}
            for index in range(3):
                model = CandidateGainTransformer(
                    packed[0].shape[-1], packed[1].shape[-1],
                    packed[2].shape[-1], width=16)
                path = root / f"model_{index}.pt"
                torch.save({
                    "protocol": "i-k8-scale-ppo-v1",
                    "training_instances": 200,
                    "width": 16,
                    "model": model.state_dict(),
                    "batch_feature_count": packed[0].shape[-1],
                    "candidate_feature_count": packed[1].shape[-1],
                    "context_feature_count": packed[2].shape[-1],
                }, path)
                checkpoints.append(str(path))
                hashes[str(path)] = digest(path)
            bundle = {
                "protocol": "i-k8-scale-ppo-bundle-v1",
                "selected_size": 200,
                "models": {"200": {
                    "scaler": scaler,
                    "target_scale": .01,
                    "checkpoints": checkpoints,
                    "checkpoint_sha256": hashes,
                }},
            }
            ranker = FrozenIInitializationRanker(bundle, digest)
            self.assertEqual(len(ranker.models), 3)
            self.assertIn(ranker.select(rows[0]), range(8))

    def test_contextual_bandit_ppo_performs_actor_and_critic_steps(self):
        rows = self.rows()
        scaler = fit_scaler(rows)
        packed = pack_rows(rows[:1], scaler)
        models = [CandidateGainTransformer(
            packed[0].shape[-1], packed[1].shape[-1],
            packed[2].shape[-1], width=16) for _ in range(3)]
        critic = torch.nn.Sequential(
            torch.nn.Linear(packed[2].shape[-1], 8), torch.nn.Tanh(),
            torch.nn.Linear(8, 1))
        actor_optimizer = torch.optim.Adam([
            parameter for model in models for parameter in model.parameters()],
            lr=1e-5)
        critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-4)
        report = ppo_update(
            models, critic, actor_optimizer, critic_optimizer,
            rows, scaler, .01, np.random.default_rng(3))
        self.assertEqual(report["states"], 16)
        self.assertEqual(report["actor_optimizer_steps"], 4)
        self.assertEqual(report["critic_optimizer_steps"], 4)
        self.assertEqual(sum(report[key] for key in (
            "positive_rewards", "zero_rewards", "negative_rewards")), 16)


if __name__ == "__main__":
    unittest.main()
