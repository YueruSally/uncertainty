from pathlib import Path
import tempfile
import unittest

import torch

from experiments.run_m_prescreen_ppo_selection_v1 import (
    ALL_UPDATES, PPO_UPDATES, method, validation_summary)
from m_operator_offline import digest
from m_prescreen_scale_policy import (FrozenScaleRanker,
    ScaleSearchPolicy, is_transformer_method)
from m_prescreen_surrogate import CandidateGainTransformer
from train_m_prescreen_ppo_v1 import ppo_update


class PrescreenPPOTests(unittest.TestCase):
    def test_ppo_method_identity_is_accepted_as_transformer(self):
        self.assertEqual(ALL_UPDATES, (0, 2, 4, 8))
        self.assertEqual(PPO_UPDATES, (2, 4, 8))
        self.assertEqual(method(4), "TRANSFORMER_K8_PPO_U004")
        self.assertTrue(is_transformer_method(method(8)))
        predictor = object()
        policy = ScaleSearchPolicy("fixture", 1, method(2), predictor)
        self.assertIs(policy.predictor, predictor)

    @staticmethod
    def rows(gains):
        records = [{"instance_id": f"validation-{index:03d}"}
                   for index in range(20)]
        rows = []
        for record in records:
            name = record["instance_id"]
            pairing = {"instance": name, "seed": 9}
            for update in ALL_UPDATES:
                baseline = 0.5
                rows.append({"instance": name, "strategy": method(update),
                    "final_hypervolume": baseline*(1+gains[update]/100),
                    "pairing": pairing})
        return rows, records

    def test_validation_selects_best_true_ppo_checkpoint(self):
        rows, records = self.rows({0: 0, 2: .2, 4: 1.0, 8: .7})
        result = validation_summary(rows, records)
        self.assertEqual(result["selected_update"], 4)
        self.assertAlmostEqual(
            result["selected_validation_mean_percent_vs_supervised"], 1.0)
        self.assertEqual(result["selected_validation_positive_instances"], 20)

    def test_validation_never_renames_supervised_as_ppo(self):
        rows, records = self.rows({0: 0, 2: -.1, 4: -.5, 8: -.2})
        result = validation_summary(rows, records)
        self.assertEqual(result["selected_update"], 2)
        self.assertLess(
            result["selected_validation_mean_percent_vs_supervised"], 0)

    def test_frozen_ranker_loads_ppo_checkpoint_protocol(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); checkpoints = []; hashes = {}
            for index in range(3):
                model = CandidateGainTransformer(4, 3, 14, width=16)
                path = root/f"model_{index}.pt"
                torch.save({"protocol": "m-prescreen-scale-ppo-v1",
                    "training_instances": 200, "width": 16,
                    "model": model.state_dict(), "batch_feature_count": 4,
                    "candidate_feature_count": 3,
                    "context_feature_count": 14}, path)
                checkpoints.append(str(path)); hashes[str(path)] = digest(path)
            bundle = {"models": {"200": {
                "scaler": {"clip": 10.0}, "target_scale": 1.0,
                "checkpoints": checkpoints, "checkpoint_sha256": hashes}}}
            ranker = FrozenScaleRanker(bundle, digest)
            self.assertEqual(len(ranker.models), 3)

    def test_contextual_bandit_ppo_performs_actor_and_critic_steps(self):
        class Normalizer:
            @staticmethod
            def transform(values):
                return values

        rows = []
        for state in range(16):
            labels = [{"feasible": True,
                "objectives": [1.0+candidate, 2.0, 3.0],
                "hv_contribution": .001*candidate*(1+state/100)}
                for candidate in range(8)]
            rows.append({"generation": 10+state, "evaluation": 220+state,
                "population_points": [[.2, .3, .4], [.3, .2, .5]],
                "candidate_features": [[candidate, 0, 1]
                                       for candidate in range(8)],
                "candidate_batch_tokens": [
                    [[candidate, 0], [candidate, 1]]
                    for candidate in range(8)],
                "labels": labels})
        scaler = {"clip": 10.0,
            "batch": {"mean": [0, 0], "std": [1, 1]},
            "candidate": {"mean": [0, 0, 0], "std": [1, 1, 1]},
            "context": {"mean": [0]*14, "std": [1]*14},
            "target": {"mean": [0, 0, 0], "std": [1, 1, 1]}}
        models = [CandidateGainTransformer(2, 3, 14, width=16)
                  for _ in range(3)]
        critic = torch.nn.Sequential(torch.nn.Linear(14, 8), torch.nn.Tanh(),
                                     torch.nn.Linear(8, 1))
        actor_optimizer = torch.optim.Adam(
            [parameter for model in models for parameter in model.parameters()],
            lr=1e-5)
        critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-4)
        report = ppo_update(models, critic, actor_optimizer,
            critic_optimizer, rows, scaler, Normalizer(), .01,
            __import__("numpy").random.default_rng(3))
        self.assertEqual(report["states"], 16)
        self.assertEqual(report["actor_optimizer_steps"], 4)
        self.assertEqual(report["critic_optimizer_steps"], 4)
        self.assertEqual(sum(report[key] for key in (
            "positive_rewards", "zero_rewards", "negative_rewards")), 16)


if __name__ == "__main__":
    unittest.main()
