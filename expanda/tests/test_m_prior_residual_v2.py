import unittest
from types import SimpleNamespace

from transformer_ppo.config import PPOConfig
from transformer_ppo.observations import (
    joint_balanced_action_prior,
    joint_balanced_candidate_scores,
    parent_proxy_position,
)


def _candidate(batch_id, path_id, cost, emission, time, lateness, on_time):
    return {
        "eligible": True,
        "target": {"batch_id": batch_id},
        "path_id": path_id,
        "batch_quantity": 10.0,
        "replacement_delta_cost_per_teu": cost,
        "replacement_delta_emission_per_teu": emission,
        "replacement_delta_nominal_time_h": time,
        "replacement_max_lateness_h": lateness,
        "replacement_on_time_probability": on_time,
    }


class MPriorResidualV2Tests(unittest.TestCase):
    def test_joint_balanced_prior_argmax_matches_balanced_score(self):
        candidates = [
            _candidate(1, "slow", -2, 3, 4, 4, 0.50),
            _candidate(2, "balanced", -1, -1, -1, 0, 0.95),
            _candidate(3, "dirty", -4, 8, -2, 1, 0.90),
        ]
        eligible = [True, True, True]
        scores = joint_balanced_candidate_scores(candidates, eligible)
        prior = joint_balanced_action_prior(candidates, eligible, 0.25)
        self.assertEqual(max(range(3), key=prior.__getitem__),
                         min(range(3), key=scores.__getitem__))
        self.assertAlmostEqual(sum(prior), 1.0)

    def test_path_v5_joint_prior_configuration_is_frozen(self):
        config = PPOConfig(
            rollout_steps=16,
            gae_horizon=8,
            actor_policy="base_residual",
            residual_logit_limit=0.75,
            candidate_representation="path_v5",
            credit_assignment="operation_terminal",
            mutation_control="replace_path",
            mutation_action_prior="joint_balanced",
            mutation_prior_temperature=0.25,
        )
        config.validate()
        with self.assertRaisesRegex(ValueError, "joint_balanced"):
            PPOConfig(
                actor_policy="direct",
                candidate_representation="path_v5",
                credit_assignment="operation_terminal",
                mutation_control="replace_path",
                mutation_action_prior="joint_balanced",
            ).validate()

    def test_parent_proxy_position_exposes_front_location_without_evaluation(self):
        batch = SimpleNamespace(batch_id=1, quantity=10.0, ET=2.0)

        def individual(cost, emission, time):
            path = SimpleNamespace(
                base_cost_per_teu=cost,
                base_emission_per_teu=emission,
                base_travel_time_h=time,
            )
            allocation = SimpleNamespace(share=1.0, path=path)
            return SimpleNamespace(od_allocations={("A", "B", 1): [allocation]})

        low_cost = individual(1.0, 5.0, 4.0)
        low_emission = individual(5.0, 1.0, 2.0)
        middle = individual(3.0, 3.0, 3.0)
        position = parent_proxy_position(
            middle, [low_cost, low_emission], [batch])
        self.assertEqual(position, [0.5, 0.5, 0.5])


if __name__ == "__main__":
    unittest.main()
