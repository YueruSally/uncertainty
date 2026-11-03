from pathlib import Path
import unittest

from transformer_ppo.config import PPOConfig
from transformer_ppo.controller import PPOController


def _candidate(batch_id, path_id, cost, emission, time, lateness, on_time):
    from learning_mutation import MutationTarget

    return {
        "eligible": True,
        "target": MutationTarget(batch_id, replacement_path_id=path_id),
        "path_id": path_id,
        "batch_quantity": 10.0,
        "replacement_delta_cost_per_teu": cost,
        "replacement_delta_emission_per_teu": emission,
        "replacement_delta_nominal_time_h": time,
        "replacement_max_lateness_h": lateness,
        "replacement_on_time_probability": on_time,
    }


class MPairedHeuristicV4Tests(unittest.TestCase):
    def test_paired_credit_accepts_only_joint_replace_configuration(self):
        PPOConfig(
            rollout_steps=16,
            gae_horizon=8,
            actor_policy="base_residual",
            candidate_representation="path_v5",
            credit_assignment="paired_heuristic_terminal",
            mutation_control="replace_path",
            mutation_action_prior="joint_balanced",
        ).validate()

    def test_reference_index_is_exact_balanced_action(self):
        candidates = [
            _candidate(1, "cheap-dirty", -4, 8, 2, 3, 0.70),
            _candidate(2, "balanced", -1, -1, -1, 0, 0.98),
            _candidate(3, "fast-costly", 6, -2, -4, 1, 0.90),
        ]
        self.assertEqual(
            PPOController._joint_balanced_candidate_index(candidates), 1)

    def test_v4_runner_freezes_equal_ccp100_test_parameters(self):
        script = (
            Path(__file__).resolve().parents[1]
            / "run_m_ccp100_paired_heuristic_v4.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("--scenario-count 100", script)
        self.assertIn("--ccp-confidence 0.90", script)
        self.assertIn("--credit-assignment paired_heuristic_terminal", script)
        self.assertGreaterEqual(script.count("--evaluation-budget 800"), 2)
        self.assertGreaterEqual(script.count("--pop 20"), 2)
        self.assertIn("equal_test_parameters: true", script)


if __name__ == "__main__":
    unittest.main()
