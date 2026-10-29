import json
from pathlib import Path
import tempfile
import unittest

from audit_learning_components_stage1 import (
    _candidate_file_audit,
    _training_audit,
)


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


class Stage1AuditTest(unittest.TestCase):
    def test_training_uses_policy_reward_events_as_denominator(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write(root / "training_configuration.json", {
                "component": "M", "architecture": "transformer",
                "actor_conditioning": (
                    "candidate-plus-operator-plus-global-before-encoder"),
                "configuration": "S0", "evaluation_mode": "deterministic",
                "ppo": {"dropout": 0.1}, "population": 20,
                "evaluation_budget_per_episode": 400,
            })
            _write(root / "training_summary.json", {
                "completed": True, "component": "M",
                "total_actual_evaluations": 400,
                "episodes": [{
                    "instance": "train-S0-001",
                    "decision_counts": {"model_choice": 3, "no_candidate": 1},
                    "ppo_updates_total": 1,
                    "optimisation_seconds": 4.0,
                    "inference_seconds": 1.0,
                }],
            })
            outcome = root / "episodes/train-S0-001/mutation_outcomes.jsonl"
            outcome.parent.mkdir(parents=True)
            _write(outcome.parent / "configuration.json", {
                "evaluation_mode": "deterministic", "training_size": 1,
            })
            _write(outcome.parent / "ppo_updates.jsonl", {
                "update": 1, "transition_count": 3,
                "optimizer_step_count": 2,
                "early_stopped_on_target_kl": False,
                "first_minibatch_pre_step_logp_abs_error_mean": 0.01,
            })
            outcome.write_text("\n".join(json.dumps(row) for row in (
                {"reward_recorded_for_policy": True, "reward": 0.5},
                {"reward_recorded_for_policy": True, "reward": 0.0},
                {"reward_recorded_for_policy": False, "reward": 2.0},
            )) + "\n", encoding="utf-8")
            result = _training_audit(root, "M")
            self.assertEqual(result["model_action_transition_count"], 3)
            self.assertEqual(result["effective_reward_event_count"], 2)
            self.assertEqual(result["positive_reward_event_count"], 1)
            self.assertEqual(result["positive_reward_fraction"], 0.5)
            self.assertEqual(result["mean_model_actions_per_ppo_update"], 3.0)
            self.assertEqual(result["configured_ppo_dropout"], 0.1)
            self.assertTrue(result["deterministic_configuration_confirmed"])
            self.assertEqual(result["logged_optimizer_step_count"], 2)
            self.assertEqual(
                result["mean_first_minibatch_pre_step_logp_abs_error"], 0.01)

    def test_candidate_audit_measures_nonuniformity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "initialization_candidates.jsonl"
            path.write_text("\n".join(json.dumps(row) for row in (
                {"event_id": "e1", "decision_kind": "model_choice",
                 "selection_probability": 0.8, "chosen": True, "path_id": "p1"},
                {"event_id": "e1", "decision_kind": "model_choice",
                 "selection_probability": 0.2, "chosen": False, "path_id": "p2"},
            )) + "\n", encoding="utf-8")
            metrics, positions, indices = _candidate_file_audit(path, "I")
            self.assertEqual(len(metrics), 1)
            self.assertGreater(metrics[0]["total_variation_from_uniform"], 0.0)
            self.assertEqual(metrics[0]["selected_probability"], 0.8)
            self.assertEqual(positions["path:p1"], 1)
            self.assertEqual(indices[0], 1)

    def test_initialization_decisions_are_not_merged_by_event(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "initialization_candidates.jsonl"
            rows = []
            for batch_id in (1, 2):
                rows.extend((
                    {"event_id": "same-individual", "batch_id": batch_id,
                     "selection_step": 0, "decision_kind": "model_choice",
                     "selection_probability": 0.6, "chosen": True,
                     "path_id": f"p{batch_id}a"},
                    {"event_id": "same-individual", "batch_id": batch_id,
                     "selection_step": 0, "decision_kind": "model_choice",
                     "selection_probability": 0.4, "chosen": False,
                     "path_id": f"p{batch_id}b"},
                ))
            path.write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8")
            metrics, _, _ = _candidate_file_audit(path, "I")
            self.assertEqual(len(metrics), 2)


if __name__ == "__main__":
    unittest.main()
