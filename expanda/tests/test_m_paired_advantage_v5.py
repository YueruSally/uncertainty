import json
from pathlib import Path
import tempfile
import unittest

from train_m_paired_advantage_transformer import (
    _restrict_dataset_top_k, _stratified_subset, load_dataset,
)


class MPairedAdvantageV5Tests(unittest.TestCase):
    def test_quick_subset_balances_override_and_fallback_labels(self):
        rows = [
            {"override_target": index < 10, "index": index}
            for index in range(20)
        ]
        selected = _stratified_subset(rows, 8, 17)
        self.assertEqual(len(selected), 8)
        self.assertEqual(sum(row["override_target"] for row in selected), 4)

    def test_top_k_drops_unreachable_positive_label(self):
        rows = [{
            "observation": {
                "eligible": [True, True, True],
                "action_prior": [0.6, 0.3, 0.1],
            },
            "target_action": 2,
            "alternative_action": 0,
            "heuristic_action": 0,
            "override_target": True,
            "pairwise_target": True,
        }]

        def restrict(prior, eligible, top_k, required_index):
            return [True, True, False], [2 / 3, 1 / 3, 0.0]

        restricted, dropped = _restrict_dataset_top_k(rows, 2, restrict)
        self.assertEqual(restricted, [])
        self.assertEqual(dropped, 1)

    def test_positive_selected_action_wins_and_negative_falls_back(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / "episodes" / "train-S0-002"
            run.mkdir(parents=True)
            states = []
            outcomes = []
            for event, advantage in ((1, 0.25), (2, -0.10), (3, 0.0)):
                states.append({
                    "instance_id": "train-S0-002",
                    "generation": 10,
                    "evaluation": 200 + event,
                    "mixture_event_index": event,
                    "operator": "replace",
                    "observation": {
                        "eligible": [True, True, False],
                    },
                })
                outcomes.append({
                    "mixture_event_index": event,
                    "paired_heuristic_reward": {
                        "selected_candidate_index": 1,
                        "heuristic_candidate_index": 0,
                        "selected_minus_heuristic_hv": advantage,
                        "label_source": "archive",
                    },
                })
            (run / "mutation_policy_states.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in states),
                encoding="utf-8")
            (run / "mutation_outcomes.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in outcomes),
                encoding="utf-8")
            rows = load_dataset(temporary)
        self.assertEqual([row["target_action"] for row in rows], [1, 0, 0])
        self.assertEqual(
            [row["override_target"] for row in rows], [True, False, False])

    def test_v5_runner_keeps_the_test_budget_equal(self):
        script = (
            Path(__file__).resolve().parents[1]
            / "run_m_ccp100_advantage_pretrain_v5.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("--scenario-count 100", script)
        self.assertIn("--ccp-confidence 0.90", script)
        self.assertIn("--evaluation-budget 800", script)
        self.assertIn("--pop 20", script)
        self.assertGreaterEqual(
            script.count("--residual-logit-limit 2.0"), 2)
        self.assertIn(".validation.override_recall >= 0.10", script)
        self.assertIn(".validation.fallback_accuracy >= 0.80", script)
        self.assertIn("equal_test_parameters: true", script)
        self.assertIn("$model.mean_paired_percent > $heuristic.mean_paired_percent", script)


if __name__ == "__main__":
    unittest.main()
