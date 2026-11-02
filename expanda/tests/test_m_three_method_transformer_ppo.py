from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluate_m_balanced_teacher_transformer import _command
from train_m_balanced_teacher_transformer import (
    _limited_rows, _teacher_probabilities,
)


class MThreeMethodTransformerPPOTest(unittest.TestCase):
    def test_development_subset_is_deterministic_and_bounded(self):
        rows = [{"value": value} for value in range(20)]
        first = _limited_rows(rows, 7, 42)
        second = _limited_rows(rows, 7, 42)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 7)
        self.assertEqual(_limited_rows(rows, None, 42), rows)

    def test_soft_teacher_target_prefers_lower_balanced_score(self):
        probabilities = _teacher_probabilities([0.0, 1.0, 2.0], 0.25)
        self.assertAlmostEqual(float(probabilities.sum()), 1.0)
        self.assertGreater(probabilities[0], probabilities[1])
        self.assertGreater(probabilities[1], probabilities[2])
        self.assertGreater(probabilities[0], 0.95)

    def test_three_roles_share_budget_and_use_expected_replace_selectors(self):
        args = SimpleNamespace(
            runner=Path("runner.py"),
            normalization=Path("normalization.json"),
            scenario_count=200,
            ccp_confidence=0.9,
            evaluation_budget=400,
            pop=20,
            gens=1000,
            device="cpu",
            bootstrap_checkpoint=Path("bootstrap.pt"),
            transformer_checkpoint=Path("ppo.pt"),
            minimum_generation=10,
        )
        commands = {
            role: _command(
                args, Path("instance.json"), Path(role), role, 123)
            for role in ("BASE", "HEURISTIC", "TRANSFORMER")
        }
        for command in commands.values():
            self.assertIn("--evaluation-budget", command)
            self.assertEqual(
                command[command.index("--evaluation-budget") + 1], "400")
            self.assertEqual(
                command[command.index("--scenario-count") + 1], "200")
        heuristic = commands["HEURISTIC"]
        transformer = commands["TRANSFORMER"]
        self.assertEqual(
            heuristic[heuristic.index("--m-forced-replace-selector") + 1],
            "joint_balanced")
        self.assertEqual(
            transformer[transformer.index("--m-forced-replace-selector") + 1],
            "model")
        self.assertEqual(
            transformer[
                transformer.index("--m-forced-operator-min-generation") + 1],
            "10")


if __name__ == "__main__":
    unittest.main()
