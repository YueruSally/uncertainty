import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analyze_s0_signal_pilots import analyze  # noqa: E402


class AnalyzeS0SignalPilotsTests(unittest.TestCase):
    def _training(self, root, component):
        root.mkdir()
        (root / "training_configuration.json").write_text(json.dumps({
            "configuration": "S0", "evaluation_mode": "deterministic",
            "evaluation_budget_per_episode": 300, "population": 20,
        }))
        (root / "training_summary.json").write_text(json.dumps({
            "completed": True, "component": component,
            "total_actual_evaluations": 300,
            "episodes": [{
                "instance": "train-S0-001", "ppo_updates_total": 2,
            }],
        }))
        episode = root / "episodes" / "train-S0-001"
        episode.mkdir(parents=True)
        name = "crossover_outcomes.jsonl" if component == "C" else "mutation_outcomes.jsonl"
        rows = [
            {"evaluation": 50, "reward": 1.0, "reward_hv_delta": .01,
             "reward_archive_size": 2, "reward_recorded_for_policy": True},
            {"evaluation": 150, "reward": 0.0, "reward_hv_delta": 0.0,
             "reward_archive_size": 3, "reward_recorded_for_policy": True},
            {"evaluation": 250, "reward": 2.0, "reward_hv_delta": .02,
             "reward_archive_size": 4, "reward_recorded_for_policy": True},
        ]
        (episode / name).write_text("".join(json.dumps(row) + "\n" for row in rows))
        updates = [{
            "update": index, "transition_count": 10,
            "positive_reward_transition_count": index - 1,
            "reward_total": float(index), "entropy": .5, "kl": .01,
            "policy_loss": .1, "value_loss": .2, "update_seconds": 1.,
        } for index in (1, 2)]
        (episode / "ppo_updates.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in updates))

    def test_training_only_phase_aggregation(self):
        with tempfile.TemporaryDirectory() as temporary:
            temporary = Path(temporary)
            c_root, m_root = temporary / "c", temporary / "m"
            self._training(c_root, "C")
            self._training(m_root, "M")
            result = analyze(c_root, m_root, temporary / "analysis")
            self.assertTrue(result["completed"])
            self.assertFalse(result["test_instances_used"])
            for component in ("C", "M"):
                rows = [row for row in result["phase_summary"]
                        if row["component"] == component]
                self.assertEqual([row["reward_events"] for row in rows], [1, 1, 1])
                self.assertEqual(
                    [row["positive_reward_events"] for row in rows], [1, 0, 1])
            self.assertTrue((temporary / "analysis" / "analysis.md").is_file())
            self.assertTrue((temporary / "analysis" / "positive_reward_by_phase.svg").is_file())


if __name__ == "__main__":
    unittest.main()
