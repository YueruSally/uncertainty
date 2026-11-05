from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest

from analyze_m_operator_oracle_audit import summarize_state
from analyze_m_operator_repeat_check import analyze, heldout_repeats, main, metrics
from tests.test_m_operator_oracle_analysis import fixture, save_run


class OperatorRepeatCheckTests(unittest.TestCase):
    def test_lucky_winner_is_not_scored_on_its_selection_repeats(self):
        state, rows = fixture(legal=("add", "replace"))
        for row in rows:
            row["population_delta_hv"] = (1.0 if row["operator"] == "replace"
                                           else [4.0, 0.0, 0.0][row["replicate"]])
        sample = summarize_state(state, rows)
        self.assertGreater(sample["oracle_gain_over_balanced"], 0)
        folds = heldout_repeats(sample)
        self.assertEqual([r["selected_operator"] for r in folds], ["replace", "add", "add"])
        self.assertEqual([r["heldout_advantage"] for r in folds], [0, -1, -1])
        result = metrics([{"repeat_folds": folds}])
        self.assertAlmostEqual(result["mean_net_gain_all_states"], -2 / 3)
        self.assertAlmostEqual(result["negative_gain_total"], -2 / 3)

    def test_ties_prefer_balanced_and_never_select_illegal_operator(self):
        state, rows = fixture(legal=("mode", "replace"))
        folds = heldout_repeats(summarize_state(state, rows))
        self.assertTrue(all(row["selected_operator"] == "replace" for row in folds))
        result = metrics([{"repeat_folds": folds}])
        self.assertEqual(result["mean_net_gain_all_states"], 0)
        self.assertEqual(result["switch_fraction"], 0)
        self.assertIsNone(result["mean_gain_when_switched"])

    def test_all_states_include_zero_for_fallback(self):
        switched = {"repeat_folds": [{"selected_operator": "mode", "heldout_advantage": 2.0}] * 3}
        fallback = {"repeat_folds": [{"selected_operator": "replace", "heldout_advantage": 0.0}] * 3}
        result = metrics([switched, fallback])
        self.assertEqual(result["mean_net_gain_all_states"], 1)
        self.assertEqual(result["mean_gain_when_switched"], 2)
        self.assertEqual(result["positive_gain_total"], 2)
        self.assertEqual(result["state_count"], 2)
        self.assertEqual(result["rotation_decision_count"], 6)

    def test_grouping_evidence_and_cli_preserve_raw_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = []
            for i in range(2):
                state, rows = fixture(legal=("mode", "replace"))
                state["instance_id"] = f"instance_{i}"
                for row in rows:
                    row["fingerprint"] = f"{row['operator']}_{row['replicate']}"
                    row["population_delta_hv"] = float(i == 0 and row["operator"] == "mode")
                    row["repair"] = {"repair_action_count": i}
                run = root / f"run{i}"
                save_run(run, [state], rows)
                runs.append(run)
            original = {p: p.read_bytes() for r in runs for p in r.iterdir()}
            with redirect_stdout(StringIO()):
                main(["--runs", *map(str, runs), "--out", str(root / "analysis"), "--expected-state-count", "2"])
            report = json.loads((root / "analysis" / "REPEAT_CHECK.json").read_text())
            self.assertEqual(report["per_instance"]["instance_0"]["mean_net_gain_all_states"], 1)
            self.assertEqual(report["per_instance"]["instance_1"]["mean_net_gain_all_states"], 0)
            self.assertEqual(report["per_operator"]["del"]["comparable_legal_states"], 0)
            self.assertEqual(report["tie_diagnostics_nonexclusive"]["same_child_each_repeat"], 0)
            self.assertEqual(report["tie_diagnostics_nonexclusive"]["all_replicate_contributions_zero"], 1)
            self.assertFalse(report["training_data_approved"])
            self.assertIn("legacy_repair_modified_child_replay_required", report["data_reuse_status_counts"])
            self.assertEqual(original, {p: p.read_bytes() for p in original})

    def test_missing_feedback_is_not_silently_removed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "run"
            a, a_rows = fixture(1)
            b, b_rows = fixture(2)
            save_run(root, [a, b], a_rows + b_rows[:-1])
            with self.assertRaisesRegex(ValueError, "incomplete"):
                analyze([root], expected_state_count=2)


if __name__ == "__main__":
    unittest.main()
