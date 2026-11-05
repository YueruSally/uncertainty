"""Regression checks for masks, missing feedback and repeat-count accounting."""
from copy import deepcopy
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest

from analyze_m_operator_oracle_audit import (
    OPERATORS, analyze_runs, main, summarize_state,
)


def fixture(state_id=1, legal=("add", "mod", "replace")):
    candidates = [
        {"candidate_id": index, "candidate_operator": operator,
         "eligible": operator in legal}
        for index, operator in enumerate(OPERATORS)
    ]
    state = {
        "state_id": state_id, "instance_id": "toy", "generation": state_id,
        "evaluation": state_id * 6, "evaluation_fraction": state_id / 33.0,
        "population_hv": 0.5, "candidates": candidates,
        "audited_candidate_ids": [
            row["candidate_id"] for row in candidates if row["eligible"]],
        "observation": {"eligible": [row["eligible"] for row in candidates]},
    }
    outcomes = [
        {"state_id": state_id, "candidate_id": index, "operator": operator,
         "replicate": repeat, "population_delta_hv": 0.0,
         "mutation_success": True, "feasible": True}
        for repeat in range(3) for index, operator in enumerate(OPERATORS)
        if operator in legal
    ]
    return state, outcomes


def save_run(root, states, outcomes):
    root.mkdir(parents=True)
    for name, rows in (("states", states), ("outcomes", outcomes)):
        (root / f"mutation_candidate_audit_{name}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class OperatorOracleAnalysisTests(unittest.TestCase):
    def test_three_legal_operators_are_complete_and_illegal_values_are_null(self):
        state, outcomes = fixture()
        row = summarize_state(state, outcomes)
        self.assertEqual(row["eligible_operators"], ["add", "mod", "replace"])
        self.assertEqual(row["illegal_operators"], ["del", "mode"])
        self.assertEqual(row["mean_q_values"], [0.0, None, 0.0, None, 0.0])
        self.assertEqual(row["best_operator"], "replace")
        self.assertFalse(row["stable_nonbalanced_opportunity"])

    def test_masks_preserve_all_64_states_and_666_evaluations(self):
        with tempfile.TemporaryDirectory() as directory:
            runs = []
            for run_index, all_five_count in enumerate((7, 8)):
                states, outcomes = [], []
                for index in range(32):
                    legal = OPERATORS if index < all_five_count else ("add", "mod", "replace")
                    state, rows = fixture(index + 1, legal)
                    states.append(state)
                    outcomes.extend(rows)
                run = Path(directory) / str(run_index)
                save_run(run, states, outcomes)
                runs.append(run)
            summary, dataset = analyze_runs(runs, expected_state_count=64)
            self.assertEqual(summary["state_count"], 64)
            self.assertEqual(len(dataset), 64)
            self.assertEqual(summary["all_five_legal_state_count"], 15)
            self.assertEqual(summary["counterfactual_evaluation_count"], 666)
            self.assertEqual(summary["expected_counterfactual_evaluation_count"], 666)
            self.assertEqual(summary["analyzed_counterfactual_evaluation_count"], 666)
            self.assertEqual(summary["excluded_state_count"], 0)
            self.assertTrue(summary["completeness_pass"])

    def test_missing_repeat_is_reported_and_completeness_fails(self):
        first, first_outcomes = fixture(1)
        second, second_outcomes = fixture(2)
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "run"
            save_run(run, [first, second], first_outcomes + second_outcomes[:-1])
            summary, _ = analyze_runs([run], expected_state_count=2)
            self.assertEqual(summary["raw_state_count"], 2)
            self.assertEqual(summary["state_count"], 1)
            self.assertEqual(summary["excluded_state_count"], 1)
            self.assertIn("missing", summary["excluded_states"][0]["reason"])
            self.assertFalse(summary["completeness_pass"])

    def test_duplicate_repeat_does_not_count_as_complete_feedback(self):
        state, outcomes = fixture()
        outcomes[-1] = deepcopy(outcomes[-2])
        with self.assertRaisesRegex(ValueError, "duplicate operator replicate"):
            summarize_state(state, outcomes)

    def test_mask_mismatch_is_detected(self):
        state, outcomes = fixture()
        state["observation"]["eligible"][0] = False
        with self.assertRaisesRegex(ValueError, "eligibility disagrees"):
            summarize_state(state, outcomes)

    def test_absent_baseline_is_not_imputed(self):
        state, outcomes = fixture(legal=("add", "mod"))
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "run"
            save_run(run, [state], outcomes)
            summary, rows = analyze_runs([run])
            self.assertTrue(summary["completeness_pass"])
            self.assertEqual(summary["balanced_comparable_state_count"], 0)
            self.assertEqual(summary["balanced_unavailable_state_count"], 1)
            self.assertIsNone(summary["mean_one_step_oracle_gain_over_balanced"])
            self.assertTrue(all(value is None for value in rows[0]["mean_relative_q_values"]))

    def test_one_lucky_repeat_is_not_reported_as_repeat_consistent(self):
        state, outcomes = fixture()
        for row in outcomes:
            if row["operator"] == "add" and row["replicate"] == 0:
                row["population_delta_hv"] = 0.03
        result = summarize_state(state, outcomes)
        self.assertEqual(result["best_operator"], "add")
        self.assertAlmostEqual(result["oracle_gain_over_balanced"], 0.01)
        self.assertFalse(result["stable_nonbalanced_opportunity"])

    def test_cli_exports_json_and_does_not_change_raw_logs(self):
        state, outcomes = fixture()
        with tempfile.TemporaryDirectory() as directory:
            run, out = Path(directory) / "run", Path(directory) / "analysis"
            save_run(run, [state], outcomes)
            raw_before = {path.name: path.read_bytes() for path in run.iterdir()}
            with redirect_stdout(StringIO()):
                main(["--runs", str(run), "--out", str(out), "--expected-state-count", "1"])
            result = json.loads((out / "OPERATOR_ORACLE_AUDIT.json").read_text())
            dataset = json.loads((out / "OPERATOR_COUNTERFACTUAL_DATASET.json").read_text())
            self.assertTrue(result["completeness_pass"])
            self.assertEqual(dataset["states"][0]["mean_q_values"][1], None)
            self.assertEqual(raw_before, {path.name: path.read_bytes() for path in run.iterdir()})


if __name__ == "__main__":
    unittest.main()
