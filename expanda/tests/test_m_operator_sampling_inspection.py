"""Separate sampling shortages from corrupt feedback without new evaluations."""
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import json
import tempfile
import unittest

import collect_m_operator_feedback_v1 as collector
from inspect_m_operator_sampling_v1 import inspect_collection, inspect_run, phase_allocation
from analyze_m_operator_oracle_audit import phase
from transformer_ppo.controller import PPOController
from tests.test_m_operator_feedback_collection import completed_run, plan_fixture


def write_policy_opportunities(run, counts):
    config = collector.read_json(run / "configuration.json")
    rows = []
    for part, count in enumerate(counts):
        for index in range(count):
            rows.append({
                "policy_state_id": len(rows) + 1, "instance_id": config["instance_id"],
                "evaluation": int((part / 3 + (index + 1) / (count + 1) / 3) * 400),
                "candidates": [{"eligible": True}, {"eligible": True}],
                "observation": {"eligible": [True, True]},
            })
    (run / "mutation_policy_states.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))


class OperatorSamplingInspectionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.plan = plan_fixture()
        self.plan_hash = collector.json_hash(self.plan)
        self.instance, self.seed = collector.INSTANCE_IDS[0], collector.ALGORITHM_SEEDS[0]
        self.run = completed_run(self.root, self.instance, self.seed, self.plan, self.plan_hash)
        write_policy_opportunities(self.run, (20, 20, 20))

    def test_current_threshold_sampler_can_miss_early_states_despite_capacity(self):
        controller = SimpleNamespace(candidate_audit_states=32, candidate_audit_state_count=0,
                                     _candidate_audit_pending=None, budget=400)
        evaluations = [20, 22, 24, 26, 28, 30, 100, 112, 124, 130] + list(range(134, 265, 6)) + list(range(270, 381, 5))
        sampled = []
        for evaluation in evaluations:
            controller.evaluations = evaluation
            if PPOController._should_capture_candidate_audit(controller, [{"eligible": True}] * 3):
                controller.candidate_audit_state_count += 1
                sampled.append(evaluation)
        available = Counter(phase(value / 400) for value in evaluations)
        captured = Counter(phase(value / 400) for value in sampled)
        self.assertEqual(available["early"], 10)
        self.assertEqual(captured["early"], 6)
        self.assertLess(len(sampled), 32)
        self.assertIsNotNone(phase_allocation(available))

    def test_valid_31_state_feedback_is_not_reported_as_missing_operator_labels(self):
        for name in ("states", "outcomes"):
            path = self.run / f"mutation_candidate_audit_{name}.jsonl"
            rows = [row for row in collector.read_jsonl(path) if row["state_id"] != 32]
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        result = collector.read_json(self.run / "COMPLETE.json")
        result.update(candidate_audit_state_count=31, candidate_audit_evaluation_count=31 * 9)
        collector.write_json(self.run / "COMPLETE.json", result)
        report = inspect_run(self.run)
        self.assertEqual(report["sampled_states"], 31)
        self.assertTrue(report["recorded_state_feedback_complete"])
        self.assertFalse(report["sample_target_met"])
        self.assertEqual(report["diagnosis"], "ENOUGH_LOGGED_OPPORTUNITIES_BUT_SAMPLER_MISSED_QUOTA")
        with self.assertRaisesRegex(ValueError, "sample count shortfall.*actual=31"):
            collector.validate_run(self.run, self.instance, self.seed, self.plan, self.plan_hash)

    def test_missing_one_operator_repeat_remains_a_real_feedback_error(self):
        path = self.run / "mutation_candidate_audit_outcomes.jsonl"
        path.write_text("\n".join(path.read_text().splitlines()[:-1]) + "\n")
        report = inspect_run(self.run)
        self.assertFalse(report["recorded_state_feedback_complete"])
        self.assertFalse(report["completion_counters_match_logs"])
        self.assertEqual(report["diagnosis"], "COMPLETION_COUNTERS_DISAGREE_WITH_SAVED_LOGS")

    def test_insufficient_early_opportunities_cannot_be_fixed_by_balancing_selection(self):
        self.assertIsNone(phase_allocation({"early": 6, "middle": 40, "late": 40}))
        self.assertIsNone(phase_allocation({"early": 8, "middle": 8, "late": 8}))
        self.assertEqual(sum(phase_allocation({"early": 8, "middle": 40, "late": 40}).values()), 32)

    def test_inspection_preserves_all_source_files_and_does_not_approve_training(self):
        original = {p.name: p.read_bytes() for p in self.run.iterdir()}
        report = inspect_run(self.run)
        self.assertEqual(report["diagnosis"], "ORIGINAL_SAMPLE_CHECKS_MET")
        self.assertEqual(report["eligible_opportunities_by_phase"], {"early": 20, "middle": 20, "late": 20})
        self.assertEqual(original, {p.name: p.read_bytes() for p in self.run.iterdir()})

    def test_policy_mask_corruption_cannot_be_called_available_capacity(self):
        path = self.run / "mutation_policy_states.jsonl"
        rows = collector.read_jsonl(path)
        rows[0]["observation"]["eligible"][0] = False
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        with self.assertRaisesRegex(ValueError, "candidate mask mismatch"):
            inspect_run(self.run)

    def test_collection_inspection_uses_saved_plan_without_certifying_data(self):
        self.plan.update(instance_ids=[self.instance], states_per_run=32,
                         batches={"1": {"seed": self.seed}})
        plan_hash = collector.freeze_plan(self.root, self.plan)
        collector.write_json(self.root / "requests" / (self.run.name + ".json"),
                             {"plan_sha256": plan_hash, "instance": self.instance, "seed": self.seed})
        original_plan = (self.root / "PLAN.json").read_bytes()
        report = inspect_collection(self.root)
        self.assertTrue(report["diagnostic_completed"])
        self.assertEqual(report["new_objective_evaluations"], 0)
        self.assertEqual(report["new_training_steps"], 0)
        self.assertEqual(report["recorded_states_in_checked_runs"], 32)
        self.assertFalse(report["training_data_approved"])
        self.assertFalse((self.root / "BATCH_1_COMPLETE").exists())
        self.assertEqual((self.root / "PLAN.json").read_bytes(), original_plan)


if __name__ == "__main__":
    unittest.main()
