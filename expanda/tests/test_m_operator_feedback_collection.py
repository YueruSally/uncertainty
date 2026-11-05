"""Collection protocol checks; synthetic feedback, no model or server required."""
from contextlib import redirect_stdout, redirect_stderr
from copy import deepcopy
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import collect_m_operator_feedback_v1 as collector
from tests.test_m_operator_oracle_analysis import fixture, save_run


def plan_fixture():
    inputs = {name: {"path": "/fixture/" + name, "sha256": name + "-hash"}
              for name in (*collector.INSTANCE_IDS, "workbook", "normalization", "checkpoint")}
    return {"inputs": inputs, "sampling_policy": "frozen V6 operator; no training",
            "split_unit": "instance then trajectory", "source_hashes": {"source.py": "v1"}}


def completed_run(root, instance, seed, plan, plan_hash, *, tied=False):
    run = root / "runs" / f"{instance}_seed_{seed}"
    states, outcomes = [], []
    for index in range(1, 33):
        state, rows = fixture(index, legal=("add", "mode", "replace"))
        state["instance_id"] = instance
        for row in rows:
            row["execution_contract"] = collector.CONTRACT
            # A dataset in which no alternative improves over Balanced must
            # still pass data quality. Quality cannot select positive results.
            row["population_delta_hv"] = 0.0 if tied else float(row["operator"] == "replace") * .01
        states.append(state)
        outcomes.extend(rows)
    save_run(run, states, outcomes)
    collector.write_json(root / "requests" / (run.name + ".json"),
                         {"plan_sha256": plan_hash, "instance": instance, "seed": seed})
    collector.write_json(run / "configuration.json", {
        "component": "M", "evaluation_mode": "ccp", "instance_id": instance,
        "population": 20, "evaluation_budget": 400, "training_size": 100,
        "generations": 1000, "alpha": .9, "policy_training": False,
        "candidate_audit": {"states": 32, "candidates": 5, "replicates": 3},
        "seeds": {"algorithm": seed},
        "uncertainty": {"name": "transport_border_saa_v1", "scenario_count": 100},
        "scenario_digest": "scenario-" + instance,
        "policy_checkpoint": {"checkpoint_sha256": plan["inputs"]["checkpoint"]["sha256"]},
        "policy_inference": {"action_mode": "argmax", "policy_seed": 68101},
    })
    collector.write_json(run / "COMPLETE.json", {
        "completed": True, "real_objective_evaluation_count": 400,
        "stop_reason": "evaluation_budget", "ppo_updates": 0, "mutation_control": "operator",
        "candidate_audit_evaluation_count": len(outcomes), "optimisation_seconds": .1,
    })
    return run


class OperatorFeedbackCollectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.plan = plan_fixture()
        self.plan_hash = collector.freeze_plan(self.root, self.plan)
        self.instance, self.seed = collector.INSTANCE_IDS[0], collector.ALGORITHM_SEEDS[0]

    def test_frozen_plan_rejects_code_or_input_changes_without_overwriting(self):
        original = (self.root / "PLAN.json").read_bytes()
        changed = deepcopy(self.plan)
        changed["source_hashes"]["source.py"] = "v2"
        with self.assertRaisesRegex(ValueError, "PLAN/input/code mismatch"):
            collector.freeze_plan(self.root, changed)
        self.assertEqual((self.root / "PLAN.json").read_bytes(), original)
        self.assertEqual(collector.freeze_plan(self.root, self.plan), self.plan_hash)

    def test_child_launch_keeps_output_empty_and_preserves_completed_resume(self):
        run = self.root / "runs" / f"{self.instance}_seed_{self.seed}"

        def fake_rollout(command, **kwargs):
            self.assertFalse(run.exists(), "episode runner rejects a nonempty output directory")
            options = dict(zip(command[3::2], command[4::2]))
            self.assertEqual(options["--scenario-count"], "100")
            self.assertEqual(options["--pop"], "20")
            self.assertEqual(options["--evaluation-budget"], "400")
            self.assertEqual(options["--algorithm-seed"], str(self.seed))
            self.assertEqual(options["--inference-action-mode"], "argmax")
            self.assertEqual(options["--candidate-audit-replicates"], "3")
            self.assertNotIn("--resume", options)
            self.assertEqual(kwargs["env"]["OMP_NUM_THREADS"], "1")
            completed_run(self.root, self.instance, self.seed, self.plan, self.plan_hash)
            return SimpleNamespace(returncode=0)

        with patch.object(collector.subprocess, "run", side_effect=fake_rollout) as execute, redirect_stdout(StringIO()):
            first = collector.collect_one(self.root, self.instance, self.seed, self.plan, self.plan_hash, 2)
            second = collector.collect_one(self.root, self.instance, self.seed, self.plan, self.plan_hash, 2)
        self.assertEqual(first["status"], "complete")
        self.assertEqual(second["status"], "existing")
        self.assertEqual(execute.call_count, 1)

    def test_retry_quarantines_failed_output_and_keeps_the_same_seed(self):
        run = self.root / "runs" / f"{self.instance}_seed_{self.seed}"
        commands = []

        def interrupted_then_complete(command, **kwargs):
            commands.append(command)
            self.assertFalse(run.exists())
            if len(commands) == 1:
                run.mkdir()
                (run / "partial.txt").write_text("interrupted")
                return SimpleNamespace(returncode=137)
            completed_run(self.root, self.instance, self.seed, self.plan, self.plan_hash)
            return SimpleNamespace(returncode=0)

        with patch.object(collector.subprocess, "run", side_effect=interrupted_then_complete), redirect_stdout(StringIO()):
            result = collector.collect_one(self.root, self.instance, self.seed, self.plan, self.plan_hash, 2)
        self.assertEqual(commands[0], commands[1])
        self.assertEqual(len(list((self.root / "interrupted_runs").glob("*/partial.txt"))), 1)
        self.assertEqual(len(list((self.root / "logs").glob("*.log"))), 2)
        self.assertEqual(result["states"], 32)

    def test_completed_bad_configuration_is_rejected_and_never_resampled(self):
        run = completed_run(self.root, self.instance, self.seed, self.plan, self.plan_hash)
        config = collector.read_json(run / "configuration.json")
        config["training_size"] = 200
        collector.write_json(run / "configuration.json", config)
        with patch.object(collector.subprocess, "run") as execute:
            with self.assertRaisesRegex(ValueError, "wrong training_size"):
                collector.collect_one(self.root, self.instance, self.seed, self.plan, self.plan_hash, 3)
        execute.assert_not_called()
        self.assertTrue((run / "COMPLETE.json").exists())

    def test_missing_feedback_or_legacy_labels_cannot_pass_quality(self):
        run = completed_run(self.root, self.instance, self.seed, self.plan, self.plan_hash)
        path = run / "mutation_candidate_audit_outcomes.jsonl"
        original = path.read_text()
        path.write_text("\n".join(original.splitlines()[:-1]) + "\n")
        with self.assertRaisesRegex(ValueError, "incomplete legal feedback"):
            collector.validate_run(run, self.instance, self.seed, self.plan, self.plan_hash)
        path.write_text(original.replace(collector.CONTRACT, "legacy_unverified"))
        with self.assertRaisesRegex(ValueError, "unverified legacy"):
            collector.validate_run(run, self.instance, self.seed, self.plan, self.plan_hash)

    def test_imbalanced_phase_coverage_is_reported_without_success_marker(self):
        run = completed_run(self.root, self.instance, self.seed, self.plan, self.plan_hash)
        path = run / "mutation_candidate_audit_states.jsonl"
        states = collector.read_jsonl(path)
        for state in states:
            state["evaluation_fraction"] = .1
        path.write_text("".join(collector.json.dumps(state) + "\n" for state in states))
        with self.assertRaisesRegex(ValueError, "phase coverage"):
            collector.validate_run(run, self.instance, self.seed, self.plan, self.plan_hash)
        self.assertFalse((self.root / "BATCH_1_COMPLETE").exists())

    def test_192_then_384_panel_passes_quality_even_without_positive_gain(self):
        for batch, seed in enumerate(collector.ALGORITHM_SEEDS, 1):
            for instance in collector.INSTANCE_IDS:
                completed_run(self.root, instance, seed, self.plan, self.plan_hash)
            quality = collector.finish_batch(self.root, batch, self.plan, self.plan_hash)
            self.assertTrue(quality["data_quality_pass"])
            self.assertEqual(quality["state_count"], batch * 192)
            self.assertEqual(quality["search_objective_evaluations"], batch * 2400)
            self.assertEqual(quality["counterfactual_evaluations"], batch * 1728)
            self.assertEqual(quality["new_training_steps"], 0)
            data = collector.read_json(self.root / f"batch_{batch}/DATASET.json")
            self.assertEqual(len(data["states"]), 192 * batch)
            self.assertTrue(all(state["oracle_gain_over_balanced"] == 0 for state in data["states"]))
            self.assertTrue(all(state["mean_relative_q_values"][1] is None for state in data["states"]))

    def test_second_batch_cannot_silently_change_scenario_set(self):
        for seed in collector.ALGORITHM_SEEDS:
            for instance in collector.INSTANCE_IDS:
                run = completed_run(self.root, instance, seed, self.plan, self.plan_hash)
        config = collector.read_json(run / "configuration.json")
        config["scenario_digest"] = "different"
        collector.write_json(run / "configuration.json", config)
        with self.assertRaisesRegex(ValueError, "CCP scenarios changed"):
            collector.finish_batch(self.root, 2, self.plan, self.plan_hash)
        self.assertFalse((self.root / "BATCH_2_COMPLETE").exists())

    def test_collection_entrypoint_resumes_existing_batch_without_model_processes(self):
        for instance in collector.INSTANCE_IDS:
            completed_run(self.root, instance, self.seed, self.plan, self.plan_hash)
        with patch.object(collector, "make_plan", return_value=self.plan), \
                patch.object(collector.subprocess, "run") as execute, redirect_stdout(StringIO()):
            collector.main([
                "--instances", str(self.root), "--normalization", str(self.root / "normalization.json"),
                "--checkpoint", str(self.root / "policy.pt"), "--out", str(self.root),
                "--batch", "1", "--workers", "2"])
        execute.assert_not_called()
        status = collector.read_json(self.root / "STATUS.json")
        self.assertEqual(status["state"], "complete")
        self.assertEqual(status["completed_tasks"], 6)
        self.assertEqual(status["state_count"], 192)

    def test_entrypoint_rejects_batch_two_before_batch_one_quality(self):
        with patch.object(collector, "make_plan", return_value=self.plan), \
                patch.object(collector.subprocess, "run") as execute, \
                redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            with self.assertRaises(SystemExit) as caught:
                collector.main([
                    "--instances", str(self.root), "--normalization", str(self.root / "normalization.json"),
                    "--checkpoint", str(self.root / "policy.pt"), "--out", str(self.root), "--batch", "2"])
        self.assertEqual(caught.exception.code, 1)
        execute.assert_not_called()
        self.assertEqual(collector.read_json(self.root / "STATUS.json")["state"], "failed")


if __name__ == "__main__":
    unittest.main()
