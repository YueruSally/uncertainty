from collections import Counter
from contextlib import redirect_stdout
from copy import deepcopy
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
import json
import shutil
import tempfile
import unittest
from unittest.mock import patch

import collect_m_operator_feedback_v1 as common
import collect_m_operator_feedback_v2 as collect
from analyze_m_operator_oracle_audit import phase
from transformer_ppo.audit_sampling import PHASES, select_phase_states, validate_policy_state_ids
from transformer_ppo.controller import PPOController
from tests.test_m_operator_feedback_collection import plan_fixture
from tests.test_m_operator_oracle_analysis import fixture


COUNTS = ((20, 21, 20), (6, 23, 16), (19, 22, 25), (17, 25, 18), (16, 24, 21), (16, 19, 17))


def create_source(run, instance, seed, plan, counts=(20, 21, 20)):
    run.mkdir(parents=True)
    policy = []
    for part, count in enumerate(counts):
        for index in range(count):
            state_id = len(policy) + 1
            state, _ = fixture(state_id, legal=("add", "mode", "replace"))
            evaluation = int((part / 3 + (index + 1) / (count + 1) / 3) * 400)
            policy.append({"policy_state_id": state_id, "instance_id": instance,
                           "generation": evaluation // 20, "evaluation": evaluation,
                           "candidates": state["candidates"], "observation": state["observation"]})
    (run / "mutation_policy_states.jsonl").write_text("".join(json.dumps(row) + "\n" for row in policy))
    common.write_json(run / "configuration.json", {
        "component": "M", "evaluation_mode": "ccp", "instance_id": instance,
        "population": 20, "generations": 1000, "evaluation_budget": 400,
        "training_size": 100, "alpha": .9, "policy_training": False,
        "seeds": {"algorithm": seed}, "uncertainty": {"name": "transport_border_saa_v1", "scenario_count": 100},
        "policy_checkpoint": {"checkpoint_sha256": plan["inputs"]["checkpoint"]["sha256"]},
        "policy_inference": {"action_mode": "argmax", "policy_seed": 68101},
        "scenario_digest": "scenario-" + instance, "candidate_audit": None})
    common.write_json(run / "COMPLETE.json", {"completed": True, "ppo_updates": 0,
                     "mutation_control": "operator", "stop_reason": "evaluation_budget",
                     "real_objective_evaluation_count": 400, "candidate_audit_state_count": 0,
                     "candidate_audit_evaluation_count": 0, "final_hypervolume": .5})
    common.write_json(run / "final_archive_lineage.json", [{"decision_fingerprint": "final-solution"}])


def create_replay(run, selection):
    source = Path(selection["source_run"])
    run.mkdir(parents=True)
    for name in ("configuration.json", "mutation_policy_states.jsonl", "final_archive_lineage.json", "COMPLETE.json"):
        shutil.copyfile(source / name, run / name)
    config = common.read_json(run / "configuration.json")
    config["candidate_audit"] = {"states": 32, "candidates": 5, "replicates": 3,
                                  "policy_state_ids": selection["policy_state_ids"]}
    common.write_json(run / "configuration.json", config)
    states, outcomes = [], []
    for row in collect.iter_jsonl(run / "mutation_policy_states.jsonl"):
        if row["policy_state_id"] not in selection["policy_state_ids"]:
            continue
        state, feedback = fixture(len(states) + 1, legal=("add", "mode", "replace"))
        state.update({key: row[key] for key in ("instance_id", "generation", "evaluation", "observation", "policy_state_id")})
        state["evaluation_fraction"] = row["evaluation"] / 400
        states.append(state)
        for item in feedback:
            item["execution_contract"] = common.CONTRACT
        outcomes.extend(feedback)
    for name, rows in (("states", states), ("outcomes", outcomes)):
        (run / f"mutation_candidate_audit_{name}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    result = common.read_json(run / "COMPLETE.json")
    result.update(candidate_audit_state_count=32, candidate_audit_evaluation_count=len(outcomes))
    common.write_json(run / "COMPLETE.json", result)


class CapacitySamplingTests(unittest.TestCase):
    def opportunities(self, counts):
        result, start = {}, 1
        for name, count in zip(PHASES, counts):
            result[name] = list(range(start, start + count))
            start += count
        return result

    def test_observed_scarce_early_phase_uses_six_real_states_and_redistributes(self):
        result = select_phase_states(self.opportunities((6, 23, 16)))
        self.assertEqual(result["quotas"], {"early": 6, "middle": 13, "late": 13})
        self.assertEqual(result["policy_state_ids"][:6], list(range(1, 7)))
        self.assertEqual(len(set(result["policy_state_ids"])), 32)
        self.assertEqual(result["capacity_limited_phases"], ["early"])

    def test_every_observed_trajectory_has_32_distinct_outcome_blind_selections(self):
        for counts in COUNTS:
            result = select_phase_states(self.opportunities(counts))
            self.assertEqual(len(result["policy_state_ids"]), 32)
            if counts[0] != 6:
                self.assertEqual(result["quotas"], {"early": 10, "middle": 11, "late": 11})

    def test_insufficient_opportunities_never_pad_or_duplicate_states(self):
        for counts in ((10, 10, 10), (0, 30, 30)):
            with self.assertRaises(ValueError):
                select_phase_states(self.opportunities(counts))
        for ids in ([1, 1], [2, 1], [0, 1], [True, 2]):
            with self.assertRaises(ValueError):
                validate_policy_state_ids(ids, 2)

    def test_explicit_sampler_hits_requested_states_even_before_old_thresholds(self):
        controller = SimpleNamespace(candidate_audit_states=32, candidate_audit_state_count=0,
                                     _candidate_audit_pending=None, budget=400)
        result = select_phase_states(self.opportunities((6, 23, 16)))
        controller.candidate_audit_policy_state_ids = result["policy_state_ids"]
        captured = []
        for state_id in range(1, 46):
            controller.policy_state_count, controller.evaluations = state_id, state_id * 7
            if PPOController._should_capture_candidate_audit(controller, [{"eligible": True}] * 5):
                captured.append(state_id)
                controller.candidate_audit_state_count += 1
        self.assertEqual(captured, result["policy_state_ids"])


class FeedbackReplayTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.plan = plan_fixture()
        self.instance, self.seed = common.INSTANCE_IDS[1], common.ALGORITHM_SEEDS[0]
        self.source = self.root / "source"
        create_source(self.source, self.instance, self.seed, self.plan, (6, 23, 16))
        self.selection = collect.inventory(self.source, self.instance, self.seed, self.plan)

    def test_inventory_does_not_need_or_read_any_counterfactual_rewards(self):
        (self.source / "mutation_candidate_audit_outcomes.jsonl").write_text("corrupt old rewards, deliberately unread")
        self.assertEqual(collect.inventory(self.source, self.instance, self.seed, self.plan), self.selection)

    def test_exact_replay_validates_all_32_legal_feedback_states_and_capacity(self):
        run = self.root / "replay"
        create_replay(run, self.selection)
        result = collect.validate_replay(run, self.selection, self.plan)
        self.assertTrue(result["replay_exact"])
        self.assertEqual(result["phase_counts"], {"early": 6, "middle": 13, "late": 13})
        self.assertEqual(result["counterfactual_evaluations"], 288)

    def test_changed_trace_final_hv_or_selected_snapshot_rejects_replay(self):
        for mode in ("trace", "hv", "snapshot", "missing_feedback"):
            run = self.root / mode
            create_replay(run, self.selection)
            if mode == "hv":
                value = common.read_json(run / "COMPLETE.json")
                value["final_hypervolume"] += .001
                common.write_json(run / "COMPLETE.json", value)
            elif mode in ("trace", "snapshot"):
                path = run / ("mutation_policy_states.jsonl" if mode == "trace" else "mutation_candidate_audit_states.jsonl")
                rows = collect.read_jsonl(path)
                rows[0]["observation"]["new_feature"] = 1
                path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            else:
                path = run / "mutation_candidate_audit_outcomes.jsonl"
                path.write_text("\n".join(path.read_text().splitlines()[:-1]) + "\n")
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                collect.validate_replay(run, self.selection, self.plan)

    def test_completed_invalid_replay_is_not_resampled(self):
        root = self.root / "new"
        run = root / "runs" / f"{self.instance}_seed_{self.seed}"
        create_replay(run, self.selection)
        config = common.read_json(run / "configuration.json")
        config["training_size"] = 200
        common.write_json(run / "configuration.json", config)
        with patch.object(collect.subprocess, "run") as execute, self.assertRaises(ValueError):
            collect.replay_one(root, self.selection, self.plan, 2)
        execute.assert_not_called()

    def test_synthetic_full_batch_reuses_sources_then_completes_and_resumes(self):
        source_root, root = self.root / "old", self.root / "new"
        old_plan = deepcopy(self.plan)
        common.freeze_plan(source_root, old_plan)
        for instance, counts in zip(common.INSTANCE_IDS, COUNTS):
            run = source_root / "runs" / f"{instance}_seed_{self.seed}"
            create_source(run, instance, self.seed, old_plan, counts)
            common.write_json(source_root / "requests" / (run.name + ".json"),
                              {"plan_sha256": common.json_hash(old_plan), "instance": instance, "seed": self.seed})
        before = {str(path): path.read_bytes() for path in source_root.rglob("*") if path.is_file()}

        def fake_rollout(command, **kwargs):
            run = Path(command[command.index("--out") + 1])
            self.assertFalse(run.exists(), "do not prefill the episode output directory")
            self.assertEqual(command[command.index("--scenario-count") + 1], "100")
            self.assertEqual(command[command.index("--evaluation-budget") + 1], "400")
            instance = Path(command[command.index("--instance") + 1]).name
            seed = int(command[command.index("--algorithm-seed") + 1])
            if "--candidate-audit-policy-state-ids" not in command:
                self.assertEqual(command[command.index("--candidate-audit-states") + 1], "0")
                create_source(run, instance, seed, self.plan, COUNTS[common.INSTANCE_IDS.index(instance)])
                return SimpleNamespace(returncode=0)
            source = ((source_root / "runs") if seed == self.seed else (root / "discovery")) / f"{instance}_seed_{seed}"
            selection = collect.inventory(source, instance, seed, self.plan)
            actual_ids = list(map(int, command[command.index("--candidate-audit-policy-state-ids") + 1:]))
            self.assertEqual(actual_ids, selection["policy_state_ids"])
            batch = 1 if seed == self.seed else 2
            self.assertTrue((root / f"batch_{batch}/SAMPLING_PLAN.json").exists())
            create_replay(run, selection)
            return SimpleNamespace(returncode=0)

        arguments = ["--instances", str(self.root), "--normalization", str(self.root / "norm"),
                     "--checkpoint", str(self.root / "policy"), "--source-root", str(source_root), "--out", str(root)]
        with patch.object(common, "make_plan", side_effect=lambda *args: deepcopy(self.plan)), \
                patch.object(collect.subprocess, "run", side_effect=fake_rollout) as execute, redirect_stdout(StringIO()):
            collect.main(arguments)
            collect.main(arguments)
            collect.main(arguments + ["--batch", "2"])
            collect.main(arguments + ["--batch", "2"])
        self.assertEqual(execute.call_count, 18)
        result = common.read_json(root / "batch_1/QUALITY.json")
        self.assertEqual(result["state_count"], 192)
        self.assertEqual(result["phase_counts"], {"early": 56, "middle": 68, "late": 68})
        self.assertEqual(result["new_discovery_search_evaluations"], 0)
        self.assertEqual(result["replay_search_evaluations"], 2400)
        self.assertEqual(result["new_training_steps"], 0)
        cumulative = common.read_json(root / "batch_2/QUALITY.json")
        self.assertEqual(cumulative["state_count"], 384)
        self.assertEqual(cumulative["new_discovery_search_evaluations"], 2400)
        self.assertEqual(cumulative["discovery_search_evaluations_including_reused"], 4800)
        self.assertEqual(cumulative["replay_search_evaluations"], 4800)
        self.assertEqual(before, {str(path): path.read_bytes() for path in source_root.rglob("*") if path.is_file()})
