from copy import deepcopy
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import baseline_uncertainty as base
import tests.test_m_operator_execution_consistency as fixtures
from m_candidate_information import (CandidateController, CandidatePolicy, build_proposals,
    proposal_features, METHODS, SPECIFIC_NAMES, THRESHOLD_SCALED, TEST_INSTANCES, TEST_SEEDS)
from m_operator_offline import TinyOperatorTransformer, digest, write_json
from multi_instance_logging import fingerprint
from run_m_candidate_information_v1 import summarize
from transformer_ppo.archive import RewardArchive, RewardConfig


class CandidateInformationTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.OperatorExecutionConsistencyTests()
        self.fixture.setUp()
        f = self.fixture
        self.policy = CandidatePolicy("fixture", 17, METHODS[0])
        (Path(f.tmp.name)/"concrete").mkdir()
        self.controller = CandidateController(Path(f.tmp.name)/"concrete", "fixture", "frozen100", 400,
            trainer=self.policy, archive=RewardArchive(f.normalizer), reward_config=RewardConfig(),
            batches=f.batches, generations=1000, pop_size=20, method=self.policy.name,
            instance={"instance_id": "fixture"}, seeds={"algorithm": 17})
        self.controller.generation, self.controller._active_mixture_event_index = 10, 1

    def tearDown(self):
        self.controller.close()
        self.fixture.tearDown()

    def proposals(self):
        f, c = self.fixture, self.controller
        candidates = c._operator_candidates(f.parent, f.batches, f.paths, f.tt, f.lookup, f.options)
        return build_proposals(c, f.parent, f.batches, f.paths, f.tt, f.lookup, f.options, candidates)

    def test_candidates_need_no_objective_evaluation_and_preserve_rng_parent(self):
        py, np_state, parent = random.getstate(), np.random.get_state(), fingerprint(self.fixture.parent)
        with patch.object(base, "evaluate_individual", side_effect=AssertionError("leaked evaluation")), \
             patch.object(base, "_evaluate_individual_impl", side_effect=AssertionError("leaked evaluation")):
            first, second = self.proposals(), self.proposals()
        self.assertEqual([p["fingerprint"] for p in first], [p["fingerprint"] for p in second])
        self.assertEqual(random.getstate(), py)
        np.testing.assert_equal(np.random.get_state(), np_state)
        self.assertEqual(fingerprint(self.fixture.parent), parent)
        self.assertTrue(all(np.isfinite(p["features"]).all() for p in first))
        self.assertTrue(any(p["fingerprint"] != parent for p in first))

    def test_executor_uses_frozen_child_without_resampling(self):
        c, f = self.controller, self.fixture
        c.proposals, c.proposal_parent = self.proposals(), fingerprint(f.parent)
        for p in c.proposals:
            child = deepcopy(f.parent)
            with patch.object(base, "apply_mutation_op", side_effect=AssertionError("redrawn modification")):
                result = c._execute_operator_action(child, p["operator"], f.batches, f.paths, f.tt, f.lookup, f.options)
            self.assertEqual(fingerprint(child), p["fingerprint"])
            self.assertEqual(result[0], p["ok"])
        self.assertEqual(c.execution_checks, 5)

    def test_candidate_inputs_ignore_objectives_feasibility_and_cached_scenarios(self):
        f = self.fixture
        p = self.proposals()[3]
        child, parent = deepcopy(p["child"]), deepcopy(f.parent)
        child.objectives, parent.objectives = (999., 888., 777.), (-9., -8., -7.)
        child.feasible, parent.feasible = False, True
        child.cost_s = np.full(100, 1e20)
        parent.rank, parent.crowding_distance = 888, 999.
        features = proposal_features(parent, child, p["batch"], p["target"], f.batches, p["ok"])
        np.testing.assert_array_equal(features, p["features"])

    def test_state_only_arm_cannot_read_candidate_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dim, summary_dim = 8+len(SPECIFIC_NAMES), 8
            torch.manual_seed(123)
            model = TinyOperatorTransformer(dim)
            checkpoints = {}
            for method in METHODS[1:]:
                path = root / (method+".pt")
                torch.save({"method": method, "model": model.state_dict()}, path)
                checkpoints[method] = [str(path)]
            bundle = root / "bundle.json"
            write_json(bundle, {"protocol": "candidate-information-v1", "threshold_scaled": THRESHOLD_SCALED,
                "summary_dim": summary_dim, "scaler": {"mean": [0.]*dim, "std": [1.]*dim,
                    "feature_clip": 10., "target_scale": 1.}, "checkpoints": checkpoints,
                "checkpoint_sha256": {paths[0]: digest(paths[0]) for paths in checkpoints.values()}})
            summary, specific, mask = np.zeros((5,8)), np.zeros((5,len(SPECIFIC_NAMES))), np.ones(5, bool)
            state = CandidatePolicy("fixture", 17, METHODS[1], bundle)
            candidate = CandidatePolicy("fixture", 17, METHODS[2], bundle)
            before = state.decide(summary, specific, mask).scores
            # The Decision class stores scores under its declared field name.
            after = state.decide(summary, specific+3., mask).scores
            np.testing.assert_array_equal(before, after)
            self.assertFalse(np.allclose(candidate.decide(summary, specific, mask).scores,
                                         candidate.decide(summary, specific+3., mask).scores))

    def test_no_candidate_hv_feedback_is_computed_in_inference(self):
        f, c = self.fixture, self.controller
        c.current_individual, c.current_population = deepcopy(f.parent), []
        with patch.object(base, "_evaluate_individual_impl", side_effect=AssertionError("unselected evaluation")):
            decision, _ = c._choose_operator(c.current_individual, f.batches, f.paths, f.tt, f.lookup, f.options)
        self.assertEqual(decision.chosen_index, 4)
        self.assertIsNone(c.feedback_pending)
        self.assertEqual(c.feedback_evaluations, 0)

    def test_search_summary_requires_full_paired_design(self):
        with self.assertRaisesRegex(ValueError, "missing predeclared"):
            summarize([])
        runs = []
        for name in TEST_INSTANCES:
            for seed in TEST_SEEDS:
                for index, method in enumerate(METHODS):
                    runs.append({"instance": name, "algorithm_seed": seed, "strategy": method,
                        "pairing": {"initial": "same", "scenarios": "same"}, "real_objective_evaluation_count": 400,
                        "ppo_updates": 0, "training_feedback_evaluations": 0,
                        "candidate_contract": "concrete-proposal-v1", "minimum_control_generation": 10,
                        "model_bundle_sha256": "same", "final_hypervolume": 1.+.01*index,
                        "optimisation_seconds": 1., "candidate_construction_seconds": .1, "closed_loop_switches": index})
        report = summarize(runs)
        self.assertTrue(report["candidate_information_promising"])
        self.assertFalse(report["automatic_ppo"])
        broken = deepcopy(runs)
        broken[-1]["training_feedback_evaluations"] = 1
        with self.assertRaisesRegex(ValueError, "contract violated"):
            summarize(broken)


if __name__ == "__main__":
    unittest.main()
