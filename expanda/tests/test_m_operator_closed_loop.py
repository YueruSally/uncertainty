import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from m_operator_closed_loop import FrozenPathPolicy, FrozenOperatorController, export_frozen_policy
from m_operator_offline import choose, digest, fit_scaler, preaction_features, transform, write_json
from m_operator_path_representation import MatchedPathTransformer, parent_arrays, fit_parent_scaler, build_parent_inputs, predict
from run_m_operator_closed_loop_v1 import METHODS, paired_results
from tests.test_m_operator_path_representation import example_parent


def fixture(root, threshold=0., feature_dimensions=None):
    """Random test weights, not an experiment/model trained on transport data."""
    nc, ng = feature_dimensions or (5, 2)
    obs = {"parent": example_parent(), "global": [0.] * ng,
        "candidates": np.ones((5, nc)).tolist(), "eligible": [True] * 5}
    x, mask = preaction_features(obs)
    common = fit_scaler(x[None], mask[None], np.array([[.001, -.002, .003, -.001, 0.]]))
    arrays, groups = parent_arrays([{"observation": obs}])
    parent = fit_parent_scaler(arrays, [0])
    preprocessing = {"common_candidate_context": common, "shared_parent_columns": parent}
    paths, models, hashes = [], [], {}
    for seed in (1, 2, 3):
        torch.manual_seed(seed)
        m = MatchedPathTransformer(x.shape[-1]).eval()
        path = root / f"seed_{seed}.pt"
        torch.save({"arm": "PATH_TOKEN", "model": m.state_dict(), "feature_count": x.shape[-1],
                    "preprocessing": preprocessing}, path)
        paths.append(str(path))
        hashes[str(path)] = digest(path)
        models.append(m)
    bundle = root / "bundle.json"
    write_json(bundle, {"selected": {"source_fold": "fixture", "frozen_threshold": threshold,
        "checkpoints": paths}, "input_sha256": hashes})
    inputs = build_parent_inputs(arrays, groups, parent, "PATH_TOKEN")
    preds = np.asarray([predict(m, transform(x[None], common), mask[None], inputs)[0] * common["target_scale"] for m in models])
    return bundle, obs, preds.mean(0) - preds.std(0)


class ClosedLoopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_frozen_inference_matches_offline_scores_and_gate(self):
        with tempfile.TemporaryDirectory() as d:
            bundle, obs, expected = fixture(Path(d))
            policy = FrozenPathPolicy(bundle)
            actual, _ = policy.predict_observation(obs)
            np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-12)
            policy.observation_provider = lambda *_: obs
            decision = policy.choose([{"eligible": True}] * 5)
            self.assertEqual(decision.chosen_index, choose(expected[None], np.ones((1, 5), bool), 0.)[0])
            self.assertTrue(all(not p.requires_grad and p.grad is None for m in policy.models for p in m.parameters()))
            self.assertEqual(policy.updates, 0)

    def test_frozen_null_gate_and_unsupported_mask(self):
        with tempfile.TemporaryDirectory() as d:
            bundle, obs, _ = fixture(Path(d), threshold=None)
            for heuristic in (False, True):
                policy = FrozenPathPolicy(bundle, heuristic=heuristic)
                policy.observation_provider = lambda *_: obs
                self.assertEqual(policy.choose([{"eligible": True}] * 5).chosen_index, 4)
                self.assertIsNone(policy.choose([{"eligible": True}] * 4 + [{"eligible": False}]).chosen_index)
                self.assertEqual(policy.switches, 0)

    def test_changed_checkpoint_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            bundle, _, _ = fixture(root)
            (root / "seed_1.pt").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                FrozenPathPolicy(bundle)

    def test_older_torch_load_without_weights_only(self):
        with tempfile.TemporaryDirectory() as d:
            bundle, obs, expected = fixture(Path(d))
            original = torch.load

            def older_load(path, *args, **kwargs):
                if "weights_only" in kwargs:
                    raise TypeError("'weights_only' is an invalid keyword argument for Unpickler()")
                return original(path, *args, **kwargs)

            with patch("m_operator_closed_loop.torch.load", side_effect=older_load):
                policy = FrozenPathPolicy(bundle)
            actual, _ = policy.predict_observation(obs)
            np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-12)

    def test_controller_schedule_preserves_early_base(self):
        controller = object.__new__(FrozenOperatorController)
        controller.mixture_event_count = 0
        for generation, expected in ((0, False), (9, False), (10, True), (18, True)):
            controller.generation = generation
            self.assertEqual(controller._next_control_branch()[0], expected)

    def test_selection_uses_inner_validation_not_outer_results(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            collection, path = root / "collection", root / "path"
            write_json(collection / "batch_2/DATASET.json", {"fixture": True})
            folds = [{"test": "a", "validation": "v", "training": ["z"]},
                     {"test": "b", "validation": "v", "training": ["z"]}]
            write_json(path / "PLAN.json", {"protocol": "operator-path-representation-v1", "folds": folds,
                "seeds": [1], "inputs_sha256": {str((collection / "batch_2/DATASET.json").resolve()): digest(collection / "batch_2/DATASET.json")}})
            (path / "REPRESENTATION_COMPLETE").touch()
            write_json(path / "QUICK_RESULT.json", {"representation_screen_pass": True})
            for name in ("a", "b"):
                folder = path / name
                write_json(folder / "FOLD_COMPLETE.json", {"preprocessing": {"common_candidate_context": {"target_scale": .1}},
                    "methods": {"PATH_TOKEN": {"threshold": None if name == "a" else .05, "outer_gain": 999 if name == "a" else -999}}})
                checkpoint = folder / "PATH_TOKEN/seed_1.pt"
                checkpoint.parent.mkdir()
                checkpoint.write_bytes(b"test only")
                write_json(checkpoint.with_suffix(".json"), {"checkpoint_sha256": digest(checkpoint), "predictions": [[1., 0., 0., 0., 0.]]})
            panel = ([{"instance": "v"}], None, np.ones((1, 5), bool), np.array([[.2, -.1, -.1, -.1, 0.]]), None, None)
            with patch("m_operator_closed_loop.load_panel", return_value=panel):
                result = export_frozen_policy(path, collection, root / "frozen.json")
            self.assertEqual(result["selected"]["source_fold"], "b")
            write_json(path / "QUICK_RESULT.json", {"representation_screen_pass": False})
            with self.assertRaisesRegex(ValueError, "offline representation screen"):
                export_frozen_policy(path, collection, root / "blocked.json")
            write_json(path / "QUICK_RESULT.json", {"representation_screen_pass": True})
            write_json(path / "b/FOLD_COMPLETE.json", {"preprocessing": {"common_candidate_context": {"target_scale": .1}},
                "methods": {"PATH_TOKEN": {"threshold": 0.}}})
            with patch("m_operator_closed_loop.load_panel", return_value=panel):
                with self.assertRaisesRegex(ValueError, "positive threshold"):
                    export_frozen_policy(path, collection, root / "zero_gate.json")

    def test_pairing_and_per_instance_pass_criteria(self):
        runs = []
        for instance in ("a", "b"):
            for method, hv in zip(METHODS, (1., 1.1, 1.2)):
                runs.append({"strategy": method, "instance": instance, "algorithm_seed": 1,
                    "pairing": {"scenario": "same", "initial_population": [1]}, "evaluation_budget": 400,
                    "real_objective_evaluation_count": 400, "ppo_updates": 0, "worker_seconds": 2.,
                    "final_hypervolume": hv, "optimisation_seconds": 1., "closed_loop_switches": 1})
        result = paired_results(runs)
        self.assertTrue(result["screen_pass"])
        self.assertAlmostEqual(result["per_instance"][0]["transformer_minus_heuristic_percentage_points"], 10.)
        other = copy.deepcopy(runs)
        other[-1]["pairing"]["scenario"] = "wrong"
        with self.assertRaisesRegex(ValueError, "pairing mismatch"):
            paired_results(other)
        runs[-1]["final_hypervolume"] = 1.1
        self.assertFalse(paired_results(runs)["screen_pass"])


if __name__ == "__main__":
    unittest.main()
