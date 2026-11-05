import copy
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from m_operator_offline import (
    BALANCED, OPERATORS, PHASES, TinyOperatorTransformer, choose, fit_ridge,
    fit_scaler, gain_metrics, grouped_folds, load_panel, masked_loss,
    preaction_features, predict_ridge, predict_transformer, train_transformer,
    transform, tune_threshold, write_json,
)
from evaluate_m_operator_offline_v1 import execute


def silent_execute(*args, **kwargs):
    # Synthetic fixture output must not look like the user's experimental result.
    with redirect_stdout(io.StringIO()):
        return execute(*args, **kwargs)


def make_panel(root):
    """Synthetic fixtures ONLY; never claimed as transport experiment results."""
    plan = {"synthetic_test_fixture": True, "scenario_count": 100, "ccp_confidence": 0.9,
            "population_size": 20, "evaluation_budget": 400, "replicates_per_legal_operator": 3,
            "execution_contract": "operator_shared_v1"}
    sha = hashlib.sha256(json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    rows, checked = [], []
    for instance in range(6):
        for seed in range(2):
            name = f"fixture-{instance}"
            run = f"/synthetic/{name}/seed_{seed}"
            checked.append({"run": run, "instance": name, "seed": seed, "replay_exact": True})
            for i in range(32):
                signal = 1.0 if i % 2 else -1.0
                legal = [True, i % 3 == 0, i % 3 == 0, True, True]
                q = [0.0015 + 0.0005 * signal, 0.0005, 0.0004, 0.0015 - 0.0005 * signal, 0.0015]
                operators = {a: {"population_delta_hv_by_replicate": [q[j]] * 3}
                             for j, a in enumerate(OPERATORS) if legal[j]}
                phase = PHASES[min(i // 11, 2)]
                rows.append({"instance": name, "run": run, "state_id": i + 1,
                    "phase": phase, "action_order": list(OPERATORS), "balanced_available": True,
                    "eligible_operators": [a for a, ok in zip(OPERATORS, legal) if ok],
                    "operators": operators, "observation": {
                        "parent": [[1.0, 0.4], [2.0, 0.6]], "global": [signal, i / 32],
                        "candidates": np.eye(5).tolist(), "eligible": legal}})
    quality = {"data_quality_pass": True, "replay_exact": True, "state_count": 384,
        "trajectory_count": 12, "instance_count": 6, "scenario_count": 100,
        "population_size": 20, "evaluation_budget_per_trajectory": 400, "new_training_steps": 0,
        "checked_runs": checked, "plan_sha256": sha,
        "phase_counts": {p: sum(r["phase"] == p for r in rows) for p in PHASES}}
    write_json(root / "PLAN.json", plan)
    write_json(root / "BATCH_2_COMPLETE", {"states": 384, "plan_sha256": sha})
    write_json(root / "batch_2/QUALITY.json", quality)
    write_json(root / "batch_2/DATASET.json", {"states": rows, "plan_sha256": sha})
    return rows


class OperatorOfflineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_split_keeps_instances_and_repeats_together(self):
        folds = grouped_folds([f"i{i}" for i in range(6)] * 64)
        self.assertEqual(len(folds), 6)
        for fold in folds:
            self.assertEqual(len(fold["training"]), 4)
            self.assertNotIn(fold["test"], fold["training"] + [fold["validation"]])

    def test_loader_checks_integrity_and_preserves_illegal_mask(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_panel(root)
            rows, x, mask, y, q, _ = load_panel(root)
            self.assertEqual(x.shape[:2], (384, 5))
            self.assertTrue(np.isnan(y[~mask]).all())
            self.assertTrue(np.all(y[:, BALANCED] == 0))
            path = root / "batch_2/DATASET.json"
            data = json.loads(path.read_text())
            data["states"][0]["operators"]["add"]["population_delta_hv_by_replicate"].pop()
            write_json(path, data)
            with self.assertRaisesRegex(ValueError, "three finite repeats"):
                load_panel(root)

    def test_features_ignore_outcomes_and_scale_uses_training_only(self):
        observation = {"candidates": np.eye(5).tolist(), "parent": [[1, 2]],
                       "global": [3], "eligible": [True] * 5}
        original, mask = preaction_features(observation)
        augmented = dict(observation, best_operator="mode", future_reward=1e9)
        np.testing.assert_equal(original, preaction_features(augmented)[0])
        x = np.stack([original, original + 0.1])
        y = np.ones((2, 5))
        scaler = fit_scaler(x, np.tile(mask, (2, 1)), y)
        untouched = copy.deepcopy(scaler)
        transform(x * 1e9, scaler)
        self.assertEqual(untouched, scaler)
        self.assertLessEqual(abs(transform(x * 1e9, scaler)).max(), 10)

    def test_masked_nan_labels_do_not_poison_backward(self):
        model = TinyOperatorTransformer(7)
        x = torch.randn(8, 5, 7)
        mask = torch.tensor([[True, False, False, True, True]] * 8)
        target = torch.randn(8, 5)
        target[~mask] = float("nan")
        loss = masked_loss(model(x, mask), target, mask)
        loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        with self.assertRaisesRegex(ValueError, "all-masked"):
            model(x, torch.zeros_like(mask))

    def test_ridge_can_learn_different_state_dependence_per_operator(self):
        rng = np.random.default_rng(4)
        x = rng.normal(size=(40, 5, 2)).astype(np.float32)
        mask = np.ones((40, 5), bool)
        y = np.zeros((40, 5))
        y[:, 0] = x[:, 0, 0]
        y[:, 3] = -x[:, 0, 0]
        model = fit_ridge(x, mask, y, alpha=0.01)
        pred = predict_ridge(model, x)
        self.assertGreater(np.mean(choose(pred, mask, 0) == np.where(x[:, 0, 0] > 0, 0, 3)), 0.95)

    def test_gate_never_selects_illegal_actions_and_prefers_fallback_on_ties(self):
        pred = np.array([[999.0, 0.0, 0.0, 0.3, 0.0]] * 4)
        mask = np.array([[False, False, False, True, True]] * 4)
        np.testing.assert_equal(choose(pred, mask, 0), [3] * 4)
        np.testing.assert_equal(choose(pred, mask, 0.4), [4] * 4)
        actual = np.array([[np.nan, np.nan, np.nan, -0.1, 0.0]] * 4)
        self.assertIsNone(tune_threshold(pred, mask, actual, 1.0))
        actual[:, 3] = 0.0
        self.assertIsNone(tune_threshold(pred, mask, actual, 1.0))

    def test_metrics_include_unswitched_zero_returns(self):
        data = [{"mean_heldout_gain": v, "selected_operator": a}
                for v, a in [(2, "mode"), (-1, "add"), (0, "replace"), (0, "replace")]]
        metrics = gain_metrics(data)
        self.assertEqual(metrics["mean_net_gain_all_states"], 0.25)
        self.assertEqual(metrics["mean_gain_when_switched"], 0.5)

    def test_actual_transformer_learns_known_signal(self):
        rng = np.random.default_rng(42)
        x = rng.normal(size=(32, 5, 8)).astype(np.float32)
        x[:, :, :5] = np.eye(5)
        signal = np.linspace(-1, 1, 32).astype(np.float32)
        x[:, :, 5] = signal[:, None]
        mask = np.ones((32, 5), bool)
        y = np.zeros((32, 5), np.float32)
        y[:, 0], y[:, 3] = signal, -signal
        model, result = train_transformer((x, mask, y), (x, mask, y), seed=10, epochs=80, patience=80)
        # Deliberate overfit is a numerical/learnability unit test, not validation.
        pred = predict_transformer(model, x, mask)
        self.assertLess(min(h["train_mse"] for h in result["history"]), 0.1 * result["history"][0]["train_mse"])
        self.assertGreater(np.mean(choose(pred, mask, 0) == np.where(signal > 0, 0, 3)), 0.85)

    def test_full_pipeline_resume_and_input_change_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            collection, out = Path(tmp) / "data", Path(tmp) / "offline"
            make_panel(collection)
            out.mkdir()
            report = silent_execute(collection, out, epochs=1, patience=1)
            self.assertEqual(report["new_objective_evaluations"], 0)
            self.assertEqual(report["new_ppo_updates"], 0)
            self.assertEqual(report["methods"]["RIDGE"]["state_count"], 384)
            self.assertTrue((out / "OFFLINE_COMPLETE").exists())
            # Simulate interruption after five folds; preserve completed fits.
            (out / "OFFLINE_COMPLETE").unlink()
            (out / "fixture-5/FOLD_COMPLETE.json").unlink()
            resumed = silent_execute(collection, out, epochs=1, patience=1)
            self.assertEqual(report["methods"], resumed["methods"])
            with self.assertRaisesRegex(ValueError, "changed"):
                silent_execute(collection, out, epochs=2, patience=1)

    def test_no_repeatable_gain_stops_before_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            collection, out = Path(tmp) / "data", Path(tmp) / "offline"
            make_panel(collection)
            panel = json.loads((collection / "batch_2/DATASET.json").read_text())
            for row in panel["states"]:
                for operator in row["operators"].values():
                    operator["population_delta_hv_by_replicate"] = [0.0] * 3
            write_json(collection / "batch_2/DATASET.json", panel)
            out.mkdir()
            report = silent_execute(collection, out, epochs=1, patience=1)
            self.assertFalse(report["cross_repeat_pass"])
            self.assertFalse(report["training_performed"])
            self.assertEqual(list(out.glob("*/seed_*.pt")), [])


if __name__ == "__main__":
    unittest.main()
