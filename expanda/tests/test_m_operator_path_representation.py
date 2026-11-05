import copy
from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from m_operator_offline import digest, load_panel, masked_loss, read_json, write_json
from m_operator_path_representation import (
    ARMS, MatchedPathTransformer, build_parent_inputs, fit, fit_parent_scaler,
    pack_paths, parent_arrays, predict, weight_digest,
)
from run_m_operator_path_representation_v1 import execute
from tests.test_m_operator_offline_v1 import make_panel, silent_execute


def example_parent(costs=((1., 2.), (4., 8.))):
    """Synthetic paths with equal-length groups and distinct arc attributes."""
    parent = []
    for values in costs:
        header = np.zeros(19)
        header[[0, 1, 7]] = [10., .5, len(values)]
        parent.append(header.copy())
        for i, cost in enumerate(values):
            arc = header.copy()
            arc[[2, 3, 4, 5, 6, 8]] = [12, cost, 3, 40, i / max(1, len(values) - 1), 1]
            parent.append(arc)
    return np.asarray(parent).tolist()


def structure_panel(root):
    rows = make_panel(root)
    for i, row in enumerate(rows):
        signal = 1 if i % 2 else -1
        row["observation"]["parent"] = example_parent(((2 + signal, 3), (4, 5 - signal)))
    data = read_json(root / "batch_2/DATASET.json")
    data["states"] = rows
    write_json(root / "batch_2/DATASET.json", data)
    return rows


class OperatorPathRepresentationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def prepare(self, parents):
        rows = [{"observation": {"parent": parent}} for parent in parents]
        arrays, groups = parent_arrays(rows)
        scaler = fit_parent_scaler(arrays, list(range(len(arrays))))
        return {arm: build_parent_inputs(arrays, groups, scaler, arm) for arm in ARMS}, scaler

    def test_only_path_arm_sees_structure_with_identical_summary(self):
        a, b = example_parent(), example_parent()
        # Swap two arcs at the SAME within-path position across paths. This
        # preserves every marginal mean/std, but changes feature membership.
        b[2], b[5] = b[5], b[2]
        inputs, _ = self.prepare([a, b])
        np.testing.assert_array_equal(inputs["SUMMARY"][0][0], inputs["SUMMARY"][1][0])
        self.assertFalse(np.array_equal(inputs["PATH_TOKEN"][0][0], inputs["PATH_TOKEN"][1][0]))
        torch.manual_seed(17)
        model = MatchedPathTransformer(7)
        x = np.ones((2, 5, 7), dtype=np.float32)
        mask = np.ones((2, 5), bool)
        summaries = predict(model, x, mask, inputs["SUMMARY"])
        paths = predict(model, x, mask, inputs["PATH_TOKEN"])
        np.testing.assert_allclose(summaries[0], summaries[1], atol=5e-7)
        self.assertGreater(float(np.max(abs(paths[0] - paths[1]))), 1e-6)

    def test_path_order_invariant_but_arc_positions_preserved(self):
        parent = example_parent()
        reversed_paths = parent[3:] + parent[:3]
        inputs, _ = self.prepare([parent, reversed_paths])
        model = MatchedPathTransformer(7)
        x = np.ones((2, 5, 7), np.float32)
        m = np.ones((2, 5), bool)
        out = predict(model, x, m, inputs["PATH_TOKEN"])
        np.testing.assert_allclose(out[0], out[1], atol=2e-7)
        # Path position is retained, not replaced by a global/arbitrary path ID.
        self.assertNotEqual(inputs["PATH_TOKEN"][0][0][1, 6], inputs["PATH_TOKEN"][0][0][2, 6])

    def test_padding_and_illegal_nan_targets_do_not_contaminate_gradients(self):
        inputs, _ = self.prepare([example_parent(), example_parent(((1, 2, 3),))])
        model = MatchedPathTransformer(7)
        x = torch.randn(2, 5, 7)
        m = torch.tensor([[True, False, False, True, True]] * 2)
        packed, valid, counts = pack_paths(inputs["PATH_TOKEN"], [0, 1])
        ordinary = model(x, m, packed, valid, counts)
        extra = torch.cat([packed, torch.full((len(packed), 2, 21), 10.)], dim=1)
        extended_mask = torch.cat([valid, torch.zeros((len(valid), 2), dtype=torch.bool)], dim=1)
        extended = model(x, m, extra, extended_mask, counts)
        torch.testing.assert_close(ordinary, extended, atol=5e-7, rtol=1e-5)
        target = torch.randn(2, 5)
        target[~m] = float("nan")
        masked_loss(ordinary, target, m).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        self.assertTrue(torch.all(ordinary[:, 4] == 0))
        with self.assertRaisesRegex(ValueError, "all-masked"):
            model(x, m, packed, torch.zeros_like(valid), counts)

    def test_scaler_training_only_and_no_outcome_inputs(self):
        rows = [{"observation": {"parent": example_parent()}},
                {"observation": {"parent": example_parent(((10000, 20000),))}}]
        parents, groups = parent_arrays(rows)
        scaler = fit_parent_scaler(parents, [0])
        expected = copy.deepcopy(scaler)
        rows[1]["future_reward"] = 1e12
        rows[1]["observation"]["best_operator"] = "mode"
        parents[1][:, 3] *= 100
        self.assertEqual(expected, fit_parent_scaler(parents, [0]))
        inputs = build_parent_inputs(parents, groups, scaler, "PATH_TOKEN")
        self.assertLessEqual(max(abs(p).max() for paths in inputs for p in paths), 10.)
        bad = copy.deepcopy(rows)
        bad[0]["observation"]["parent"][2][6] = .123
        with self.assertRaisesRegex(ValueError, "position"):
            parent_arrays(bad)

    def test_same_weights_and_parameter_count_for_both_arms(self):
        digests, counts = [], []
        for _ in ARMS:
            torch.manual_seed(101)
            model = MatchedPathTransformer(145)
            digests.append(weight_digest(model))
            counts.append(sum(p.numel() for p in model.parameters()))
        self.assertEqual(digests[0], digests[1])
        self.assertEqual(counts[0], counts[1])

    def test_microbatch_gradient_equals_effective_batch_gradient(self):
        inputs, _ = self.prepare([example_parent()] * 8)
        torch.manual_seed(5)
        model = MatchedPathTransformer(7)
        other = copy.deepcopy(model)
        x = torch.randn(8, 5, 7)
        m = torch.tensor([[True, i % 3 == 0, i % 3 == 0, True, True] for i in range(8)])
        target = torch.randn(8, 5)
        target[~m] = float("nan")
        output = model(x, m, *pack_paths(inputs["PATH_TOKEN"], range(8)))
        masked_loss(output, target, m).backward()
        for start in (0, 4):
            ids = np.arange(start, start + 4)
            output = other(x[ids], m[ids], *pack_paths(inputs["PATH_TOKEN"], ids))
            (masked_loss(output, target[ids], m[ids]) * (m[ids, :4].sum() / m[:, :4].sum())).backward()
        for p, q in zip(model.parameters(), other.parameters()):
            torch.testing.assert_close(p.grad, q.grad, atol=5e-7, rtol=2e-5)

    def test_path_model_can_fit_structure_signal_not_visible_in_summaries(self):
        parents = []
        for i in range(32):
            p = example_parent()
            if i % 2:
                p[2], p[5] = p[5], p[2]
            parents.append(p)
        inputs, _ = self.prepare(parents)
        x = np.tile(np.eye(5, dtype=np.float32)[None, :, :], (32, 1, 1))
        m = np.ones((32, 5), bool)
        y = np.zeros((32, 5), np.float32)
        y[:, 0] = np.where(np.arange(32) % 2, 1, -1)
        model, result = fit(x, m, y, inputs["PATH_TOKEN"], np.arange(32), np.arange(32), seed=47, max_updates=160)
        # Deliberate overfit is only a plumbing check, not a generalization claim.
        history = result["history"]
        self.assertLess(min(r["training_mse"] for r in history), history[0]["training_mse"] * .2)
        pred = predict(model, x, m, inputs["PATH_TOKEN"])
        self.assertGreater(np.mean((pred[:, 0] > 0) == (y[:, 0] > 0)), .95)

    def test_outer_labels_do_not_change_weights_or_selected_checkpoint(self):
        rng = np.random.default_rng(28)
        inputs, _ = self.prepare([example_parent()] * 96)
        x = rng.normal(size=(96, 5, 7)).astype(np.float32)
        m = np.ones((96, 5), bool)
        y = rng.normal(size=(96, 5)).astype(np.float32)
        y[:, 4] = 0
        changed = y.copy()
        changed[64:, :4] = 1e6
        original, a = fit(x, m, y, inputs["PATH_TOKEN"], np.arange(32), np.arange(32, 64), 81, max_updates=8)
        perturbed, b = fit(x, m, changed, inputs["PATH_TOKEN"], np.arange(32), np.arange(32, 64), 81, max_updates=8)
        self.assertEqual(a["selected_update"], b["selected_update"])
        self.assertEqual(weight_digest(original), weight_digest(perturbed))

    def test_full_paired_pipeline_preserves_sources_resume_and_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            data, offline, out = [Path(tmp) / name for name in ("data", "offline", "path")]
            structure_panel(data)
            offline.mkdir()
            silent_execute(data, offline, epochs=1, patience=8)
            before = {p: digest(p) for p in list(data.rglob("*.json")) + list(offline.rglob("*.json"))}
            out.mkdir()
            with redirect_stdout(io.StringIO()):
                result = execute(data, offline, out)
            self.assertTrue(result["paired_initialization_exact"])
            self.assertEqual(result["new_objective_evaluations"], 0)
            self.assertEqual(result["new_ppo_updates"], 0)
            self.assertEqual(result["methods"]["PATH_TOKEN"]["state_count"], 384)
            self.assertTrue((out / "REPRESENTATION_COMPLETE").exists())
            self.assertEqual(before, {p: digest(p) for p in before})
            (out / "REPRESENTATION_COMPLETE").unlink()
            # Reuse complete fits after losing a fold's summary, without training.
            (out / "fixture-0/FOLD_COMPLETE.json").unlink()
            with redirect_stdout(io.StringIO()):
                resumed = execute(data, offline, out)
            self.assertEqual(result["methods"], resumed["methods"])
            plan = read_json(out / "PLAN.json")
            plan["max_optimizer_updates"] += 8
            write_json(out / "PLAN.json", plan)
            with self.assertRaisesRegex(ValueError, "plan changed"):
                execute(data, offline, out)


if __name__ == "__main__":
    unittest.main()
