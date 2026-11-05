from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from m_operator_evidence import (
    correlation, decision_quality, ranks, recover_path_groups, repeat_statistics,
    structure_report, training_subsets,
)
from m_operator_offline import digest, grouped_folds, predict_transformer, train_transformer
from run_m_operator_evidence_v1 import execute, train_fixed_updates
from tests.test_m_operator_offline_v1 import make_panel, silent_execute


def parent_fixture():
    header = np.zeros(19)
    header[0], header[1], header[7] = 10, 0.5, 2
    arcs = np.tile(header, (2, 1))
    arcs[:, 8], arcs[:, 6] = 1, [0, 1]
    arcs[:, 2:6] = [10, 2, 3, 40]
    return np.vstack([header, arcs, header, arcs]).tolist()


class OperatorEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_repeat_correlation_and_reliability(self):
        values = np.arange(20, dtype=float) / 100
        reliable = repeat_statistics(np.tile(values[:, None], (1, 3)))
        self.assertAlmostEqual(reliable["estimated_mean_label_reliability_clipped"], 1)
        self.assertTrue(all(abs(r["pearson"] - 1) < 1e-12 for r in reliable["correlations"]))
        self.assertEqual(reliable["mean_within_state_repeat_variance"], 0)

    def test_noise_only_is_not_misreported_as_high_reliability(self):
        noise = np.random.default_rng(41).normal(size=(8000, 3))
        result = repeat_statistics(noise)
        self.assertLess(result["estimated_mean_label_reliability_clipped"], 0.08)
        self.assertTrue(all(abs(r["pearson"]) < 0.06 for r in result["correlations"]))

    def test_constant_or_missing_labels_are_undefined_not_zero_correlation(self):
        for values in (np.zeros((20, 3)), np.full((20, 3), .1), np.zeros((0, 3)), np.ones((1, 3))):
            result = repeat_statistics(values)
            self.assertIsNone(result["correlations"][0]["pearson"])
            self.assertIsNone(result["estimated_mean_label_reliability_clipped"])
        np.testing.assert_equal(ranks([3, 1, 1, 2]), [3, 0.5, 0.5, 2])

    def test_path_boundaries_recovered_without_inventing_batch_ids(self):
        parent = parent_fixture()
        groups = recover_path_groups(parent)
        self.assertEqual([g["arc_indices"] for g in groups], [[1, 2], [4, 5]])
        self.assertTrue(all(g["batch_id"] is None for g in groups))
        report = structure_report([{"observation": {"parent": parent}}])
        self.assertTrue(report["path_grouping_exact"])
        self.assertFalse(report["full_batch_path_token_experiment_ready"])
        parent[-1][6] = 0.2
        with self.assertRaisesRegex(ValueError, "position"):
            recover_path_groups(parent)

    def test_sign_metric_reports_negative_class_baseline_and_positive_recall(self):
        target = np.tile([-9., 1., 0., 0., 0.], (2, 1))
        mask = np.ones((2, 5), bool)
        pred = np.full((2, 5), -1.0)
        result = decision_quality(pred, target, mask, target, mask, 1)
        self.assertAlmostEqual(result["weighted_sign_accuracy"], .9)
        self.assertAlmostEqual(result["always_nonpositive_weighted_accuracy"], .9)
        self.assertEqual(result["positive_gain_weighted_recall"], 0)
        self.assertTrue(result["mse_ratio_undefined"])

    def test_subsets_are_exhaustive_and_never_include_validation_or_test(self):
        for fold in grouped_folds([str(i) for i in range(6)]):
            for size, count in ((2, 6), (3, 4), (4, 1)):
                subsets = training_subsets(fold, size)
                self.assertEqual(len(subsets), count)
                self.assertEqual(len(set(map(tuple, subsets))), count)
                for subset in subsets:
                    self.assertNotIn(fold["test"], subset)
                    self.assertNotIn(fold["validation"], subset)

    def test_fixed_update_schedule_matches_old_four_instance_epoch(self):
        rng = np.random.default_rng(11)
        x = rng.normal(size=(256, 5, 10)).astype(np.float32)
        y = rng.normal(size=(256, 5)).astype(np.float32)
        mask = np.ones((256, 5), bool)
        train, validation = (x, mask, y), (x[:64], mask[:64], y[:64])
        old, old_fit = train_transformer(train, validation, seed=12, epochs=2, patience=8)
        new, new_fit = train_fixed_updates(train, validation, seed=12, max_updates=16)
        np.testing.assert_equal(predict_transformer(old, x, mask), predict_transformer(new, x, mask))
        self.assertEqual(old_fit["selected_epoch"] * 8, new_fit["selected_update"])
        for n in (128, 192):
            _, fit = train_fixed_updates((x[:n], mask[:n], y[:n]), validation, seed=12, max_updates=16)
            self.assertEqual(fit["optimizer_updates"], 16)

    def test_end_to_end_preserves_original_and_resumes(self):
        with tempfile.TemporaryDirectory() as tmp:
            data, offline, evidence = [Path(tmp) / p for p in ("data", "offline", "evidence")]
            make_panel(data)
            offline.mkdir()
            silent_execute(data, offline, epochs=1, patience=8)
            originals = list(offline.rglob("*.json")) + list(data.rglob("*.json"))
            before = {p: digest(p) for p in originals}
            evidence.mkdir()
            with redirect_stdout(io.StringIO()):
                report = execute(data, offline, evidence, sizes=(4,), max_updates=8)
            self.assertTrue(report["completed"])
            self.assertEqual(report["new_objective_evaluations"], 0)
            self.assertEqual(report["new_ppo_updates"], 0)
            self.assertTrue((evidence / "LEARNING_CURVE.png").exists())
            self.assertEqual(before, {p: digest(p) for p in originals})
            (evidence / "EVIDENCE_COMPLETE").unlink()
            with redirect_stdout(io.StringIO()):
                resumed = execute(data, offline, evidence, sizes=(4,), max_updates=8)
            self.assertEqual(report["curve"], resumed["curve"])
            with self.assertRaisesRegex(ValueError, "changed"):
                execute(data, offline, evidence, sizes=(3, 4), max_updates=8)


if __name__ == "__main__":
    unittest.main()
