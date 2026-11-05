from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from analyze_m_operator_offline_v1 import decompose, gate_diagnostic, inspect
from m_operator_offline import digest, read_json, write_json
from tests.test_m_operator_offline_v1 import make_panel, silent_execute


class OfflineDiagnosisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_separates_disagreement_penalty_from_threshold(self):
        rows = [{"instance": "i", "run": "trajectory", "state_id": i,
                 "phase": "early"} for i in range(2)]
        predictions = np.array([
            [[1.2, 99, 99, -0.1, 0], [-0.1, 99, 99, 0.2, 0]],
            [[0, 99, 99, -0.1, 0], [-0.1, 99, 99, 0.2, 0]],
            [[0, 99, 99, -0.1, 0], [-0.1, 99, 99, 0.2, 0]],
        ])
        mask = np.array([[True, False, False, True, True]] * 2)
        targets = np.array([[0.1, np.nan, np.nan, -0.1, 0], [-0.1, np.nan, np.nan, -0.2, 0]])
        report = decompose(rows, predictions, mask, targets, 0.3)
        self.assertEqual(report["mean_prediction_zero_threshold"]["metrics"]["switch_count"], 2)
        self.assertEqual(report["minus_disagreement_zero_threshold"]["metrics"]["switch_count"], 1)
        self.assertEqual(report["frozen_policy"]["metrics"]["switch_count"], 0)
        self.assertEqual(report["layer_effects"]["switches_removed_by_disagreement"], 1)
        self.assertEqual(report["layer_effects"]["switches_removed_by_threshold"], 1)
        self.assertAlmostEqual(report["mean_prediction_zero_threshold"]["metrics"]["mean_net_gain_all_states"], -0.05)

    def test_gate_diagnostic_preserves_original_threshold(self):
        pred = np.array([[1, 0, 0, 0, 0]], float)
        mask = np.ones((1, 5), bool)
        positive = np.array([[0.1, -0.1, -0.1, -0.1, 0]])
        report = gate_diagnostic(pred, mask, positive, None, 1.0)
        self.assertTrue(report["always_balanced"])
        self.assertIsNone(report["frozen_threshold"])
        self.assertGreater(report["positive_inner_validation_grid_choices"], 0)
        negative = -positive
        self.assertEqual(gate_diagnostic(pred, mask, negative, None, 1.0)["positive_inner_validation_grid_choices"], 0)

    def test_saved_fit_diagnosis_is_readonly_and_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            collection, offline = Path(tmp) / "data", Path(tmp) / "offline"
            make_panel(collection)
            offline.mkdir()
            original = silent_execute(collection, offline, epochs=1, patience=1)
            files = list(collection.rglob("*.json")) + list(offline.rglob("*.json")) + list(offline.rglob("*.pt"))
            before = {path: digest(path) for path in files}
            with redirect_stdout(io.StringIO()):
                report = inspect(collection, offline)
            summary = report["summary"]
            self.assertTrue(summary["posthoc_diagnostic_only"])
            self.assertTrue(summary["saved_decisions_reproduced_exactly"])
            self.assertEqual(summary["original_offline_screen_pass"], original["offline_screen_pass"])
            self.assertEqual(summary["checkpoint_count"], 18)
            self.assertEqual(summary["new_training_steps"], 0)
            self.assertEqual(summary["new_objective_evaluations"], 0)
            self.assertEqual(before, {path: digest(path) for path in files})
            model_report = offline / "fixture-0/FOLD_COMPLETE.json"
            value = read_json(model_report)
            record = value["methods"]["TRANSFORMER_ENSEMBLE"]["decisions"][0]
            record["selected_operator"] = "mode" if record["selected_operator"] == "replace" else "replace"
            write_json(model_report, value)
            with self.assertRaisesRegex(ValueError, "reproduce original"):
                inspect(collection, offline)


if __name__ == "__main__":
    unittest.main()
