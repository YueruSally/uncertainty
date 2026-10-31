import json
import tempfile
import unittest
from pathlib import Path

from analyze_m_hybrid_policy import build_report


class MHybridPolicyTests(unittest.TestCase):
    def test_joins_oof_state_gate_and_transformer_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            transformer = root / "transformer"
            gate_rows = []
            for instance, probability, selected_gain, oracle in (
                    ("S0_001", 0.9, 12.0, 15.0),
                    ("S0_002", 0.1, -2.0, 0.0)):
                key = f"{instance}:777001:1"
                fold = transformer / f"heldout_{instance}"
                fold.mkdir(parents=True)
                (fold / "report.json").write_text(json.dumps({
                    "heldout_instance": instance,
                    "validation": [{
                        "instance": instance,
                        "state_key": key,
                        "generation": 1,
                        "oracle_percent": oracle,
                        "selected_candidate_index": 3,
                        "selected_gain_percent": selected_gain,
                    }],
                }), encoding="utf-8")
                gate_rows.append({
                    "state_key": key,
                    "instance": instance,
                    "generation": 1,
                    "oracle_percent": oracle,
                    "probability": probability,
                })
            gate_path = root / "gate.json"
            gate_path.write_text(json.dumps({
                "model": "balanced_logistic_state_gate",
                "labels": {"positive_10": {
                    "loo_roc_auc": 1.0,
                    "loo_average_precision": 1.0,
                    "predictions": gate_rows,
                }},
            }), encoding="utf-8")

            report = build_report(gate_path, transformer, [0.5], "positive_10")

            self.assertTrue(report["state_keys_exact"])
            self.assertEqual(report["state_count"], 2)
            selected = report["threshold_sweep"][0]
            self.assertEqual(selected["selected_states"], 1)
            self.assertEqual(selected["mean_selected_action_gain_percent"], 12.0)
            self.assertEqual(selected["overall_mean_deployed_gain_percent"], 6.0)
            self.assertEqual(selected["positive_folds"], 1)


if __name__ == "__main__":
    unittest.main()
