import json
import io
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stdout

import joblib
import pandas as pd

from segment_learning_features import (CATEGORICAL_FEATURES, FEATURES,
                                       NUMERIC_FEATURES)
from train_segment_models import main


class TrainSegmentModelsTests(unittest.TestCase):
    def test_trains_requested_target_and_skips_deployment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            for i in range(24):
                row = {name: float(i + 1) for name in NUMERIC_FEATURES}
                row.update({name: f"value-{i % 2}" for name in CATEGORICAL_FEATURES})
                row.update({
                    "run_number": 1 if i < 16 else 2,
                    "event_id": f"event-{i}", "operator": "mode",
                    "split": "train" if i < 16 else "validation",
                    "cost_improved": bool(i % 2),
                    "cost_improved_eligible": True,
                })
                rows.append(row)
            dataset = root / "dataset.csv"
            manifest = root / "manifest.json"
            out = root / "model"
            pd.DataFrame(rows, columns=list(dict.fromkeys(
                FEATURES + list(rows[0])))).to_csv(dataset, index=False)
            manifest.write_text(json.dumps({
                "targets": ["cost_improved"], "train_runs": [1],
                "validation_runs": [2]}), encoding="utf-8")
            with redirect_stdout(io.StringIO()):
                result = main([
                    "--dataset", str(dataset), "--manifest", str(manifest),
                    "--out", str(out), "--targets", "cost_improved", "--trees", "8",
                ])
            self.assertEqual(result, 0)
            artifact = joblib.load(out / "segment_models.joblib")
            self.assertEqual(artifact["schema_version"], 2)
            self.assertEqual(artifact["deployment_status"], "offline_validation_only")
            self.assertIn("cost_improved", artifact["pipelines"])
            report = json.loads((out / "training_metrics.json").read_text())
            self.assertEqual(report["targets"]["cost_improved"]["status"], "trained")


if __name__ == "__main__":
    unittest.main()
