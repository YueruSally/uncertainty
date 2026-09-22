import json
from pathlib import Path
import tempfile
import unittest

import joblib

from promote_segment_models import OBJECTIVE_HEADS, RISK_HEAD, main


class PromoteSegmentModelsTests(unittest.TestCase):
    def test_promotion_freezes_policy_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "offline.joblib"
            metrics = root / "metrics.json"
            output = root / "pilot.joblib"
            required = OBJECTIVE_HEADS + [RISK_HEAD]
            joblib.dump({
                "schema_version": 2, "deployment_status": "offline_validation_only",
                "pipelines": {target: f"pipeline-{target}" for target in required},
            }, source)
            metrics.write_text(json.dumps({
                "oos_used": False, "train_runs": [1, 2], "validation_runs": [3],
                "targets": {target: {"status": "trained"} for target in required},
            }), encoding="utf-8")
            self.assertEqual(main([
                "--model", str(source), "--metrics", str(metrics),
                "--out", str(output)]), 0)
            promoted = joblib.load(output)
            self.assertEqual(promoted["deployment_status"], "pilot_only")
            self.assertEqual(promoted["policy"]["guided_operator"], "mode")
            self.assertEqual(promoted["policy"]["objective_heads"], OBJECTIVE_HEADS)
            self.assertTrue(output.with_suffix(".manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
