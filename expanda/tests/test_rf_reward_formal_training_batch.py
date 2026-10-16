import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_rf_reward_formal_training_batch import (
    ALGORITHM_SEED_BASE, EVALUATION_BUDGET, FIXED_PARAMETERS,
    INSTANCES_PER_CONFIGURATION, POLICY_SEED_BASE, TRAJECTORIES,
    EXPECTED_NORMALIZATION_SHA256,
    _apply_frozen_args, _batch_configuration, _formal_model_summary,
)


class FrozenProtocolTests(unittest.TestCase):
    def test_protocol_has_expected_scale_and_disjoint_seed_namespaces(self):
        args = _apply_frozen_args(SimpleNamespace())
        self.assertEqual(args.instances_per_config, 50)
        self.assertEqual(args.trajectories, 3)
        self.assertEqual(args.evaluation_budget, 5000)
        self.assertEqual(args.uniform_mixture, 0.5)
        self.assertEqual(INSTANCES_PER_CONFIGURATION * 9 * TRAJECTORIES,
                         1350)
        self.assertEqual(450 * TRAJECTORIES * EVALUATION_BUDGET, 6_750_000)
        self.assertGreater(POLICY_SEED_BASE, ALGORITHM_SEED_BASE + 3_000_000)

    def test_batch_configuration_records_all_selected_instances(self):
        selected = [{
            "instance_id": f"train-S{configuration}-{index:03d}",
            "configuration": f"S{configuration}",
            "instance_digest": f"digest-{configuration}-{index}",
        } for configuration in range(9) for index in range(1, 51)]
        args = SimpleNamespace(
            catalog_digest="catalog",
            normalization_digest=EXPECTED_NORMALIZATION_SHA256,
            workers=4, rf_jobs=4)
        config = _batch_configuration(args, selected, "source")
        self.assertEqual(config["collection_count"], 1350)
        self.assertEqual(config["maximum_ccp100_evaluations"], 6_750_000)
        self.assertEqual(config["fixed_rf_parameters"], FIXED_PARAMETERS)
        self.assertEqual(len(config["selected_instances"]), 450)

    def test_completed_model_requires_frozen_parameters_and_hash(self):
        expected_instances = ["train-S0-001", "train-S1-001"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_dir = root / "model"
            model_dir.mkdir()
            model = model_dir / "rf_reward.joblib"
            model.write_bytes(b"model")
            import hashlib
            model_hash = hashlib.sha256(b"model").hexdigest()
            summary = {
                "instance_count": 2,
                "rf_seed": 170_000_000,
                "cv_seed": 180_000_000,
                "grouped_validation_folds": 5,
                "configuration_balanced_folds": True,
                "fixed_parameters": FIXED_PARAMETERS,
                "parameter_protocol": (
                    "predeclared fixed parameters; CV is diagnostic only"),
                "model_sha256": model_hash,
                "sample_weighting": "none",
                "maximum_archive_delta_error": 0.0,
                "best": {"validation": {"per_instance": [
                    {"instance_id": value} for value in expected_instances]}},
            }
            (model_dir / "training_summary.json").write_text(json.dumps(summary))
            self.assertEqual(
                _formal_model_summary(root, expected_instances, 170_000_000),
                summary)
            summary["fixed_parameters"] = dict(FIXED_PARAMETERS, max_depth=9)
            (model_dir / "training_summary.json").write_text(json.dumps(summary))
            with self.assertRaises(ValueError):
                _formal_model_summary(root, expected_instances, 170_000_000)


if __name__ == "__main__":
    unittest.main()
