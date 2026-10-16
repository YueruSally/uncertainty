import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_rf_reward_cplus_batch import (
    COLLECTION_METHOD, _completed, _select_records, _seeds,
)


def _catalog():
    return {
        "instances": [
            {
                "instance_id": f"train-S{configuration}-{index:03d}",
                "split": "train",
                "configuration": f"S{configuration}",
                "instance_digest": f"digest-{configuration}-{index}",
            }
            for configuration in range(9)
            for index in range(1, 51)
        ]
    }


class SelectionTests(unittest.TestCase):
    def test_selection_is_balanced_deterministic_and_seeded(self):
        first = _select_records(_catalog(), 5, 82_000_000)
        repeat = _select_records(_catalog(), 5, 82_000_000)
        other = _select_records(_catalog(), 5, 82_000_001)
        self.assertEqual(first, repeat)
        self.assertNotEqual(first, other)
        self.assertEqual(len(first), 45)
        self.assertEqual(
            {f"S{i}": 5 for i in range(9)},
            {configuration: sum(
                row["configuration"] == configuration for row in first)
             for configuration in {row["configuration"] for row in first}})

    def test_selection_rejects_nonformal_training_catalog(self):
        catalog = _catalog()
        catalog["instances"].pop()
        with self.assertRaises(ValueError):
            _select_records(catalog, 5, 1)


class ResumeValidationTests(unittest.TestCase):
    def test_completed_collection_requires_exact_frozen_metadata(self):
        record = _catalog()["instances"][0]
        args = SimpleNamespace(
            evaluation_budget=1000, uniform_mixture=0.5,
            normalization_digest="normalization-sha",
            algorithm_seed_base=83_000_000, policy_seed_base=93_000_000,
            pop=100, gens=1000,
        )
        algorithm_seed, policy_seed = _seeds(args, 0, 0)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "COMPLETE.json").write_text(json.dumps({
                "completed": True, "method": COLLECTION_METHOD,
                "evaluation_budget": 1000, "ccp_evaluation_count": 997,
            }))
            (root / "configuration.json").write_text(json.dumps({
                "instance_id": record["instance_id"],
                "instance_digest": record["instance_digest"],
                "population": 100, "generations": 1000,
            }))
            (root / "rf_collection_configuration.json").write_text(json.dumps({
                "algorithm_seed": algorithm_seed, "policy_seed": policy_seed,
                "uniform_mixture": 0.5,
                "normalization_sha256": "normalization-sha",
            }))
            (root / "final_feasible_nondominated.json").write_text("[]")
            self.assertTrue(_completed(root, record, args, 0, 0))
            complete = json.loads((root / "COMPLETE.json").read_text())
            complete["ccp_evaluation_count"] = 1001
            (root / "COMPLETE.json").write_text(json.dumps(complete))
            with self.assertRaises(ValueError):
                _completed(root, record, args, 0, 0)


if __name__ == "__main__":
    unittest.main()
