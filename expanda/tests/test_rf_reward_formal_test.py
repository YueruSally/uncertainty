from pathlib import Path
import hashlib
import json
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import analyze_rf_reward_five_method_formal as analysis
from analyze_rf_reward_five_method_formal import (
    METHODS, _paired_statistics, pooled_bounds, sign_flip_mean_p_value,
)
from run_rf_reward_formal_test_batch import (
    BLOCKS, EVALUATION_BUDGET, EXPECTED_CATALOG_DIGEST,
    EXPECTED_NORMALIZATION_SHA256, _audit_existing_run, _completed_rf,
    _exact_block_mapping,
)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FormalTestProtocolTests(unittest.TestCase):
    def test_frozen_formal_scale_and_inputs(self):
        self.assertEqual(BLOCKS, (1, 2, 3))
        self.assertEqual(120 * len(BLOCKS), 360)
        self.assertEqual(360 * EVALUATION_BUDGET, 5_400_000)
        self.assertEqual(
            EXPECTED_CATALOG_DIGEST,
            "9d4f557c7f61919c88a21df42dc6a2b56b36bcfdb812f8f45a81f9c8bf752087")
        self.assertEqual(
            EXPECTED_NORMALIZATION_SHA256,
            "de73404449c2b612609ad5248f815793f99b676e4260e2068ae403f043addbcd")

    def test_block_mapping_requires_exactly_three_unique_blocks(self):
        mapping = _exact_block_mapping([
            (1, Path("one")), (2, Path("two")), (3, Path("three"))])
        self.assertEqual(set(mapping), {1, 2, 3})
        with self.assertRaises(ValueError):
            _exact_block_mapping([
                (1, Path("one")), (1, Path("again")), (3, Path("three"))])

    def test_existing_control_audit_checks_exact_pairing_metadata(self):
        record = {
            "instance_id": "test-S0-001", "instance_digest": "digest",
            "configuration": "S0", "ccp100_seed": 40,
            "path_seed": 30, "oos5000_seed": 50,
        }
        summary = {
            "algorithm_seed_base": 80_000_000,
            "policy_seed_base": 90_000_000,
            "normalization_digest": "normalization",
            "checkpoints": {},
        }
        block, index = 2, 4
        algorithm_seed, policy_seed = 81_000_004, 91_000_004
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            method_root = root / "instances/test-S0-001/rule"
            run, oos = method_root / "run", method_root / "oos"
            front_path = run / "final_feasible_nondominated.json"
            write_json(front_path, [{"decision_fingerprint": "fixed"}])
            write_json(run / "COMPLETE.json", {
                "completed": True, "method": "Rule-CCP100",
                "stop_reason": "evaluation_budget", "evaluation_budget": 15000,
                "ccp_evaluation_count": 14999,
            })
            write_json(run / "configuration.json", {
                "method": "Rule-CCP100", "instance_id": record["instance_id"],
                "instance_digest": record["instance_digest"], "population": 100,
                "generations": 1000, "evaluation_budget": 15000,
                "scenario_digest": "scenario",
                "seeds": {"algorithm": algorithm_seed, "policy": policy_seed,
                          "training": 40, "path": 30},
            })
            write_json(oos / "oos_5000.json", {
                "instance_id": record["instance_id"],
                "instance_digest": record["instance_digest"], "size": 5000,
                "oos_seed": 50, "source_sha256": sha256(front_path),
                "evaluated_decisions": 1,
                "candidates": [{"decision_fingerprint": "fixed"}],
                "oos_master_digest": "master",
            })
            (oos / "oos_5000.npz").write_bytes(b"npz")
            result = _audit_existing_run(
                root, record, "rule", block, index, summary)
            self.assertEqual(result["algorithm_seed"], algorithm_seed)
            self.assertEqual(result["oos_master_digest"], "master")

    def test_rf_completion_requires_model_and_common_scenario_provenance(self):
        record = {
            "instance_id": "test-S0-001", "instance_digest": "digest",
            "configuration": "S0", "ccp100_seed": 40,
            "path_seed": 30, "oos5000_seed": 50,
        }
        model = {
            "sha256": "model", "training_data_digest": "data", "rf_seed": 170,
        }
        pairing = {
            "algorithm_seed": 80, "scenario_digest": "scenario",
            "oos_master_digest": "master",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run, oos = root / "run", root / "oos"
            front_path = run / "final_feasible_nondominated.json"
            write_json(front_path, [{"decision_fingerprint": "fixed"}])
            write_json(run / "COMPLETE.json", {
                "completed": True, "method": "rf-reward-regressor",
                "stop_reason": "evaluation_budget", "evaluation_budget": 15000,
                "ccp_evaluation_count": 14998,
            })
            write_json(run / "configuration.json", {
                "method": "rf-reward-regressor", "instance_id": record["instance_id"],
                "instance_digest": record["instance_digest"], "population": 100,
                "generations": 1000, "evaluation_budget": 15000,
                "scenario_digest": "scenario", "policy_training": False,
                "seeds": {"algorithm": 80, "training": 40, "path": 30},
                "policy_checkpoint": {
                    "model_sha256": "model", "training_data_digest": "data",
                    "rf_seed": 170, "predict_jobs": 1,
                },
            })
            write_json(oos / "oos_5000.json", {
                "instance_id": record["instance_id"],
                "instance_digest": record["instance_digest"], "size": 5000,
                "oos_seed": 50, "oos_master_digest": "master",
                "source_sha256": sha256(front_path), "evaluated_decisions": 1,
                "candidates": [{"decision_fingerprint": "fixed"}],
            })
            (oos / "oos_5000.npz").write_bytes(b"npz")
            self.assertTrue(_completed_rf(run, oos, record, model, pairing))
            config = json.loads((run / "configuration.json").read_text())
            config["scenario_digest"] = "wrong"
            write_json(run / "configuration.json", config)
            with self.assertRaises(ValueError):
                _completed_rf(run, oos, record, model, pairing)


class FrozenAnalysisTests(unittest.TestCase):
    def test_pooled_nondominated_bounds_and_zero_span_rule(self):
        bounds = pooled_bounds([
            [(1.0, 4.0, 7.0), (4.0, 1.0, 7.0)],
            [(2.0, 2.0, 7.0), (5.0, 5.0, 7.0)],
        ])
        self.assertEqual(bounds["minimum"], [1.0, 1.0, 7.0])
        self.assertEqual(bounds["maximum"], [4.0, 4.0, 7.0])
        self.assertEqual(bounds["scale"], [3.0, 3.0, 1.0])
        self.assertEqual(bounds["zero_span"], [False, False, True])
        self.assertEqual(bounds["pooled_nondominated_size"], 3)

    def test_sign_flip_test_is_two_sided_and_deterministic(self):
        original = analysis.SIGN_FLIP_SAMPLES
        analysis.SIGN_FLIP_SAMPLES = 2_000
        try:
            first = sign_flip_mean_p_value([0.1] * 12, seed=7)
            second = sign_flip_mean_p_value([0.1] * 12, seed=7)
        finally:
            analysis.SIGN_FLIP_SAMPLES = original
        self.assertEqual(first, second)
        self.assertLess(first, 0.01)

    def test_primary_statistics_aggregate_three_blocks_then_six_holm_tests(self):
        metrics = []
        values = {
            "random": 0.4, "rule": 0.5, "rf": 0.6,
            "transformer": 0.7, "mlp": 0.65,
        }
        for configuration_index in range(12):
            configuration = f"S{configuration_index}"
            for instance_index in range(10):
                instance_id = f"test-{configuration}-{instance_index + 1:03d}"
                for block in BLOCKS:
                    for method in METHODS:
                        metrics.append({
                            "configuration": configuration,
                            "instance_id": instance_id,
                            "block": block,
                            "method": method,
                            "oos_hv": values[method] + block * 0.001,
                        })
        audit = {
            "rf_models": {
                str(block): {"sha256": f"rf-{block}"} for block in BLOCKS
            },
            "blocks": {
                str(block): {"checkpoints": {
                    "transformer": {"sha256": f"transformer-{block}"},
                    "mlp": {"sha256": f"mlp-{block}"},
                }} for block in BLOCKS
            },
        }
        old_bootstrap = analysis.BOOTSTRAP_SAMPLES
        old_sign_flip = analysis.SIGN_FLIP_SAMPLES
        analysis.BOOTSTRAP_SAMPLES = 500
        analysis.SIGN_FLIP_SAMPLES = 2_000
        try:
            result = _paired_statistics(metrics, audit)
        finally:
            analysis.BOOTSTRAP_SAMPLES = old_bootstrap
            analysis.SIGN_FLIP_SAMPLES = old_sign_flip
        self.assertEqual(len(result["comparisons"]), 6)
        self.assertTrue(all(row["wins"] == row["instance_count"]
                            for row in result["comparisons"]))
        self.assertTrue(all(
            row["holm_adjusted_p_value_six_test_family"] <= 0.05
            for row in result["comparisons"]))
        self.assertEqual(len(result["model_seed_performance"]), 2 * 5 * 3)


if __name__ == "__main__":
    unittest.main()
