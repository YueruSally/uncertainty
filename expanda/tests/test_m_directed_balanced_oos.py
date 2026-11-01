import json
import sys
import tempfile
from pathlib import Path
import types
import unittest

# The workspace test runtime intentionally omits plotting libraries.  This
# evaluator never plots, so lightweight stubs keep the unit tests focused on
# run discovery and statistical aggregation.
matplotlib = types.ModuleType("matplotlib")
matplotlib.use = lambda *_args, **_kwargs: None
pyplot = types.ModuleType("matplotlib.pyplot")
mpl_toolkits = types.ModuleType("mpl_toolkits")
mplot3d = types.ModuleType("mpl_toolkits.mplot3d")
mplot3d.Axes3D = object
sys.modules.setdefault("matplotlib", matplotlib)
sys.modules.setdefault("matplotlib.pyplot", pyplot)
sys.modules.setdefault("mpl_toolkits", mpl_toolkits)
sys.modules.setdefault("mpl_toolkits.mplot3d", mplot3d)

from evaluate_m_directed_balanced_oos import (
    _aggregate, derive_common_oos_reference, discover_run_roots, resolve_run,
)


class DirectedBalancedOOSTests(unittest.TestCase):
    def test_discovers_nested_reuse_roots_and_resolves_one_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            primary = root / "primary"
            secondary = root / "secondary"
            nested = root / "nested"
            primary.mkdir()
            secondary.mkdir()
            nested.mkdir()
            (primary / "COMPLETE.json").write_text(json.dumps({
                "reuse_roots": [str(secondary)],
            }))
            (secondary / "COMPLETE.json").write_text(json.dumps({
                "reuse_roots": [str(nested)],
            }))
            run = nested / "S0_002/seed_1/BASE"
            run.mkdir(parents=True)
            for name in (
                    "COMPLETE.json", "configuration.json",
                    "final_feasible_nondominated.json"):
                (run / name).write_text("{}")
            roots = discover_run_roots(primary)
            self.assertEqual(roots, [primary, secondary, nested])
            self.assertEqual(resolve_run(roots, "S0_002", 1, "BASE"), run)

    def test_discovers_initialization_reuse_base_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            primary = root / "i_validation"
            base_root = root / "base_validation"
            primary.mkdir()
            base_root.mkdir()
            (primary / "COMPLETE.json").write_text(json.dumps({
                "reuse_base_root": str(base_root),
            }))
            run = base_root / "S0_002/seed_7/BASE"
            run.mkdir(parents=True)
            for name in (
                    "COMPLETE.json", "configuration.json",
                    "final_feasible_nondominated.json"):
                (run / name).write_text("{}")
            roots = discover_run_roots(primary)
            self.assertEqual(roots, [primary, base_root])
            self.assertEqual(resolve_run(roots, "S0_002", 7, "BASE"), run)

    def test_aggregate_reports_paired_percentages(self):
        rows = [
            {"instance": "S0_002", "algorithm_seed": 1,
             "paired_oos_percent": 10.0},
            {"instance": "S0_002", "algorithm_seed": 2,
             "paired_oos_percent": -2.0},
            {"instance": "S0_005", "algorithm_seed": 1,
             "paired_oos_percent": 4.0},
            {"instance": "S0_005", "algorithm_seed": 2,
             "paired_oos_percent": 0.0},
        ]
        result = _aggregate(rows, {1, 2})
        self.assertEqual(result["run_count"], 4)
        self.assertEqual(result["mean_paired_oos_percent"], 3.0)
        self.assertEqual(result["wins"], 2)
        self.assertEqual(result["ties"], 1)
        self.assertEqual(result["losses"], 1)
        self.assertEqual(len(result["per_instance"]), 2)

    def test_common_reference_dominates_pooled_points(self):
        rows = [{
            "methods": {
                "BASE": {
                    "oos_normalized_nondominated_points": [
                        [0.2, 1.2, 0.4], [0.4, 0.8, 0.6]]},
                "REPLACE_G10_JOINT_BALANCED": {
                    "oos_normalized_nondominated_points": [
                        [0.1, 0.9, 1.3]]},
            },
        }]
        result = derive_common_oos_reference(rows, (1.1, 1.1, 1.1))
        self.assertEqual(result["reference"][0], 1.1)
        self.assertGreater(result["reference"][1], 1.2)
        self.assertGreater(result["reference"][2], 1.3)
        self.assertEqual(result["pooled_point_count"], 3)


if __name__ == "__main__":
    unittest.main()
