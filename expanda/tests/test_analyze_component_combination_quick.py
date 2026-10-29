import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from analyze_component_combination_quick import main


class QuickCombinationAnalysisTests(unittest.TestCase):
    def test_reports_pairwise_percent_not_ratio_of_unpaired_means(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            values = {
                ("S0_002", 11): {
                    "BASE": 0.5, "I": 0.55, "C": 0.5, "M": 0.45,
                    "IC": 0.6, "ICM": 0.65,
                },
                ("S0_002", 12): {
                    "BASE": 1.0, "I": 1.0, "C": 0.9, "M": 1.0,
                    "IC": 1.1, "ICM": 1.2,
                },
            }
            for (instance, seed), methods in values.items():
                for method, hv in methods.items():
                    path = root / instance / f"seed_{seed}" / method
                    path.mkdir(parents=True)
                    (path / "COMPLETE.json").write_text(json.dumps({
                        "completed": True,
                        "component": method,
                        "seeds": {"algorithm": seed},
                        "final_hypervolume": hv,
                    }))
            with patch("builtins.print") as output:
                main([
                    "--root", str(root), "--instances", "S0_002",
                    "--seeds", "11", "12",
                ])
            report = json.loads(output.call_args.args[0])
            self.assertAlmostEqual(
                report["aggregate"]["I"]["mean_paired_percent"], 5.0)
            self.assertEqual(report["aggregate"]["IC"]["wins"], 2)


if __name__ == "__main__":
    unittest.main()
