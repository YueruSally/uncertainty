import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest

from evaluate_m_late_replace_frequency import _strategy_run


class MFrequencyReuseTests(unittest.TestCase):
    def test_partial_reuse_falls_back_to_fresh_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = SimpleNamespace(
                reuse_root=root / "reuse",
                positive_probe_root=None,
                reversible_probe_root=root / "reversible",
                out=root / "fresh",
            )
            reused_base = args.reuse_root / "S0_005/seed_1/BASE"
            reused_base.mkdir(parents=True)
            (reused_base / "COMPLETE.json").write_text("{}")
            reused_reversible = (
                args.reversible_probe_root
                / "S0_005/seed_1/REPLACE_LATE_REVERSIBLE_PROBE")
            reused_reversible.mkdir(parents=True)
            (reused_reversible / "COMPLETE.json").write_text("{}")

            self.assertEqual(
                _strategy_run(args, "S0_005", 1, "BASE"), reused_base)
            self.assertEqual(
                _strategy_run(
                    args, "S0_005", 1,
                    "REPLACE_LATE_REVERSIBLE_PROBE"),
                reused_reversible)
            self.assertEqual(
                _strategy_run(args, "S0_006", 1, "BASE"),
                args.out / "S0_006/seed_1/BASE")
            self.assertEqual(
                _strategy_run(
                    args, "S0_006", 1,
                    "REPLACE_LATE_REVERSIBLE_PROBE"),
                args.out
                / "S0_006/seed_1/REPLACE_LATE_REVERSIBLE_PROBE")


if __name__ == "__main__":
    unittest.main()
