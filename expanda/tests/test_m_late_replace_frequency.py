import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest

from evaluate_m_late_replace_frequency import _command, _strategy_run


class MFrequencyReuseTests(unittest.TestCase):
    def test_command_uses_schedule_specific_generation(self):
        args = SimpleNamespace(
            runner=Path("runner.py"),
            normalization=Path("normalization.json"),
            checkpoint=Path("checkpoint.pt"),
            scenario_count=200,
            ccp_confidence=0.9,
            evaluation_budget=400,
            pop=20,
            gens=1000,
            device="cpu",
        )
        command = _command(
            args, Path("instance.json"), Path("output"), 777001,
            0.5, False, False, 1, 0, 12)
        generation_flag = command.index(
            "--m-forced-operator-min-generation")
        probability_flag = command.index(
            "--m-forced-operator-probability")
        self.assertEqual(command[generation_flag + 1], "12")
        self.assertEqual(command[probability_flag + 1], "0.5")

    def test_partial_reuse_falls_back_to_fresh_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = SimpleNamespace(
                reuse_root=root / "reuse",
                reuse_roots=[root / "secondary"],
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
            reused_grid = (
                args.reuse_roots[0]
                / "S0_006/seed_1/REPLACE_LATE_P100")
            reused_grid.mkdir(parents=True)
            (reused_grid / "COMPLETE.json").write_text("{}")
            reused_g10 = (
                args.reuse_roots[0]
                / "S0_007/seed_1/REPLACE_G10_P100")
            reused_g10.mkdir(parents=True)
            (reused_g10 / "COMPLETE.json").write_text("{}")

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
                _strategy_run(args, "S0_006", 1, "REPLACE_G15_P100"),
                reused_grid)
            self.assertEqual(
                _strategy_run(args, "S0_006", 1, "REPLACE_G12_P100"),
                args.out / "S0_006/seed_1/REPLACE_G12_P100")
            self.assertEqual(
                _strategy_run(args, "S0_007", 1, "REPLACE_G10_P100"),
                reused_g10)
            self.assertEqual(
                _strategy_run(
                    args, "S0_006", 1,
                    "REPLACE_LATE_REVERSIBLE_PROBE"),
                args.out
                / "S0_006/seed_1/REPLACE_LATE_REVERSIBLE_PROBE")


if __name__ == "__main__":
    unittest.main()
