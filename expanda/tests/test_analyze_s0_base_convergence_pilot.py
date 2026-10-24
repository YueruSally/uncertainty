import unittest

from analyze_s0_base_convergence_pilot import (
    diagnostic_flags,
    extract_checkpoints,
    summarise_checkpoints,
)


def _rows():
    rows = []
    for role_index, role in enumerate(("low", "median", "high")):
        for repeat in range(1, 4):
            final = 1.0 + 0.1 * role_index + 0.01 * repeat
            for budget, fraction in ((5_000, 0.95), (50_000, 1.0)):
                rows.append({
                    "role": role,
                    "instance_id": f"train-S0-{role_index:03d}",
                    "repeat": str(repeat),
                    "algorithm_seed": str(100 * role_index + repeat),
                    "real_objective_evaluation_count": str(budget),
                    "archive_hv": str(final * fraction),
                    "runtime_seconds": str(budget / 100),
                })
    return rows


class BaseConvergenceAnalysisTest(unittest.TestCase):
    def test_extracts_exact_checkpoints_and_summarises_all_runs(self):
        extracted = extract_checkpoints(_rows(), (5_000, 50_000))
        self.assertEqual(len(extracted), 18)
        summary = summarise_checkpoints(extracted, (5_000, 50_000))
        overall = next(
            row for row in summary
            if row["role"] == "overall"
            and row["evaluation_budget"] == 5_000)
        self.assertEqual(overall["run_count"], 9)
        self.assertAlmostEqual(overall["mean_fraction_of_run_final"], 0.95)
        flags = diagnostic_flags(summary, (5_000, 50_000))
        self.assertFalse(flags[0]["screening_threshold_met"])

    def test_rejects_missing_exact_checkpoint(self):
        rows = _rows()
        rows.pop(0)
        with self.assertRaisesRegex(ValueError, "lacks exact"):
            extract_checkpoints(rows, (5_000, 50_000))


if __name__ == "__main__":
    unittest.main()
