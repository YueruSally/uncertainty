import json
from pathlib import Path
import sys
import tempfile
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analyze_mutation_candidate_audit import (
    analyze, shuffled_residual_expectations,
)


def write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8")


class MutationCandidateAuditTests(unittest.TestCase):
    def test_large_action_shuffle_is_bounded_and_reproducible(self):
        prior = [1. / 16] * 16
        residual = [float(index + 1) for index in range(16)]
        values = [float(index) for index in range(16)]
        first = shuffled_residual_expectations(
            prior, residual, values, sample_count=32)
        second = shuffled_residual_expectations(
            prior, residual, values, sample_count=32)
        self.assertEqual(len(first), 32)
        self.assertEqual(first, second)

    def test_analysis_compares_trained_prior_shuffle_and_oracle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_jsonl(root / "mutation_candidate_audit_states.jsonl", [{
                "state_id": 1,
                "generation": 3,
                "evaluation": 40,
                "evaluation_fraction": .5,
                "operator": "mod",
                "archive_hv": .2,
                "audited_candidate_ids": [0, 1],
                "audited_prior_probability_mass": 1.,
                "audited_trained_probability_mass": 1.,
            }])
            rows = []
            for replicate in (0, 1):
                for candidate, delta, quality, trained in (
                        (0, 0., .2, .2),
                        (1, .1, .8, .8)):
                    rows.append({
                        "state_id": 1,
                        "candidate_id": candidate,
                        "replicate": replicate,
                        "baseline_probability": .5,
                        "trained_probability": trained,
                        "feasible": True,
                        "normalized_violation": 0.,
                        "archive_delta_hv": delta,
                        "single_point_hv": quality,
                        "fingerprint": f"{candidate}-{replicate}",
                        "archive_diagnostics": {"would_change_archive": bool(delta)},
                    })
            write_jsonl(root / "mutation_candidate_audit_outcomes.jsonl", rows)
            result = analyze(root)
            self.assertEqual(result["state_count"], 1)
            self.assertEqual(result["counterfactual_evaluation_count"], 4)
            self.assertEqual(result["states_with_positive_archive_gap"], 1)
            self.assertEqual(result["states_trained_beats_prior_archive_delta"], 1)
            comparison = result["states"][0]["archive_delta_hv"]
            self.assertGreater(
                comparison["trained_expected"], comparison["prior_expected"])
            self.assertGreater(comparison["oracle"], comparison["trained_expected"])


if __name__ == "__main__":
    unittest.main()
