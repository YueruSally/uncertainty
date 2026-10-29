import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@unittest.skipUnless(
    importlib.util.find_spec("sklearn"), "scikit-learn not installed")
class MutationCandidateProbeTests(unittest.TestCase):
    def test_probe_learns_held_out_candidate_order(self):
        from probe_mutation_candidate_audit import probe

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = []
            for run_index in range(3):
                run = root / f"run-{run_index}"
                run.mkdir()
                rows = []
                for state_id in range(1, 4):
                    for candidate_id in range(4):
                        quality = .1 + .2 * candidate_id
                        for replicate in range(2):
                            rows.append({
                                "state_id": state_id,
                                "candidate_id": candidate_id,
                                "operator": "mod",
                                "candidate_features": [
                                    float(candidate_id), float(state_id)],
                                "global_features": [float(run_index)],
                                "parent_tokens": [[1., 2.], [2., 4.]],
                                "candidate_graph_tokens": [
                                    [0, 0, 0, 0.], [1, 0, 0, 1.]],
                                "feasible": True,
                                "single_point_hv": quality,
                                "normalized_violation": 0.,
                                "archive_delta_hv": quality / 10.,
                                "baseline_probability": .25,
                                "trained_probability": .25,
                                "fingerprint": (
                                    f"{run_index}-{state_id}-{candidate_id}-"
                                    f"{replicate}"),
                            })
                (run / "mutation_candidate_audit_outcomes.jsonl").write_text(
                    "".join(json.dumps(row) + "\n" for row in rows),
                    encoding="utf-8")
                runs.append(run)
            result = probe(runs[:2], runs[2:], seed=7)
            validation = result["models"]["ridge"]["validation"]
            self.assertEqual(validation["informative_state_count"], 3)
            self.assertGreater(validation["mean_pairwise_accuracy"], .99)
            self.assertGreater(validation["model_minus_prior_expected"], 0.)
            self.assertEqual(validation["oracle_hit_fraction"], 1.)


if __name__ == "__main__":
    unittest.main()
