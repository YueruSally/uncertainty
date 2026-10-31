import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from audit_i_single_interventions import (
    _spread_events,
    _spread_indices_with_base,
)


class InitializationSingleInterventionTests(unittest.TestCase):
    def test_spread_events_uses_distinct_path_decisions(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "initialization_candidates.jsonl"
            rows = []
            for event in range(1, 7):
                for candidate in range(3):
                    rows.append({
                        "event_id": f"I:{event}",
                        "decision_stage": "path",
                        "decision_event_index": event,
                        "batch_id": event,
                        "selection_step": 0,
                        "required_path_count": 1,
                        "candidate_id": candidate,
                        "shadow_base_chosen": candidate == 1,
                        "chosen": candidate == 1,
                    })
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8")
            selected = _spread_events(Path(directory), 4)
            self.assertEqual(
                [row["event"] for row in selected], [1, 3, 4, 6])
            self.assertTrue(all(row["base_candidate_index"] == 1
                                for row in selected))

    def test_spread_indices_always_includes_base_candidate(self):
        selected = _spread_indices_with_base(range(20), 6, 7)
        self.assertEqual(len(selected), 6)
        self.assertIn(7, selected)
        self.assertEqual(len(set(selected)), len(selected))

    def test_spread_indices_returns_all_when_request_is_large(self):
        self.assertEqual(
            _spread_indices_with_base([1, 3, 8], 10, 3), [1, 3, 8])


if __name__ == "__main__":
    unittest.main()
