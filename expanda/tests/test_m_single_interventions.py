import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from audit_m_single_interventions import (
    _action_signature, _spread_events, _spread_indices,
)


class MutationSingleInterventionAuditTests(unittest.TestCase):
    def test_spread_events_selects_replace_calls_and_caps_requested_count(self):
        rows = [
            {
                "decision_kind": "base_mixture_choice",
                "operator": operator,
                "mixture_event_index": index,
                "generation": index // 2,
            }
            for index, operator in enumerate(
                ("add", "replace", "replace", "del", "replace"), 1)
        ]
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / "mutation_outcomes.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8")
            selected = _spread_events(run, requested=10)
        self.assertEqual(
            [row["mixture_event_index"] for row in selected], [2, 3, 5])

    def test_spread_indices_is_unique_and_covers_candidate_range(self):
        selected = _spread_indices([1, 3, 4, 7, 8, 11], requested=4)
        self.assertEqual(selected[0], 1)
        self.assertEqual(selected[-1], 11)
        self.assertEqual(len(selected), len(set(selected)))

    def test_action_signature_distinguishes_replacement_paths(self):
        first = _action_signature({
            "operator": "replace", "batch_id": 2,
            "replacement_path_id": "101",
        })
        second = _action_signature({
            "operator": "replace", "batch_id": 2,
            "replacement_path_id": "202",
        })
        self.assertNotEqual(first, second)


if __name__ == "__main__":
    unittest.main()
