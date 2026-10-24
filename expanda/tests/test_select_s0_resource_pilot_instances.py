import unittest

from resource_pilot_selection import select_records


class ResourcePilotSelectionTest(unittest.TestCase):
    def test_selects_low_lower_median_and_high_deterministically(self):
        records = [
            {
                "instance_id": instance_id,
                "candidate_path_token_count": tokens,
                "candidate_path_count": paths,
            }
            for instance_id, tokens, paths in (
                ("d", 40, 4),
                ("b", 20, 2),
                ("e", 50, 5),
                ("a", 10, 1),
                ("c", 30, 3),
                ("f", 60, 6),
            )
        ]
        ranking, selected = select_records(records)
        self.assertEqual(
            [row["instance_id"] for row in ranking],
            ["a", "b", "c", "d", "e", "f"],
        )
        self.assertEqual(
            [(row["role"], row["rank"], row["instance_id"])
             for row in selected],
            [("low", 1, "a"), ("median", 3, "c"), ("high", 6, "f")],
        )

    def test_rejects_too_few_records(self):
        with self.assertRaisesRegex(ValueError, "at least three"):
            select_records([])


if __name__ == "__main__":
    unittest.main()
