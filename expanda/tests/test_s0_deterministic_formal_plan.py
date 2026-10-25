import unittest

from s0_deterministic_formal_plan import (
    evenly_spaced_ranks,
    ordered_ids_by_seed,
)


class S0DeterministicFormalPlanTest(unittest.TestCase):
    def test_selects_ten_evenly_spaced_ranks_with_endpoints(self):
        self.assertEqual(
            evenly_spaced_ranks(50, 10),
            [1, 6, 12, 17, 23, 28, 34, 39, 45, 50],
        )

    def test_seed_orders_share_exactly_the_same_instances(self):
        ids = [f"train-S0-{index:03d}" for index in range(1, 11)]
        orders = ordered_ids_by_seed(ids)
        self.assertEqual(set(orders), {"301", "302", "303"})
        for order in orders.values():
            self.assertEqual(set(order), set(ids))
        self.assertNotEqual(orders["301"], orders["302"])

    def test_rejects_impossible_rank_request(self):
        with self.assertRaisesRegex(ValueError, "2 <= count <= total"):
            evenly_spaced_ranks(5, 6)


if __name__ == "__main__":
    unittest.main()
