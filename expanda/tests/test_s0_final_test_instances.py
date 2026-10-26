import unittest

from generate_s0_final_test_instances import (
    CCP100_SEED_BASE,
    DEFAULT_MASTER_SEED,
    OOS5000_SEED_BASE,
    PATH_SEED_OFFSET,
)


class S0FinalTestInstanceTest(unittest.TestCase):
    def test_final_namespace_is_disjoint_from_existing_catalog(self):
        self.assertGreaterEqual(DEFAULT_MASTER_SEED, 60_000_000)
        seeds = [DEFAULT_MASTER_SEED + index * 100 for index in range(10)]
        self.assertEqual(len(seeds), len(set(seeds)))
        self.assertTrue(all(seed >= 60_000_000 for seed in seeds))

    def test_final_namespace_is_disjoint_from_ccp_and_oos_seeds(self):
        instance_seeds = {
            DEFAULT_MASTER_SEED + index * 100 for index in range(10)
        }
        possible_path_seeds = {
            DEFAULT_MASTER_SEED + index * 100 + PATH_SEED_OFFSET + offset
            for index in range(10)
            for offset in range(20)
        }
        ccp_seeds = {CCP100_SEED_BASE + index for index in range(10)}
        oos_seeds = {OOS5000_SEED_BASE + index for index in range(10)}
        self.assertFalse(instance_seeds & ccp_seeds)
        self.assertFalse(instance_seeds & oos_seeds)
        self.assertFalse(possible_path_seeds & ccp_seeds)
        self.assertFalse(possible_path_seeds & oos_seeds)
        self.assertFalse(ccp_seeds & oos_seeds)


if __name__ == "__main__":
    unittest.main()
