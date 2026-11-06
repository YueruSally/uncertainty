import unittest

import torch

from collect_m_prescreen_scale_v1 import trajectory_phase
from generate_m_prescreen_scale_instances import (ALGORITHM_SEED_BASE,
    CCP_SEED_BASE, FINAL_TEST_MASTER_SEED, MASTER_SEED, PATH_SEED_OFFSET,
    TRAIN_COUNT, VALIDATION_COUNT)
from train_m_prescreen_scale_curve_v1 import (SIZES, instance_order,
    pairwise_ranking_loss)


class PrescreenScaleTests(unittest.TestCase):
    def test_available_states_are_split_into_chronological_tertiles(self):
        for count in (12, 19, 30):
            phases = [trajectory_phase(index, count)
                      for index in range(count)]
            self.assertGreaterEqual(phases.count("early"), 4)
            self.assertGreaterEqual(phases.count("middle"), 4)
            self.assertGreaterEqual(phases.count("late"), 4)
            self.assertEqual(phases, sorted(phases,
                key=("early", "middle", "late").index))

    def test_generated_namespaces_do_not_touch_locked_final_test(self):
        total = TRAIN_COUNT+VALIDATION_COUNT
        instance = {MASTER_SEED+index*100 for index in range(total)}
        path = {MASTER_SEED+index*100+PATH_SEED_OFFSET+offset
                for index in range(total) for offset in range(20)}
        ccp = {CCP_SEED_BASE+index for index in range(total)}
        algorithm = {ALGORITHM_SEED_BASE+index for index in range(total)}
        self.assertEqual((TRAIN_COUNT, VALIDATION_COUNT), (200, 20))
        self.assertTrue(all(value < FINAL_TEST_MASTER_SEED
                            for value in instance|path|ccp|algorithm))
        self.assertFalse(instance & path)
        self.assertFalse(instance & ccp)
        self.assertFalse(path & ccp)
        self.assertFalse(algorithm & (instance|path|ccp))

    def test_learning_curve_is_nested_and_deterministic(self):
        names = {f"instance-{index:03d}" for index in range(200)}
        first = instance_order(names)
        second = instance_order(reversed(first))
        self.assertEqual(first, second)
        self.assertEqual(SIZES, (25, 50, 100, 200))
        for smaller, larger in zip(SIZES, SIZES[1:]):
            self.assertTrue(set(first[:smaller]) < set(first[:larger]))

    def test_pairwise_loss_rewards_correct_within_state_order(self):
        target = torch.tensor([[0.0, 2.0, -1.0]])
        correct = torch.tensor([[0.0, 2.0, -1.0]], requires_grad=True)
        reversed_order = torch.tensor([[0.0, -2.0, 1.0]],
                                      requires_grad=True)
        good = pairwise_ranking_loss(correct, target)
        bad = pairwise_ranking_loss(reversed_order, target)
        self.assertTrue(torch.isfinite(good))
        self.assertLess(good.item(), bad.item())
        good.backward()
        self.assertIsNotNone(correct.grad)


if __name__ == "__main__":
    unittest.main()
