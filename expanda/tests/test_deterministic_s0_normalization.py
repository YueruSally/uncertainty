from pathlib import Path
import unittest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prepare_deterministic_s0_normalization import _freeze, _scaled_front


class _Batch:
    def __init__(self, quantity, et, lt):
        self.quantity = quantity
        self.ET = et
        self.LT = lt


class DeterministicNormalizationTests(unittest.TestCase):
    def test_scaled_front_uses_fixed_objective_order_and_instance_divisors(self):
        front = [{"optimisation_objectives": {
            "makespan": 80.0, "emission": 600.0, "cost": 400.0,
        }}]
        batches = [_Batch(10.0, 0.0, 40.0), _Batch(10.0, 10.0, 80.0)]
        self.assertEqual(_scaled_front(front, batches), [(20.0, 30.0, 1.0)])

    def test_freeze_adds_padding_and_positive_ranges(self):
        minimum, maximum = _freeze(
            [(1.0, 10.0, 100.0), (3.0, 30.0, 300.0)], 0.0, 100.0, 0.1)
        self.assertEqual(minimum, [0.8, 8.0, 80.0])
        self.assertEqual(maximum, [3.2, 32.0, 320.0])

    def test_freeze_rejects_nonfinite_points(self):
        with self.assertRaises(ValueError):
            _freeze([(1.0, 2.0, float("nan"))], 1.0, 99.0, 0.1)


if __name__ == "__main__":
    unittest.main()
