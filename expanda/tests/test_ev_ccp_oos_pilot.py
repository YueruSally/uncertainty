import unittest

import numpy as np

import baseline_uncertainty as base
import run_ev_ccp_oos_pilot as pilot


class PilotScenarioTests(unittest.TestCase):
    def test_training_prefix_is_exact_and_draws_nothing(self):
        master = base.ScenarioSet(
            50, 7, {("A", "B", "road"): np.arange(50.0)},
            {("A", "B", "road"): np.arange(100.0, 150.0)},
            {}, {}, True)
        prefix = pilot.scenario_prefix(master, 30)
        self.assertEqual(prefix.size, 30)
        self.assertTrue(np.array_equal(
            prefix.travel_multiplier[("A", "B", "road")],
            master.travel_multiplier[("A", "B", "road")][:30]))
        self.assertTrue(np.array_equal(
            prefix.border_delay_h[("A", "B", "road")],
            master.border_delay_h[("A", "B", "road")][:30]))

    def test_scenario_digest_changes_with_stream(self):
        a = base.ScenarioSet(1, 1, {("A", "B", "road"): np.array([1.])},
                             {}, {}, {}, True)
        b = base.ScenarioSet(1, 2, {("A", "B", "road"): np.array([1.])},
                             {}, {}, {}, True)
        self.assertNotEqual(pilot.scenario_digest(a), pilot.scenario_digest(b))


if __name__ == "__main__":
    unittest.main()
