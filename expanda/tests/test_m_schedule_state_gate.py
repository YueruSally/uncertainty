import unittest

from m_schedule_state_features import (
    assert_progress_replay,
    pre_intervention_progress,
    progress_features,
)


def _row(generation, archive_hv, population_hv=None):
    return {
        "checkpoint": "environmental_selection",
        "generation": generation,
        "real_objective_evaluation_count": 20 + generation * 20,
        "population_size": 20,
        "feasible_count": 20,
        "front_size": 5 + generation,
        "population_hv": (
            archive_hv - 0.01 if population_hv is None else population_hv),
        "archive_hv": archive_hv,
    }


class MScheduleStateGateTests(unittest.TestCase):
    def test_pre_intervention_excludes_generation_ten(self):
        rows = [_row(9, 0.3), _row(10, 0.4), _row(8, 0.2)]
        selected = pre_intervention_progress(rows, 10, 400)
        self.assertEqual([row["generation"] for row in selected], [8, 9])

    def test_replay_ignores_runtime_but_requires_hv(self):
        base = pre_intervention_progress([_row(8, 0.2)], 10, 400)
        same = pre_intervention_progress([{
            **_row(8, 0.2), "runtime_seconds": 10_000,
        }], 10, 400)
        assert_progress_replay(base, same)
        different = pre_intervention_progress(
            [_row(8, 0.21, population_hv=0.19)], 10, 400)
        with self.assertRaisesRegex(ValueError, "archive_hv"):
            assert_progress_replay(base, different)

    def test_features_are_fixed_and_finite(self):
        rows = pre_intervention_progress(
            [_row(8, 0.2), _row(9, 0.3)], 10, 400)
        names, features = progress_features(rows)
        self.assertEqual(len(names), 55)
        self.assertEqual(len(features), 55)
        self.assertEqual(len(set(names)), 55)


if __name__ == "__main__":
    unittest.main()
