import unittest

from build_segment_learning_dataset import build_row, nested_run_number


class SegmentLearningDatasetTests(unittest.TestCase):
    def setUp(self):
        self.candidate = {
            "candidate_id": 3, "batch_id": 1, "allocation_index": 0,
            "arc_index": 0, "path_id": 0, "selection_probability": .25,
            "baseline_selection_probability": .25, "batch_quantity": 2.,
            "path_count": 1, "share": 1., "path_cost_per_teu": 10.,
            "path_emission_per_teu": 20., "path_nominal_time_h": 3.,
            "alternative_mode_count": 2, "from_node": "A", "to_node": "B",
            "current_mode": "road", "eligible": True, "chosen": True,
        }
        self.event = {
            "event_id": "random:1", "operator": "mode", "generation": 4,
            "phase": "offspring", "selection_policy": "random",
            "candidate_count": 4, "feasible_before": True, "feasible_after": True,
            "violation_before": 2., "violation_after": 1.,
            "violation_breakdown_before": {"miss_tt": 0., "cap_excess": 0.},
            "violation_breakdown_after": {"miss_tt": 1., "cap_excess": 2.},
            "q90_cost_before": 10., "q90_emission_before": 20.,
            "q90_makespan_before": 30., "objective_label_eligible": True,
            "effective_mutation": True,
            "outcome_attributable_to_selected_target": True,
            "delta_cost": 1., "delta_emission": -1., "delta_makespan": .5,
            "child_dominates_before": False, "tradeoff_move": True,
            "survived_environmental_selection": True,
            "nondominated_after_selection": True,
            "scenario_improvement_rate_cost": .75,
            "scenario_improvement_rate_emission": .25,
            "scenario_improvement_rate_makespan": .60,
        }

    def test_nested_formal_run_number(self):
        self.assertEqual(nested_run_number(
            __import__("pathlib").Path("formal/run07/random")), 7)

    def test_multi_objective_and_failure_labels(self):
        row = build_row(self.event, self.candidate, 7, "train", "run07/random", .6)
        self.assertTrue(row["cost_improved"])
        self.assertFalse(row["emission_improved"])
        self.assertTrue(row["makespan_improved"])
        self.assertTrue(row["violation_reduced"])
        self.assertTrue(row["schedule_failure_introduced"])
        self.assertTrue(row["capacity_worsened"])
        self.assertTrue(row["pareto_promising"])
        self.assertTrue(row["selection_survivor"])
        self.assertTrue(row["nondominated_after_selection"])
        self.assertTrue(row["scenario_stable_cost"])
        self.assertFalse(row["scenario_stable_emission"])
        self.assertTrue(row["scenario_stable_makespan"])

    def test_missing_new_log_field_is_not_eligible(self):
        self.event.pop("scenario_improvement_rate_cost")
        self.event.pop("nondominated_after_selection")
        row = build_row(self.event, self.candidate, 7, "train", "run07/random", .6)
        self.assertFalse(row["scenario_stable_cost_eligible"])
        self.assertFalse(row["nondominated_after_selection_eligible"])


if __name__ == "__main__":
    unittest.main()
