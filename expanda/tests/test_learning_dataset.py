import unittest

from build_learning_dataset import build_row, parse_runs


class LearningDatasetTests(unittest.TestCase):
    def setUp(self):
        self.candidate = {
            "candidate_id": 0, "batch_id": 1, "allocation_index": None,
            "arc_index": None, "selection_probability": .025,
            "batch_quantity": 1., "path_count": 1, "share": 1.,
            "path_cost_per_teu": 1., "path_emission_per_teu": 1.,
            "path_nominal_time_h": 1., "alternative_mode_count": 0,
            "from_node": None, "to_node": None, "current_mode": None,
            "eligible": True, "chosen": True,
        }
        self.event = {
            "event_id": "run:1", "operator": "add", "generation": 2,
            "phase": "offspring", "feasible_before": True,
            "feasible_after": True, "violation_before": 0.,
            "q90_cost_before": 10., "q90_emission_before": 20.,
            "q90_makespan_before": 30., "finite_objective_label": True,
            "objective_label_eligible": True, "effective_mutation": True,
            "outcome_attributable_to_selected_target": True,
            "child_dominates_before": False, "tradeoff_move": True,
            "before_dominates_child": False,
            "survived_environmental_selection": True,
            "delta_cost": 1., "delta_emission": -1., "delta_makespan": 0.,
        }

    def test_parse_run_ranges(self):
        self.assertEqual(parse_runs("1-3,5"), {1, 2, 3, 5})

    def test_feasible_tradeoff_is_pareto_promising(self):
        row = build_row(self.event, self.candidate, 1, "train", "run1")
        self.assertTrue(row["model_row_eligible"])
        self.assertTrue(row["pareto_promising"])

    def test_infeasible_before_tradeoff_alone_is_not_positive(self):
        self.event["feasible_before"] = False
        row = build_row(self.event, self.candidate, 1, "train", "run1")
        self.assertFalse(row["pareto_promising"])

    def test_constraint_dominance_can_recover_infeasible_before(self):
        self.event["feasible_before"] = False
        self.event["child_dominates_before"] = True
        row = build_row(self.event, self.candidate, 1, "train", "run1")
        self.assertTrue(row["pareto_promising"])


if __name__ == "__main__":
    unittest.main()
