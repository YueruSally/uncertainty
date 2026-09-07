import numpy as np

from run_run1_oos_validation import (
    all_solution_statistics,
    nondominated_mask,
    pareto_appearance_rows,
    solution_objective_summary_rows,
)


def test_all_solution_statistics_retains_every_input_and_has_required_stats():
    solutions = [
        {"solution_id": "a", "method": "EV", "cost_s": [1, 2, 10],
         "emission_s": [2, 3, 11], "makespan_s": [3, 4, 12]},
        {"solution_id": "b", "method": "CCP30", "cost_s": [9, 9, 9],
         "emission_s": [9, 9, 9], "makespan_s": [9, 9, 9]},
    ]
    rows = all_solution_statistics(solutions)
    assert [row["solution_id"] for row in rows] == ["a", "b"]
    assert rows[0]["oos_mean_cost"] == 13 / 3
    assert rows[0]["oos_median_emission"] == 3
    assert rows[0]["oos_q90_makespan"] == 12


def test_nondominance_is_three_dimensional_and_retains_duplicates():
    points = [[1, 4, 4], [4, 1, 4], [4, 4, 1], [3, 3, 3],
              [4, 4, 4], [3, 3, 3]]
    assert nondominated_mask(points).tolist() == [True, True, True, True,
                                                  False, True]


def test_hand_controlled_pooled_scenario_fronts_and_appearance_counts():
    solutions = [
        {"solution_id": "ev-a", "method": "EV",
         "cost_s": [1, 1, 2], "emission_s": [2, 4, 2],
         "makespan_s": [3, 4, 2]},
        {"solution_id": "ev-b", "method": "EV",
         "cost_s": [2, 4, 2], "emission_s": [1, 1, 2],
         "makespan_s": [3, 4, 2]},
        {"solution_id": "ccp-a", "method": "CCP30",
         "cost_s": [1, 3, 1], "emission_s": [1, 3, 3],
         "makespan_s": [3, 3, 3]},
        {"solution_id": "ccp-never", "method": "CCP30",
         "cost_s": [9, 9, 9], "emission_s": [9, 9, 9],
         "makespan_s": [9, 9, 9]},
    ]
    expected_fronts = [
        {"ccp-a"},
        {"ev-a", "ev-b", "ccp-a"},
        {"ev-a", "ev-b", "ccp-a"},
    ]
    actual_fronts = []
    for scenario_index in range(3):
        pooled_points = np.asarray([
            [solution[f"{objective}_s"][scenario_index]
             for objective in ("cost", "emission", "makespan")]
            for solution in solutions
        ])
        mask = nondominated_mask(pooled_points)
        actual_fronts.append({solution["solution_id"]
                              for solution, keep in zip(solutions, mask)
                              if keep})
    assert actual_fronts == expected_fronts

    rows = pareto_appearance_rows(solutions, 3)
    by_id = {row["solution_id"]: row for row in rows}
    expected_counts = {"ev-a": 2, "ev-b": 2, "ccp-a": 3,
                       "ccp-never": 0}
    assert {solution_id: row["pareto_appearances"]
            for solution_id, row in by_id.items()} == expected_counts
    for solution_id, expected_count in expected_counts.items():
        assert by_id[solution_id]["pareto_frequency"] == expected_count / 3
    assert by_id["ccp-never"]["pareto_appearances"] == 0
    assert by_id["ccp-never"]["pareto_frequency"] == 0
    assert [row["pareto_frequency"] for row in rows] == sorted(
        [row["pareto_frequency"] for row in rows], reverse=True)


def test_scenario_array_length_must_match_requested_count():
    solution = {"solution_id": "ev-1", "method": "EV",
                "cost_s": [1], "emission_s": [1], "makespan_s": [1]}
    try:
        pareto_appearance_rows([solution], 2)
    except RuntimeError as error:
        assert "scenario count" in str(error)
    else:
        raise AssertionError("expected mismatched scenario arrays to fail")


def test_oos_mean_summary_joins_frequencies_and_marks_pooled_front():
    solutions = [
        {"solution_id": "ev-a", "method": "EV",
         "cost_s": [1, 3], "emission_s": [4, 2],
         "makespan_s": [2, 2]},
        {"solution_id": "ccp-a", "method": "CCP30",
         "cost_s": [3, 3], "emission_s": [1, 1],
         "makespan_s": [3, 3]},
        {"solution_id": "ccp-dominated", "method": "CCP30",
         "cost_s": [5, 5], "emission_s": [5, 5],
         "makespan_s": [5, 5]},
    ]
    frequencies = pareto_appearance_rows(solutions, 2)
    rows = solution_objective_summary_rows(solutions, frequencies)
    by_id = {row["solution_id"]: row for row in rows}

    assert by_id["ev-a"]["oos_mean_cost"] == 2.0
    assert by_id["ev-a"]["oos_mean_emission"] == 3.0
    assert by_id["ev-a"]["oos_mean_makespan"] == 2.0
    assert by_id["ev-a"]["pareto_appearances"] == 2
    assert by_id["ev-a"]["pareto_frequency"] == 1.0
    assert by_id["ev-a"]["oos_mean_pareto"] is True
    assert by_id["ccp-a"]["oos_mean_pareto"] is True
    assert by_id["ccp-dominated"]["oos_mean_pareto"] is False
