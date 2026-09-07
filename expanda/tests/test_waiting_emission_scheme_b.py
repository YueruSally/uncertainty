import numpy as np

import baseline_uncertainty as model
import run_formal_ev_vs_ccp_s30_30runs as formal
import run_run1_oos_validation as run1_validation


WAIT = model.WAIT_EMISSION_gCO2_per_TEU_H_DEFAULT


def _scheduled_wait_fixture(carbon_tax=50.0, theta=1):
    road = model.Arc(
        "A", "J", "road", 60.0, 1000.0, 0.0, 10.0, 60.0,
        from_region="R0", to_region="R1")
    rail = model.Arc(
        "J", "B", "rail", 100.0, 1000.0, 0.0, 20.0, 100.0,
        from_region="R1", to_region="R2")
    path = model.Path(
        1, "A", "B", ["A", "J", "B"], ["road", "rail"],
        [road, rail], 0.0, 2600.0, 2.0)
    batch = model.Batch(1, "A", "B", 10.0, 0.0, 100.0,
                        penalty_per_teu_h=0.0)
    timetable = model.TimetableEntry(
        "J", "B", "rail", 1.0, 3.0, 24.0, travel_time_h=1.0)
    scenarios = model.ScenarioSet(
        size=2, seed=17,
        travel_multiplier={
            ("A", "J", "road"): np.array([1.0, 2.0]),
            ("J", "B", "rail"): np.ones(2)},
        border_delay_h={}, arc_border_event={}, border_event_mean_h={},
        stochastic=True)
    individual = model.Individual({
        ("A", "B", 1): [model.PathAllocation(path, 1.0)]})
    kwargs = dict(
        node_hold_cost={"J": 0.0}, node_proc_cost={},
        carbon_tax_map={"R0": 0.0, "R1": carbon_tax}, trans_map={},
        border_delay_map={}, theta_rm={("R1", "rail"): theta},
        node_trans_cost={})
    return road, rail, path, batch, timetable, scenarios, individual, kwargs


def _evaluate_fixture(carbon_tax=50.0, theta=1):
    fixture = _scheduled_wait_fixture(carbon_tax, theta)
    road, rail, path, batch, timetable, scenarios, individual, kwargs = fixture
    previous = model.ACTIVE_SCENARIO_SET
    model.ACTIVE_SCENARIO_SET = scenarios
    model._PATH_SCENARIO_CACHE = {}
    try:
        result = model.simulate_path_over_scenarios(
            path, batch, {("J", "B", "rail"): [timetable]}, {}, {},
            scenarios)
        model.evaluate_individual(
            individual, [batch], [road, rail],
            {("J", "B", "rail"): [timetable]}, 0.0, WAIT, **kwargs)
    finally:
        model.ACTIVE_SCENARIO_SET = previous
        model._PATH_SCENARIO_CACHE = {}
    weighted_wait = 10.0 * result.schedule_wait_h["J"]
    return individual, weighted_wait


def test_a_waiting_emission_delta_matches_flow_weighted_schedule_wait():
    individual, weighted_wait = _evaluate_fixture()
    assert weighted_wait.tolist() == [20.0, 10.0]
    np.testing.assert_allclose(
        np.diff(individual.emission_s), WAIT * np.diff(weighted_wait),
        rtol=1e-12, atol=1e-9)


def test_b_waiting_carbon_uses_waiting_region_and_outgoing_mode():
    individual, weighted_wait = _evaluate_fixture(carbon_tax=50.0, theta=1)
    expected = WAIT * weighted_wait / 1e6 * 50.0
    np.testing.assert_allclose(
        individual.waiting_carbon_cost_s, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        np.diff(individual.waiting_carbon_cost_s),
        np.diff(WAIT * weighted_wait) / 1e6 * 50.0,
        rtol=1e-12, atol=1e-12)


def test_c_zero_carbon_price_keeps_emission_but_zeroes_waiting_carbon():
    individual, _ = _evaluate_fixture(carbon_tax=0.0, theta=1)
    assert np.ptp(individual.emission_s) > 0.0
    assert np.array_equal(individual.waiting_carbon_cost_s, np.zeros(2))


def test_d_theta_zero_keeps_emission_but_zeroes_waiting_carbon():
    individual, _ = _evaluate_fixture(carbon_tax=50.0, theta=0)
    assert np.ptp(individual.emission_s) > 0.0
    assert np.array_equal(individual.waiting_carbon_cost_s, np.zeros(2))


def test_e_road_has_zero_schedule_wait_emission_and_waiting_carbon():
    road = model.Arc(
        "A", "B", "road", 60.0, 1000.0, 0.0, 10.0, 60.0,
        from_region="R", to_region="R")
    path = model.Path(1, "A", "B", ["A", "B"], ["road"], [road],
                      0.0, 600.0, 1.0)
    batch = model.Batch(1, "A", "B", 10.0, 0.0, 100.0,
                        penalty_per_teu_h=0.0)
    scenarios = model.ScenarioSet(
        2, 9, {("A", "B", "road"): np.array([1.0, 2.0])}, {}, {}, {},
        True)
    individual = model.Individual({
        ("A", "B", 1): [model.PathAllocation(path, 1.0)]})
    previous = model.ACTIVE_SCENARIO_SET
    model.ACTIVE_SCENARIO_SET = scenarios
    model._PATH_SCENARIO_CACHE = {}
    try:
        result = model.simulate_path_over_scenarios(
            path, batch, {}, {}, {}, scenarios)
        model.evaluate_individual(
            individual, [batch], [road], {}, 0.0, WAIT,
            carbon_tax_map={"R": 50.0}, theta_rm={("R", "road"): 1})
    finally:
        model.ACTIVE_SCENARIO_SET = previous
        model._PATH_SCENARIO_CACHE = {}
    assert result.schedule_wait_h == {}
    assert np.array_equal(individual.emission_s, np.array([6000.0, 6000.0]))
    assert np.array_equal(individual.waiting_carbon_cost_s, np.zeros(2))


def test_f_shared_constant_and_aggregation_semantics_are_unchanged():
    config = model.waiting_emission_configuration()
    assert WAIT == 320.50
    assert config["waiting_emission_gCO2_per_TEU_h"] == WAIT
    arrays = [np.array([3.0, 1.0, 2.0])] * 3
    assert model.aggregate_scenario_objectives(
        *arrays, 2.0, 2.0, 2.0, "ev", 0.9, 0.9, 0.9) == (2.0, 2.0, 2.0)
    assert model.aggregate_scenario_objectives(
        *arrays, 2.0, 2.0, 2.0, "ccp", 0.9, 0.9, 0.9) == (3.0, 3.0, 3.0)


def test_formal_drivers_persist_the_shared_waiting_configuration():
    required = model.waiting_emission_configuration()
    meta = formal.expected_meta(
        1, "EV", {"algorithm_seed": 910001, "ccp_training_seed": 920001})
    for key, value in required.items():
        assert meta[key] == value
        assert formal.WAITING_CONFIG[key] == value
        assert run1_validation.WAITING_CONFIG[key] == value
