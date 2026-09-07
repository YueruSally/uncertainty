import argparse
import json
import os
import random
import tempfile
import unittest

import numpy as np

import baseline3
import baseline_uncertainty as model


class FrozenScenarioTests(unittest.TestCase):
    def test_documented_default_penalty_and_mode_speeds(self):
        self.assertEqual(model.DEFAULT_PENALTY_PER_TEU_H, 6.25)
        self.assertEqual(model.DEFAULT_LATE_PENALTY_USD_PER_TEU_DAY, 150.0)
        self.assertEqual(model.DEFAULT_MODE_SPEED_KMH, {
            "road": 40.0, "rail": 50.0, "water": 28.0})

    def test_empirical_ccp_quantile_uses_order_statistic(self):
        values = np.arange(1.0, 11.0)
        self.assertEqual(model.empirical_ccp_quantile(values, 0.80), 8.0)
        self.assertEqual(model.empirical_ccp_quantile(values, 0.90), 9.0)
        self.assertEqual(model.empirical_ccp_quantile(values, 1.00), 10.0)
        self.assertEqual(
            model.empirical_ccp_quantile(np.arange(1.0, 201.0), 0.90),
            180.0)
        self.assertEqual(
            model.empirical_ccp_quantile(np.arange(1.0, 51.0), 0.90),
            45.0)

    def test_objectives_are_quantiled_independently(self):
        result = model.aggregate_scenario_objectives(
            cost_s=np.array([100.0, 1.0, 50.0, 10.0]),
            emission_s=np.array([4.0, 300.0, 2.0, 1.0]),
            makespan_s=np.array([8.0, 6.0, 1000.0, 7.0]),
            scenario_cost_mean=0.0,
            scenario_emission_mean=0.0,
            scenario_time_mean=0.0,
            mode="ccp", confidence_cost=0.75,
            confidence_emission=0.75, confidence_time=0.75)
        self.assertEqual(result, (50.0, 4.0, 8.0))

    def test_capped_lognormal_is_reproducible_and_mean_corrected(self):
        first = model.sample_capped_mean_one_lognormal(
            size=200000, cv=0.60, cap_factor=3.5,
            rng=np.random.default_rng(2026))
        second = model.sample_capped_mean_one_lognormal(
            size=200000, cv=0.60, cap_factor=3.5,
            rng=np.random.default_rng(2026))
        self.assertTrue(np.array_equal(first, second))
        self.assertAlmostEqual(float(np.mean(first)), 1.0, delta=0.005)
        self.assertLessEqual(float(np.max(first)), 3.5 + 1e-12)
        sigma = np.sqrt(np.log1p(0.60 ** 2))
        mu = model.calibrated_capped_lognormal_mu(0.60, 3.5)
        self.assertAlmostEqual(
            model.capped_lognormal_mean(mu, sigma, 3.5), 1.0, places=10)

    def test_same_candidate_has_identical_ccp_values(self):
        arc = model.Arc(
            from_node="A", to_node="B", mode="road",
            distance=60.0, capacity=1000.0,
            cost_per_teu_km=1.0, emission_per_teu_km=2.0,
            speed_kmh=60.0)
        path = model.Path(
            path_id=1, origin="A", destination="B",
            nodes=["A", "B"], modes=["road"], arcs=[arc],
            base_cost_per_teu=60.0,
            base_emission_per_teu=120.0,
            base_travel_time_h=1.0)
        batch = model.Batch(
            batch_id=1, origin="A", destination="B",
            quantity=1.0, ET=0.0, LT=5.0,
            penalty_per_teu_h=2.0, max_late_h=2.0)

        model.configure_scenario_set(
            arcs=[arc], border_delay_map={}, size=50,
            seed=1234, stochastic=True)

        def evaluate_once():
            individual = model.Individual(od_allocations={
                ("A", "B", 1): [model.PathAllocation(path=path, share=1.0)]
            })
            model.evaluate_individual(
                individual, [batch], [arc], {},
                waiting_cost_per_teu_h=0.0,
                wait_emis_g_per_teu_h=0.0)
            return individual

        first = evaluate_once()
        second = evaluate_once()
        self.assertEqual(first.objectives, second.objectives)
        self.assertTrue(np.array_equal(first.cost_s, second.cost_s))
        self.assertTrue(np.array_equal(first.emission_s, second.emission_s))
        self.assertTrue(np.array_equal(first.makespan_s, second.makespan_s))
        self.assertEqual(first.ev_objectives, second.ev_objectives)
        self.assertEqual(first.penalty, second.penalty)
        self.assertTrue(first.feasible)
        self.assertEqual(first.objectives[0], 60.0)
        self.assertEqual(first.objectives[1], 120.0)

    def test_border_delay_is_one_shared_directional_pair_event(self):
        definitions = model.load_border_event_definitions(
            model.DEFAULT_BORDER_EVENT_DATA_FILE)
        exact = model.Arc(
            from_node="Khorgos", to_node="Altynkol", mode="rail",
            distance=10.0, capacity=1000.0,
            cost_per_teu_km=1.0, emission_per_teu_km=1.0,
            speed_kmh=50.0, from_region="CN", to_region="CA",
            is_border_arc=True)
        legacy_alias = model.Arc(
            from_node="Khorgos", to_node="Almaty", mode="rail",
            distance=10.0, capacity=1000.0,
            cost_per_teu_km=1.0, emission_per_teu_km=1.0,
            speed_kmh=50.0, from_region="CN", to_region="CA",
            is_border_arc=True)

        scenarios = model.build_scenario_set(
            [exact, legacy_alias], border_delay_map={}, size=1, seed=7,
            stochastic=False, border_event_definitions=definitions)
        event_key = ("Khorgos", "Altynkol", "rail")
        self.assertEqual(scenarios.border_event_for_arc(exact), event_key)
        self.assertEqual(
            scenarios.border_event_for_arc(legacy_alias), event_key)
        self.assertEqual(scenarios.border_event_mean_h[event_key], 82.5)
        self.assertEqual(float(scenarios.border_delay_h[event_key][0]), 82.5)
        self.assertEqual(len(scenarios.border_delay_h), 1)

    def test_ev_scenario_uses_direct_expectations_without_sampling(self):
        arc = model.Arc(
            from_node="A", to_node="B", mode="road", distance=10.0,
            capacity=1000.0, cost_per_teu_km=1.0,
            emission_per_teu_km=1.0, speed_kmh=40.0,
            from_region="CA", to_region="RU", is_border_arc=True)
        first = model.build_expected_value_scenario_set(
            [arc], {}, seed=11,
            border_event_definitions={
                ("*", "*", "road"): model.BorderEventDefinition(
                    "*", "*", "road", 12.0)})
        second = model.build_expected_value_scenario_set(
            [arc], {}, seed=99,
            border_event_definitions={
                ("*", "*", "road"): model.BorderEventDefinition(
                    "*", "*", "road", 12.0)})
        key = ("A", "B", "road")
        self.assertEqual(first.size, 1)
        self.assertFalse(first.stochastic)
        self.assertEqual(float(first.travel_multiplier[key][0]), 1.0)
        self.assertEqual(float(first.border_delay_h[key][0]), 12.0)
        self.assertTrue(np.array_equal(first.travel_multiplier[key],
                                       second.travel_multiplier[key]))
        self.assertTrue(np.array_equal(first.border_delay_h[key],
                                       second.border_delay_h[key]))

    def test_general_road_bcp_uses_directional_arc_event(self):
        definitions = model.load_border_event_definitions(
            model.DEFAULT_BORDER_EVENT_DATA_FILE)
        arc = model.Arc(
            from_node="A", to_node="B", mode="road",
            distance=10.0, capacity=1000.0,
            cost_per_teu_km=1.0, emission_per_teu_km=1.0,
            speed_kmh=40.0, from_region="CA", to_region="RU",
            is_border_arc=True)
        scenarios = model.build_scenario_set(
            [arc], border_delay_map={}, size=1, seed=7,
            stochastic=False, border_event_definitions=definitions)
        event_key = ("A", "B", "road")
        self.assertEqual(scenarios.border_event_for_arc(arc), event_key)
        self.assertEqual(scenarios.border_event_mean_h[event_key], 9.9)

    def test_timetable_controls_waiting_not_nominal_arc_speed(self):
        arc = model.Arc(
            from_node="A", to_node="B", mode="rail",
            distance=100.0, capacity=1000.0,
            cost_per_teu_km=1.0, emission_per_teu_km=1.0,
            speed_kmh=50.0)
        entry = model.TimetableEntry(
            from_node="A", to_node="B", mode="rail",
            frequency_per_week=7.0, first_departure_hour=8.0,
            headway_hours=24.0, travel_time_h=999.0,
            legacy_time_value=300.0)
        timetable = {("A", "B", "rail"): [entry]}
        self.assertEqual(model.nominal_arc_travel_time(arc, timetable), 2.0)
        self.assertEqual(model.next_departure_time_programB(9.0, [entry]), 32.0)


class FeasibilityFirstSearchTests(unittest.TestCase):
    @staticmethod
    def _path(path_id, from_node, to_node):
        arc = model.Arc(
            from_node=from_node, to_node=to_node, mode="road",
            distance=1.0, capacity=10.0,
            cost_per_teu_km=1.0, emission_per_teu_km=1.0,
            speed_kmh=1.0)
        return model.Path(
            path_id=path_id, origin="O", destination="D",
            nodes=[from_node, to_node], modes=["road"], arcs=[arc],
            base_cost_per_teu=1.0,
            base_emission_per_teu=1.0,
            base_travel_time_h=1.0)

    def test_capacity_aware_choice_coordinates_batches(self):
        batches = [
            model.Batch(1, "O", "D", 4.0, 0.0, 10.0),
            model.Batch(2, "O", "D", 4.0, 0.0, 10.0),
        ]
        shared_1 = self._path(1, "S", "T")
        shared_2 = self._path(2, "S", "T")
        alt_1 = self._path(3, "A", "B")
        alt_2 = self._path(4, "C", "D")

        def option(path, arc_key):
            return model.ReliablePathOption(
                path=path, on_time_probability=1.0, max_lateness_h=0.0,
                resources={("arc", arc_key, 0): 4.0})

        reliable = {
            ("O", "D", 1): [
                option(shared_1, ("S", "T", "road")),
                option(alt_1, ("A", "B", "road")),
            ],
            ("O", "D", 2): [
                option(shared_2, ("S", "T", "road")),
                option(alt_2, ("C", "D", "road")),
            ],
        }
        capacities = {
            ("S", "T", "road"): 5.0,
            ("A", "B", "road"): 10.0,
            ("C", "D", "road"): 10.0,
        }
        random.seed(7)
        choice, excess = model.find_capacity_aware_choice(
            batches, reliable, capacities, restarts=5, iterations=20)

        self.assertIsNotNone(choice)
        self.assertEqual(excess, 0.0)
        self.assertFalse(choice[0] == 0 and choice[1] == 0)

    def test_node_without_positive_capacity_is_not_constrained(self):
        previous = dict(model.BORDER_CAPACITY)
        try:
            model.BORDER_CAPACITY["UnconstrainedPort"] = 0.0
            capacity = model._resource_available_capacity(
                ("node", "UnconstrainedPort", 0), {})
            self.assertEqual(capacity, float("inf"))
        finally:
            model.BORDER_CAPACITY.clear()
            model.BORDER_CAPACITY.update(previous)

    def test_structural_crossover_preserves_complete_single_path_genes(self):
        batch = model.Batch(1, "O", "D", 1.0, 0.0, 10.0)
        p1 = self._path(1, "A", "B")
        p2 = self._path(2, "C", "D")
        key = ("O", "D", 1)
        parent_1 = model.Individual({
            key: [model.PathAllocation(p1, 1.0)]})
        parent_2 = model.Individual({
            key: [model.PathAllocation(p2, 1.0)]})

        random.seed(11)
        for _ in range(20):
            child_1, child_2 = model.crossover_structural(
                parent_1, parent_2, [batch])
            self.assertEqual(len(child_1.od_allocations[key]), 1)
            self.assertEqual(len(child_2.od_allocations[key]), 1)
            self.assertEqual(child_1.od_allocations[key][0].share, 1.0)
            self.assertEqual(child_2.od_allocations[key][0].share, 1.0)

    def test_infeasible_dominance_uses_normalized_violation(self):
        lower_violation = model.Individual(
            objectives=(10.0, 10.0, 10.0), penalty=1e12,
            feasible=False, normalized_violation=0.01)
        higher_violation = model.Individual(
            objectives=(1.0, 1.0, 1.0), penalty=1.0,
            feasible=False, normalized_violation=0.02)

        self.assertTrue(model.dominates(lower_violation, higher_violation))
        self.assertFalse(model.dominates(higher_violation, lower_violation))


class ModeSpeedPrecedenceTests(unittest.TestCase):
    def test_fallback_used_only_when_no_higher_priority_value_exists(self):
        fallback = {"road": 40.0, "rail": 50.0, "water": 28.0}
        effective, source = model.resolve_effective_speed_map(
            {"road": None, "rail": None, "water": None}, {}, fallback)
        self.assertEqual(effective, fallback)
        self.assertEqual(
            source, {m: "hardcoded_fallback" for m in fallback})

    def test_workbook_speed_overrides_hardcoded_fallback(self):
        fallback = {"road": 40.0, "rail": 50.0, "water": 28.0}
        workbook = {"road": 60.0, "rail": 120.0, "water": 35.0}
        effective, source = model.resolve_effective_speed_map(
            {"road": None, "rail": None, "water": None}, workbook, fallback)
        self.assertEqual(effective, workbook)
        self.assertEqual(
            source, {m: "workbook_mode_speeds" for m in fallback})

    def test_cli_override_beats_workbook_and_fallback(self):
        fallback = {"road": 40.0, "rail": 50.0, "water": 28.0}
        workbook = {"road": 60.0, "rail": 120.0, "water": 35.0}
        effective, source = model.resolve_effective_speed_map(
            {"road": 55.0, "rail": None, "water": None}, workbook, fallback)
        self.assertEqual(effective["road"], 55.0)
        self.assertEqual(source["road"], "cli_override")
        # Untouched modes still fall through to workbook, not fallback.
        self.assertEqual(effective["rail"], 120.0)
        self.assertEqual(source["rail"], "workbook_mode_speeds")
        self.assertEqual(effective["water"], 35.0)
        self.assertEqual(source["water"], "workbook_mode_speeds")

    def test_non_positive_workbook_value_falls_through_to_fallback(self):
        fallback = {"road": 40.0, "rail": 50.0, "water": 28.0}
        workbook = {"road": 0.0, "rail": -5.0, "water": 35.0}
        effective, source = model.resolve_effective_speed_map(
            {"road": None, "rail": None, "water": None}, workbook, fallback)
        self.assertEqual(effective["road"], 40.0)
        self.assertEqual(source["road"], "hardcoded_fallback")
        self.assertEqual(effective["rail"], 50.0)
        self.assertEqual(source["rail"], "hardcoded_fallback")
        self.assertEqual(effective["water"], 35.0)
        self.assertEqual(source["water"], "workbook_mode_speeds")


class PositiveSpeedArgValidationTests(unittest.TestCase):
    """--{mode}-speed-kmh must reject <= 0 with a clear CLI error rather
    than silently being treated as "no override" -- covers the CLI
    validation gap Codex flagged in the prior review."""

    def test_positive_cli_override_is_accepted_and_wins(self):
        # positive_speed_kmh_arg is argparse's `type=` callable: it must
        # accept a valid positive value and return it as a float, which
        # then flows into resolve_effective_speed_map exactly like any
        # other explicit CLI override (cli_override beats workbook).
        parsed = model.positive_speed_kmh_arg("55")
        self.assertEqual(parsed, 55.0)
        fallback = {"road": 40.0, "rail": 50.0, "water": 28.0}
        workbook = {"road": 60.0, "rail": 120.0, "water": 35.0}
        effective, source = model.resolve_effective_speed_map(
            {"road": parsed, "rail": None, "water": None}, workbook, fallback)
        self.assertEqual(effective["road"], 55.0)
        self.assertEqual(source["road"], "cli_override")

    def test_zero_cli_override_is_rejected(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            model.positive_speed_kmh_arg("0")

    def test_negative_cli_override_is_rejected(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            model.positive_speed_kmh_arg("-10")

    def test_non_numeric_cli_override_is_rejected(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            model.positive_speed_kmh_arg("not-a-number")

    def test_nan_cli_override_is_rejected(self):
        # float("nan") parses successfully and `nan <= 0.0` is False (NaN
        # comparisons are always False), so a bare `<= 0` check would let
        # this silently through -- math.isfinite() must catch it.
        with self.assertRaises(argparse.ArgumentTypeError):
            model.positive_speed_kmh_arg("nan")

    def test_positive_infinity_cli_override_is_rejected(self):
        # float("inf") parses successfully and `inf > 0.0` is True, so a
        # bare `<= 0` check would ACCEPT it as the effective speed --
        # math.isfinite() must catch it.
        with self.assertRaises(argparse.ArgumentTypeError):
            model.positive_speed_kmh_arg("inf")

    def test_negative_infinity_cli_override_is_rejected(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            model.positive_speed_kmh_arg("-inf")

    def test_positive_finite_value_is_accepted(self):
        self.assertEqual(model.positive_speed_kmh_arg("72.5"), 72.5)

    def test_production_speed_arguments_reject_invalid_values(self):
        # Uses the REAL production parser (model.build_arg_parser()), not a
        # locally reconstructed approximation of it -- this is the
        # corrected version of the previous test, whose comment claimed to
        # confirm "the actual CLI flag definitions" while actually
        # exercising a hand-built parser that could pass even if the
        # production --road/rail/water-speed-kmh flags stopped using the
        # validator. All three production flags are checked, not just road.
        for flag in ("--road-speed-kmh", "--rail-speed-kmh", "--water-speed-kmh"):
            parser = model.build_arg_parser()
            for invalid in ("0", "-5", "nan", "inf", "-inf", "not-a-number"):
                with self.assertRaises(SystemExit):
                    parser.parse_args([flag, invalid])

    def test_production_speed_arguments_accept_valid_positive_values(self):
        parser = model.build_arg_parser()
        args = parser.parse_args([
            "--road-speed-kmh", "55",
            "--rail-speed-kmh", "90",
            "--water-speed-kmh", "31.5",
        ])
        self.assertEqual(args.road_speed_kmh, 55.0)
        self.assertEqual(args.rail_speed_kmh, 90.0)
        self.assertEqual(args.water_speed_kmh, 31.5)

    def test_production_speed_arguments_default_to_none_when_omitted(self):
        # None is the sentinel resolve_effective_speed_map() relies on to
        # distinguish "not explicitly overridden" from "user typed 0" --
        # confirms the production defaults are still None, not silently
        # reverted to DEFAULT_MODE_SPEED_KMH (which would defeat CLI >
        # workbook precedence: a "default" value would look identical to
        # an explicit override).
        parser = model.build_arg_parser()
        args = parser.parse_args([])
        self.assertIsNone(args.road_speed_kmh)
        self.assertIsNone(args.rail_speed_kmh)
        self.assertIsNone(args.water_speed_kmh)


class UncertaintyParameterTests(unittest.TestCase):
    def test_full_travel_time_cv_mapping(self):
        self.assertEqual(model.MODE_TIME_CV["road"], 0.15)
        self.assertEqual(model.MODE_TIME_CV["rail"], 0.10)
        self.assertEqual(model.MODE_TIME_CV["water"], 0.20)

    def test_water_travel_time_cv_is_020(self):
        self.assertEqual(model.MODE_TIME_CV["water"], 0.20)

    def test_full_border_delay_cv_mapping(self):
        self.assertEqual(model.BORDER_DELAY_CV["road"], 0.45)
        self.assertEqual(model.BORDER_DELAY_CV["rail"], 0.60)
        self.assertEqual(model.BORDER_DELAY_CV["water"], 0.40)

    def test_water_border_delay_cv_is_040_and_distinct_from_travel_cv(self):
        self.assertEqual(model.BORDER_DELAY_CV["water"], 0.40)
        # Confirm the two water CVs are genuinely separate constants, not
        # the same dict/value reused for both purposes.
        self.assertNotEqual(
            model.MODE_TIME_CV["water"], model.BORDER_DELAY_CV["water"])

    def test_travel_time_and_border_delay_cv_dicts_are_separate_objects(self):
        # Guards against a future refactor accidentally aliasing the two
        # dicts (e.g. `BORDER_DELAY_CV = MODE_TIME_CV`), which would make
        # every per-mode value above trivially "equal" for the wrong
        # reason.
        self.assertIsNot(model.MODE_TIME_CV, model.BORDER_DELAY_CV)
        for mode in ("road", "rail"):
            self.assertNotEqual(
                model.MODE_TIME_CV[mode], model.BORDER_DELAY_CV[mode])

    def test_zero_maritime_mean_produces_zero_stochastic_water_border_delay(self):
        arc = model.Arc(
            from_node="Shanghai", to_node="SeaLane", mode="water",
            distance=100.0, capacity=1000.0,
            cost_per_teu_km=1.0, emission_per_teu_km=1.0,
            speed_kmh=35.0, is_border_arc=True)
        definitions = {
            ("Shanghai", "SeaLane", "water"): model.BorderEventDefinition(
                exit_node="Shanghai", entry_node="SeaLane", mode="water",
                mean_delay_h=0.0),
        }
        scenarios = model.build_scenario_set(
            [arc], border_delay_map={}, size=50, seed=7,
            stochastic=True, border_event_definitions=definitions)
        event_key = ("Shanghai", "SeaLane", "water")
        self.assertEqual(scenarios.border_event_mean_h[event_key], 0.0)
        self.assertTrue(
            np.all(scenarios.border_delay_h[event_key] == 0.0))


class RiskMetricModeTests(unittest.TestCase):
    """Deterministic / EV / CCP share build_scenario_set +
    simulate_path_over_scenarios; only the final aggregation and the
    CCP-only feasibility gating differ by mode."""

    def _chance_violation_fixture(self):
        """A single-batch instance where 3 of 4 frozen scenarios arrive
        on time and 1 arrives late enough to breach the on-time chance
        constraint (0.75 on-time probability < default 0.90 confidence)
        while capacity/allocation/timetable remain trivially satisfied and
        the maximum-lateness constraint is NOT breached (isolates the test
        to the chance-constraint pathway only)."""
        arc = model.Arc(
            from_node="A", to_node="B", mode="road",
            distance=100.0, capacity=1000.0,
            cost_per_teu_km=1.0, emission_per_teu_km=1.0,
            speed_kmh=50.0)
        path = model.Path(
            path_id=1, origin="A", destination="B",
            nodes=["A", "B"], modes=["road"], arcs=[arc],
            base_cost_per_teu=100.0,
            base_emission_per_teu=200.0,
            base_travel_time_h=2.0)
        batch = model.Batch(
            batch_id=1, origin="A", destination="B",
            quantity=1.0, ET=0.0, LT=2.5,
            penalty_per_teu_h=1.0, max_late_h=5.0)
        scenario_set = model.ScenarioSet(
            size=4, seed=1,
            travel_multiplier={
                ("A", "B", "road"): np.array([1.0, 1.0, 1.0, 2.0])},
            border_delay_h={}, arc_border_event={},
            border_event_mean_h={}, stochastic=True)
        individual = model.Individual(od_allocations={
            ("A", "B", 1): [model.PathAllocation(path=path, share=1.0)]
        })
        return arc, path, batch, scenario_set, individual

    def _evaluate_with_mode(self, mode):
        arc, path, batch, scenario_set, individual = (
            self._chance_violation_fixture())
        previous_mode = model.RISK_METRIC
        previous_scenarios = model.ACTIVE_SCENARIO_SET
        model.RISK_METRIC = mode
        model.ACTIVE_SCENARIO_SET = scenario_set
        try:
            model.evaluate_individual(
                individual, [batch], [arc], {},
                waiting_cost_per_teu_h=0.0, wait_emis_g_per_teu_h=0.0)
        finally:
            model.RISK_METRIC = previous_mode
            model.ACTIVE_SCENARIO_SET = previous_scenarios
        return individual

    def test_low_on_time_probability_does_not_make_ccp_infeasible(self):
        individual = self._evaluate_with_mode("ccp")
        self.assertAlmostEqual(
            individual.vio_breakdown["min_on_time_prob"], 0.75)
        self.assertNotIn("chance_vio", individual.vio_breakdown)
        self.assertTrue(individual.feasible)
        self.assertTrue(individual.feasible_hard)
        self.assertEqual(individual.normalized_violation, 0.0)
        self.assertEqual(individual.penalty, 0.0)

    def test_missing_allocation_remains_infeasible(self):
        arc, _, batch, scenario_set, _ = self._chance_violation_fixture()
        individual = model.Individual()
        previous_scenarios = model.ACTIVE_SCENARIO_SET
        model.ACTIVE_SCENARIO_SET = scenario_set
        try:
            model.evaluate_individual(
                individual, [batch], [arc], {},
                waiting_cost_per_teu_h=0.0, wait_emis_g_per_teu_h=0.0)
        finally:
            model.ACTIVE_SCENARIO_SET = previous_scenarios
        self.assertFalse(individual.feasible)
        self.assertGreater(individual.vio_breakdown["miss_alloc"], 0.0)
        self.assertGreater(individual.normalized_violation, 0.0)

    def test_arc_capacity_excess_remains_infeasible(self):
        arc, _, batch, scenario_set, individual = self._chance_violation_fixture()
        arc.capacity = 0.5
        previous_scenarios = model.ACTIVE_SCENARIO_SET
        model.ACTIVE_SCENARIO_SET = scenario_set
        try:
            model.evaluate_individual(
                individual, [batch], [arc], {},
                waiting_cost_per_teu_h=0.0, wait_emis_g_per_teu_h=0.0)
        finally:
            model.ACTIVE_SCENARIO_SET = previous_scenarios
        self.assertFalse(individual.feasible)
        self.assertGreater(individual.vio_breakdown["cap_excess"], 0.0)
        self.assertGreater(individual.normalized_violation, 0.0)

    def test_ev_mode_ignores_chance_constraint_for_feasibility(self):
        individual = self._evaluate_with_mode("ev")
        # Diagnostic is still computed and reported...
        self.assertAlmostEqual(
            individual.vio_breakdown["min_on_time_prob"], 0.75)
        self.assertNotIn("chance_vio", individual.vio_breakdown)
        # ...but must NOT affect feasibility or normalized_violation.
        self.assertTrue(individual.feasible)
        self.assertTrue(individual.feasible_hard)
        self.assertEqual(individual.normalized_violation, 0.0)

    def test_deterministic_mode_ignores_chance_constraint_for_feasibility(self):
        individual = self._evaluate_with_mode("deterministic")
        self.assertTrue(individual.feasible)
        self.assertTrue(individual.feasible_hard)
        self.assertEqual(individual.normalized_violation, 0.0)

    def test_ev_uses_existing_scenario_means_not_a_recomputation(self):
        individual = self._evaluate_with_mode("ev")
        expected_cost_mean = individual.vio_breakdown["scenario_cost_mean"]
        expected_emission_mean = individual.vio_breakdown[
            "scenario_emission_mean"]
        expected_time_mean = individual.vio_breakdown["scenario_time_mean"]
        self.assertEqual(individual.objectives, (
            expected_cost_mean, expected_emission_mean, expected_time_mean))
        # The late-delivery cost (scenario 4 arrives 1.5h late, penalty
        # 1.0 USD/TEU/h) must already be inside the per-scenario cost
        # array feeding this mean -- no separate lateness-risk term.
        # 3 scenarios cost 100 (base) + 0 late; 1 scenario costs 100 + 1.5
        # late => mean = 100 + 1.5/4 = 100.375
        self.assertAlmostEqual(expected_cost_mean, 100.375)

    def test_deterministic_mode_returns_first_scenario_value_not_a_mean(self):
        individual = self._evaluate_with_mode("deterministic")
        # This fixture keeps all 4 scenarios (real deployment forces S=1
        # upstream via configure_scenario_set(..., stochastic=False); see
        # the --risk-metric deterministic wiring in main). What this test
        # isolates is aggregate_scenario_objectives's own branch behaviour:
        # "deterministic" must return cost_s[0] directly (100.0, scenario 0
        # is on-time) rather than np.mean(cost_s) (which would be 100.375,
        # the EV value asserted in the sibling test above).
        self.assertEqual(individual.objectives[0], 100.0)
        self.assertNotEqual(individual.objectives[0], 100.375)

    def test_ccp_mode_gives_no_penalty_for_low_on_time_probability(self):
        individual = self._evaluate_with_mode("ccp")
        self.assertEqual(individual.penalty, 0.0)

    def test_ev_mode_gives_zero_ind_penalty_for_the_same_violation(self):
        individual = self._evaluate_with_mode("ev")
        self.assertEqual(individual.penalty, 0.0)

    def test_deterministic_mode_gives_zero_ind_penalty_for_the_same_violation(self):
        individual = self._evaluate_with_mode("deterministic")
        self.assertEqual(individual.penalty, 0.0)

    def test_frozen_scenario_set_not_regenerated_per_individual(self):
        arc, path, batch, scenario_set, individual_1 = (
            self._chance_violation_fixture())
        individual_2 = model.Individual(od_allocations=dict(
            individual_1.od_allocations))
        previous_scenarios = model.ACTIVE_SCENARIO_SET
        model.ACTIVE_SCENARIO_SET = scenario_set
        try:
            model.evaluate_individual(
                individual_1, [batch], [arc], {},
                waiting_cost_per_teu_h=0.0, wait_emis_g_per_teu_h=0.0)
            self.assertIs(model.ACTIVE_SCENARIO_SET, scenario_set)
            model.evaluate_individual(
                individual_2, [batch], [arc], {},
                waiting_cost_per_teu_h=0.0, wait_emis_g_per_teu_h=0.0)
            # Same frozen object identity across both evaluations: no
            # per-individual regeneration occurred.
            self.assertIs(model.ACTIVE_SCENARIO_SET, scenario_set)
        finally:
            model.ACTIVE_SCENARIO_SET = previous_scenarios


class ModeAwareReliablePathScreeningTests(unittest.TestCase):
    def test_ccp_and_ev_both_retain_late_structurally_valid_path(self):
        arc = model.Arc(
            from_node="A", to_node="B", mode="road",
            distance=100.0, capacity=1000.0,
            cost_per_teu_km=1.0, emission_per_teu_km=1.0,
            speed_kmh=50.0)
        path = model.Path(
            path_id=1, origin="A", destination="B",
            nodes=["A", "B"], modes=["road"], arcs=[arc],
            base_cost_per_teu=100.0,
            base_emission_per_teu=200.0,
            base_travel_time_h=2.0)
        # Same chance-violation shape as RiskMetricModeTests: 1 of 4
        # scenarios arrives late enough to breach on-time probability but
        # not the (generous) maximum-lateness limit.
        batch = model.Batch(
            batch_id=1, origin="A", destination="B",
            quantity=1.0, ET=0.0, LT=2.5,
            penalty_per_teu_h=1.0, max_late_h=5.0)
        scenario_set = model.ScenarioSet(
            size=4, seed=1,
            travel_multiplier={
                ("A", "B", "road"): np.array([1.0, 1.0, 1.0, 2.0])},
            border_delay_h={}, arc_border_event={},
            border_event_mean_h={}, stochastic=True)
        path_lib = {("A", "B"): [path]}

        ccp_options = model.build_reliable_path_options(
            [batch], path_lib, {}, {}, {}, scenario_set, mode="ccp")
        ev_options = model.build_reliable_path_options(
            [batch], path_lib, {}, {}, {}, scenario_set, mode="ev")

        key = ("A", "B", 1)
        self.assertEqual(len(ccp_options[key]), 1)
        self.assertEqual(len(ev_options[key]), 1)
        self.assertIs(ccp_options[key][0].path, path)
        self.assertIs(ev_options[key][0].path, path)


class ProductionCLIRiskMetricWiringTests(unittest.TestCase):
    """--risk-metric parsed by the REAL production parser, then fed
    through the REAL resolve_stochastic_eval_and_scenarios() and
    configure_scenario_set() -- confirms deterministic mode genuinely
    forces S=1 in the actual configuration path main() uses, not a
    re-implementation of the rule."""

    def test_risk_metric_choices_parse_correctly(self):
        parser = model.build_arg_parser()
        for choice in ("deterministic", "ev", "ccp"):
            args = parser.parse_args(["--risk-metric", choice])
            self.assertEqual(args.risk_metric, choice)
        # Default (no flag passed) preserves original behaviour.
        self.assertEqual(parser.parse_args([]).risk_metric, "ccp")

    def test_risk_metric_rejects_unknown_choice(self):
        parser = model.build_arg_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["--risk-metric", "not-a-real-mode"])

    def _arc(self):
        return model.Arc(
            from_node="A", to_node="B", mode="road",
            distance=100.0, capacity=1000.0,
            cost_per_teu_km=1.0, emission_per_teu_km=1.0, speed_kmh=50.0)

    def test_deterministic_risk_metric_forces_s1_in_real_config_path(self):
        parser = model.build_arg_parser()
        args = parser.parse_args(
            ["--risk-metric", "deterministic", "--mc-scenarios", "500"])
        stochastic_eval, mc_scenarios = model.resolve_stochastic_eval_and_scenarios(
            args.risk_metric, args.no_stochastic, args.mc_scenarios)
        self.assertFalse(stochastic_eval)
        previous = model.ACTIVE_SCENARIO_SET
        try:
            scenario_set = model.configure_scenario_set(
                arcs=[self._arc()], border_delay_map={}, size=mc_scenarios,
                seed=1, stochastic=stochastic_eval)
            # --mc-scenarios 500 is genuinely ignored: the real scenario
            # generator collapses to a single nominal scenario.
            self.assertEqual(scenario_set.size, 1)
            self.assertFalse(scenario_set.stochastic)
        finally:
            model.ACTIVE_SCENARIO_SET = previous

    def test_ccp_preserves_requested_scenario_count(self):
        parser = model.build_arg_parser()
        args = parser.parse_args(
            ["--risk-metric", "ccp", "--mc-scenarios", "37"])
        stochastic_eval, mc_scenarios = (
            model.resolve_stochastic_eval_and_scenarios(
                args.risk_metric, args.no_stochastic, args.mc_scenarios))
        self.assertTrue(stochastic_eval)
        self.assertEqual(mc_scenarios, 37)

    def test_ev_ignores_mc_count_and_uses_direct_expected_inputs(self):
        parser = model.build_arg_parser()
        args = parser.parse_args(
            ["--risk-metric", "ev", "--mc-scenarios", "37"])
        stochastic_eval, mc_scenarios = (
            model.resolve_stochastic_eval_and_scenarios(
                args.risk_metric, args.no_stochastic, args.mc_scenarios))
        self.assertFalse(stochastic_eval)
        self.assertEqual(mc_scenarios, 37)
        scenario_set = model.build_expected_value_scenario_set(
            arcs=[self._arc()], border_delay_map={}, seed=1)
        self.assertEqual(scenario_set.size, 1)
        self.assertFalse(scenario_set.stochastic)


class ManifestProvenanceTests(unittest.TestCase):
    """Constructs/inspects the REAL scenario-manifest logic
    (build_scenario_manifest) per mode, rather than only checking the
    underlying constants exist."""

    def _manifest(self, risk_metric, scenario_size=50, scenario_seed=777,
                  mode_speed_kmh=None, mode_speed_source=None):
        scenario_set = model.ScenarioSet(
            size=scenario_size, seed=scenario_seed,
            travel_multiplier={}, border_delay_h={}, arc_border_event={},
            border_event_mean_h={}, stochastic=(risk_metric != "deterministic"))
        mode_speed_kmh = mode_speed_kmh or {
            "road": 60.0, "rail": 120.0, "water": 35.0}
        mode_speed_source = mode_speed_source or {
            "road": "workbook_mode_speeds", "rail": "workbook_mode_speeds",
            "water": "workbook_mode_speeds"}
        return model.build_scenario_manifest(
            risk_metric=risk_metric,
            scenario_set=scenario_set,
            confidence_cost=0.9, confidence_emission=0.9,
            confidence_time=0.9,
            mode_time_cv={"road": 0.15, "rail": 0.10, "water": 0.20},
            mode_time_cap_factor={"road": 2.0, "rail": 1.8, "water": 2.5},
            mode_speed_kmh=mode_speed_kmh,
            mode_speed_source=mode_speed_source,
            mode_speed_requested_cli_override_kmh={
                "road": None, "rail": None, "water": None},
            border_delay_cv={"road": 0.45, "rail": 0.60, "water": 0.40},
            border_delay_cap_factor=3.5,
            border_event_data_file="dummy_border_events.csv",
            border_event_definitions={},
            late_penalty_source="sourced_common_daily_rate_converted_to_hourly",
            late_penalty_input_basis="hourly_rate",
            late_penalty_baseline_usd_per_teu_h=6.25,
            use_input_late_penalties=False,
            payload_tonnes_per_teu=10.0,
            wait_emission_g_per_teu_h=0.0,
            feasibility_seed_fraction=0.3,
            feasibility_search_restarts=40,
            feasibility_search_iterations=250,
        )

    def test_risk_metric_field_matches_selected_mode(self):
        for mode in ("deterministic", "ev", "ccp"):
            self.assertEqual(self._manifest(mode)["risk_metric"], mode)

    def test_fitness_evaluation_is_mode_correct(self):
        for mode in ("deterministic", "ev", "ccp"):
            self.assertEqual(
                self._manifest(mode)["fitness_evaluation"],
                model.FITNESS_EVALUATION_DESCRIPTION[mode])

    def test_deterministic_and_ev_descriptions_exclude_ccp_only_wording(self):
        det = self._manifest("deterministic")["fitness_evaluation"]
        ev = self._manifest("ev")["fitness_evaluation"]
        for ccp_only_term in ("order_statistics", "chance-constrained",
                               "chance_constrained"):
            self.assertNotIn(ccp_only_term, det)
            self.assertNotIn(ccp_only_term, ev)
        # And the CCP description genuinely does use this language, so the
        # exclusion check above is meaningful rather than vacuous.
        self.assertIn(
            "order_statistics", self._manifest("ccp")["fitness_evaluation"])

    def test_mode_speed_kmh_records_effective_map_not_a_raw_fallback(self):
        manifest = self._manifest(
            "ev",
            mode_speed_kmh={"road": 72.0, "rail": 50.0, "water": 28.0},
            mode_speed_source={
                "road": "cli_override", "rail": "hardcoded_fallback",
                "water": "hardcoded_fallback"})
        self.assertEqual(manifest["mode_speed_kmh"]["road"], 72.0)
        self.assertEqual(manifest["mode_speed_source"]["road"], "cli_override")
        self.assertEqual(
            manifest["mode_speed_source"]["rail"], "hardcoded_fallback")

    def test_scenario_count_and_seed_recorded_correctly(self):
        manifest = self._manifest("ccp", scenario_size=321, scenario_seed=99)
        self.assertEqual(manifest["scenario_count"], 321)
        self.assertEqual(manifest["scenario_seed"], 99)

    def test_deterministic_manifest_records_non_stochastic_scenario_set(self):
        manifest = self._manifest("deterministic", scenario_size=1)
        self.assertFalse(manifest["stochastic"])
        self.assertEqual(manifest["scenario_count"], 1)


class DeterministicModeVsBaseline3RegressionTests(unittest.TestCase):
    """Compares baseline3.py's (the approved deterministic baseline)
    evaluate_individual() against baseline_uncertainty.py's
    evaluate_individual() under risk_metric="deterministic", on the SAME
    hand-computed two-arc instance. This is not a text/structural
    comparison of the two files -- it is a numerical regression check that
    deterministic mode reproduces the validated deterministic pipeline's
    arithmetic (arc travel time, arrival propagation, transport cost,
    lateness cost, emissions, makespan) on a controlled case.

    Both arcs are road (no timetable lookup in either model), and no
    border event is registered for these synthetic node names in either
    model, so travel-time/border mechanics are deliberately excluded from
    what differs between the two files here -- those are covered by the
    existing FrozenScenarioTests. What this class isolates is: does
    deterministic-mode evaluate_individual() add up the same objective
    values as baseline3.py's evaluate_individual() for the same flows?
    """

    TOL = 1e-9

    def _shared_arcs(self):
        # A -> B -> C, both road. distance/speed and cost/emission rates
        # differ per arc so the accumulation is genuinely exercised, not
        # just multiplied by a constant.
        arc1 = model.Arc(
            from_node="A", to_node="B", mode="road",
            distance=100.0, capacity=1000.0,
            cost_per_teu_km=2.0, emission_per_teu_km=3.0, speed_kmh=50.0)
        arc2 = model.Arc(
            from_node="B", to_node="C", mode="road",
            distance=50.0, capacity=1000.0,
            cost_per_teu_km=1.0, emission_per_teu_km=1.0, speed_kmh=25.0)
        return arc1, arc2

    def _shared_path_kwargs(self, arc1, arc2):
        # base_cost_per_teu / base_emission_per_teu / base_travel_time_h
        # are computed with the SAME formula both models' own path-library
        # builders use (sum of cost_per_teu_km*distance,
        # emission_per_teu_km*distance, distance/speed_kmh respectively) --
        # hand-computed here once and reused for both models' Path objects.
        base_cost = sum(a.cost_per_teu_km * a.distance for a in (arc1, arc2))
        base_emission = sum(
            a.emission_per_teu_km * a.distance for a in (arc1, arc2))
        base_travel_time = sum(
            a.distance / a.speed_kmh for a in (arc1, arc2))
        self.assertAlmostEqual(base_cost, 250.0)
        self.assertAlmostEqual(base_emission, 350.0)
        self.assertAlmostEqual(base_travel_time, 4.0)
        return dict(
            path_id=1, origin="A", destination="C",
            nodes=["A", "B", "C"], modes=["road", "road"], arcs=[arc1, arc2],
            base_cost_per_teu=base_cost,
            base_emission_per_teu=base_emission,
            base_travel_time_h=base_travel_time)

    def _run_baseline3(self, batch):
        arc1, arc2 = self._shared_arcs()
        path = baseline3.Path(**self._shared_path_kwargs(arc1, arc2))
        b3_batch = baseline3.Batch(
            batch_id=batch.batch_id, origin=batch.origin,
            destination=batch.destination, quantity=batch.quantity,
            ET=batch.ET, LT=batch.LT,
            penalty_per_teu_h=batch.penalty_per_teu_h)
        individual = baseline3.Individual(od_allocations={
            ("A", "C", batch.batch_id): [
                baseline3.PathAllocation(path=path, share=1.0)]
        })
        baseline3.evaluate_individual(
            individual, [b3_batch], [arc1, arc2], {},
            waiting_cost_per_teu_h=0.0, wait_emis_g_per_teu_h=0.0)
        return individual

    def _run_baseline_uncertainty_deterministic(self, batch):
        arc1, arc2 = self._shared_arcs()
        path = model.Path(**self._shared_path_kwargs(arc1, arc2))
        individual = model.Individual(od_allocations={
            ("A", "C", batch.batch_id): [
                model.PathAllocation(path=path, share=1.0)]
        })
        previous_mode = model.RISK_METRIC
        previous_scenarios = model.ACTIVE_SCENARIO_SET
        model.RISK_METRIC = "deterministic"
        try:
            model.configure_scenario_set(
                arcs=[arc1, arc2], border_delay_map={}, size=1, seed=1,
                stochastic=False)
            model.evaluate_individual(
                individual, [batch], [arc1, arc2], {},
                waiting_cost_per_teu_h=0.0, wait_emis_g_per_teu_h=0.0)
        finally:
            model.RISK_METRIC = previous_mode
            model.ACTIVE_SCENARIO_SET = previous_scenarios
        return individual

    def test_on_time_batch_matches_baseline3_exactly(self):
        # Arrival = 4.0h <= LT=5.0h: on time in both models.
        batch = model.Batch(
            batch_id=1, origin="A", destination="C",
            quantity=3.0, ET=0.0, LT=5.0, penalty_per_teu_h=2.0)
        b3 = self._run_baseline3(batch)
        bu = self._run_baseline_uncertainty_deterministic(batch)

        # Hand-verified expected values (see _shared_path_kwargs for the
        # base_cost/base_emission/base_travel_time derivation):
        #   flow = 3.0, arrival = 0 + 4.0 = 4.0 <= LT=5.0 -> no late cost
        #   cost = base_cost_per_teu * flow = 250.0 * 3.0 = 750.0
        #   emission = base_emission_per_teu * flow = 350.0 * 3.0 = 1050.0
        #   makespan = arrival = 4.0
        self.assertAlmostEqual(b3.objectives[0], 750.0, delta=self.TOL)
        self.assertAlmostEqual(b3.objectives[1], 1050.0, delta=self.TOL)
        self.assertAlmostEqual(b3.objectives[2], 4.0, delta=self.TOL)

        self.assertAlmostEqual(
            bu.objectives[0], b3.objectives[0], delta=self.TOL)
        self.assertAlmostEqual(
            bu.objectives[1], b3.objectives[1], delta=self.TOL)
        self.assertAlmostEqual(
            bu.objectives[2], b3.objectives[2], delta=self.TOL)

        # On-time and otherwise unconstrained: both models agree on
        # feasibility here too (no divergence to document for this case).
        self.assertTrue(b3.feasible)
        self.assertTrue(bu.feasible)

    def test_late_batch_matches_baseline3_cost_and_emission_but_feasibility_intentionally_diverges(self):
        # Arrival = 4.0h > LT=3.5h: 0.5h late. Deliberately exercises the
        # lateness-cost pathway in both models' cost accumulation.
        batch = model.Batch(
            batch_id=1, origin="A", destination="C",
            quantity=3.0, ET=0.0, LT=3.5, penalty_per_teu_h=2.0)
        b3 = self._run_baseline3(batch)
        bu = self._run_baseline_uncertainty_deterministic(batch)

        # late_h = flow * (arrival - LT) = 3.0 * 0.5 = 1.5
        # cost = 250.0*3.0 + penalty_per_teu_h*late_h = 750 + 2.0*1.5 = 753.0
        # emission unchanged = 1050.0 (no emission penalty for lateness)
        # makespan unchanged = 4.0
        self.assertAlmostEqual(b3.objectives[0], 753.0, delta=self.TOL)
        self.assertAlmostEqual(b3.objectives[1], 1050.0, delta=self.TOL)
        self.assertAlmostEqual(b3.objectives[2], 4.0, delta=self.TOL)

        self.assertAlmostEqual(
            bu.objectives[0], b3.objectives[0], delta=self.TOL)
        self.assertAlmostEqual(
            bu.objectives[1], b3.objectives[1], delta=self.TOL)
        self.assertAlmostEqual(
            bu.objectives[2], b3.objectives[2], delta=self.TOL)

        # Intentional, approved divergence (NOT a bug): baseline3.py's
        # hard_ok requires late_teu_h_total <= 1e-9 -- ANY lateness makes
        # it hard-infeasible. baseline_uncertainty.py's deterministic mode
        # deliberately excludes CCP-only constraints (including the
        # lateness-driven ones) from hard_ok per the approved architecture
        # -- lateness is a cost there, not a hard constraint, outside CCP
        # mode. Asserting this explicitly documents the divergence rather
        # than silently hiding it.
        self.assertFalse(b3.feasible)
        self.assertTrue(bu.feasible)


class VioBreakdownJsonExportTests(unittest.TestCase):
    """Regression tests for the export_pareto_points_json /
    export_best_infeasible_json fix: vio_breakdown legitimately mixes
    numeric diagnostics with a "risk_metric" string field (see
    evaluate_individual), and the exporter must serialize this
    heterogeneous dict correctly instead of blindly calling float() on
    every value."""

    @staticmethod
    def _individual(vio_breakdown):
        return model.Individual(
            objectives=(100.0, 200.0, 10.0), penalty=0.0, feasible=True,
            feasible_hard=True, vio_breakdown=dict(vio_breakdown))

    def _export_and_reload(self, individual):
        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        try:
            model.export_pareto_points_json([individual], [], out_json=path)
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        finally:
            os.remove(path)

    # --- A. risk_metric string preservation (EV) ---
    def test_risk_metric_ev_string_is_preserved_through_export(self):
        ind = self._individual({"risk_metric": "ev", "chance_vio": 0.0})
        data = self._export_and_reload(ind)
        self.assertEqual(data[0]["vio_breakdown"]["risk_metric"], "ev")
        self.assertIsInstance(data[0]["vio_breakdown"]["risk_metric"], str)

    # --- D. risk_metric string preservation (CCP) ---
    def test_risk_metric_ccp_string_is_preserved_through_export(self):
        ind = self._individual({"risk_metric": "ccp", "chance_vio": 0.15})
        data = self._export_and_reload(ind)
        self.assertEqual(data[0]["vio_breakdown"]["risk_metric"], "ccp")

    # --- B. numeric preservation ---
    def test_numeric_vio_breakdown_fields_remain_json_numbers(self):
        ind = self._individual({
            "risk_metric": "ev",
            "cap_excess": 12.5,
            "miss_alloc": 0,
        })
        data = self._export_and_reload(ind)
        vb = data[0]["vio_breakdown"]
        self.assertEqual(vb["cap_excess"], 12.5)
        self.assertNotIsInstance(vb["cap_excess"], str)
        self.assertEqual(vb["miss_alloc"], 0)
        self.assertNotIsInstance(vb["miss_alloc"], str)

    # --- C. NumPy scalar handling ---
    def test_numpy_scalar_vio_breakdown_values_serialize_correctly(self):
        ind = self._individual({
            "risk_metric": "ccp",
            "scenario_cost_mean": np.float64(4242.5),
            "chance_vio": np.int64(3),
        })
        data = self._export_and_reload(ind)
        vb = data[0]["vio_breakdown"]
        self.assertEqual(vb["scenario_cost_mean"], 4242.5)
        self.assertNotIsInstance(vb["scenario_cost_mean"], str)
        self.assertEqual(vb["chance_vio"], 3)
        self.assertNotIsInstance(vb["chance_vio"], str)

    def test_json_scalar_converts_numpy_types_to_native_python(self):
        # Direct unit test of the helper itself: numpy scalars must become
        # native Python int/float, not remain numpy types (json.dump
        # cannot serialize numpy scalar types on its own).
        converted_float = model._json_scalar(np.float64(1.5))
        converted_int = model._json_scalar(np.int32(7))
        self.assertIs(type(converted_float), float)
        self.assertIs(type(converted_int), int)

    def test_json_scalar_preserves_bool_not_as_int(self):
        # bool is a subtype of int in Python; the helper must special-case
        # it BEFORE the int/float branch or True/False would silently
        # pass the isinstance(value, (int, float)) check and be returned
        # as a bool anyway (harmlessly, in that specific case) -- but the
        # ordering itself is what this test pins down as intentional.
        self.assertIs(model._json_scalar(True), True)
        self.assertIs(model._json_scalar(False), False)

    def test_json_scalar_preserves_none(self):
        self.assertIsNone(model._json_scalar(None))

    # --- E. unsupported type: explicit failure, not silent masking ---
    def test_json_scalar_raises_on_unsupported_type(self):
        with self.assertRaises(TypeError):
            model._json_scalar([1, 2, 3])

    def test_export_raises_not_silently_corrupts_on_unsupported_vio_value(self):
        ind = self._individual({
            "risk_metric": "ev",
            "unsupported_diagnostic": {"nested": "dict"},
        })
        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        try:
            with self.assertRaises(TypeError):
                model.export_pareto_points_json([ind], [], out_json=path)
        finally:
            os.remove(path)

    # --- Companion export_best_infeasible_json shares the identical
    # blind-float-cast pattern this fix addresses; regression-test it too
    # so a second, latent crash (e.g. on an all-infeasible final
    # population) is not left unguarded. ---
    def test_best_infeasible_json_export_preserves_risk_metric_string(self):
        ind = self._individual({"risk_metric": "ccp", "chance_vio": 0.2})
        ind.feasible = False
        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        try:
            model.export_best_infeasible_json([ind], [], out_json=path)
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            self.assertEqual(data["vio_breakdown"]["risk_metric"], "ccp")
        finally:
            os.remove(path)


if __name__ == "__main__":
    unittest.main()
