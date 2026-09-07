import ast
import inspect

import numpy as np

import baseline_uncertainty as base
import run_ccp30_q90_oos_validation as validation


def test_exact_fixed_ccp30_source_and_training_metadata():
    rows = validation.load_ccp30_rows()
    assert len(rows) == 263
    validation.audit_training_semantics(rows)
    assert all(row["method"] == "CCP30" and row["run_id"] == 1 for row in rows)
    assert all(row["random_seeds"]["training"] == 920001 for row in rows)
    assert all(row["configuration"]["alpha"] == 0.9 for row in rows)


def test_training_audit_fails_closed_on_non_ccp_metadata(tmp_path):
    rows = validation.load_ccp30_rows()
    config = tmp_path / "configuration.json"
    config.write_text('{"method":"EV"}', encoding="utf-8")
    try:
        validation.audit_training_semantics(rows, config)
    except RuntimeError as error:
        assert "configuration mismatch" in str(error)
    else:
        raise AssertionError("expected non-CCP training metadata to fail")


def test_stored_s30_values_are_copied_from_optimisation_objectives():
    source = {
        "source_solution_id": "fixed-1",
        "optimisation_objectives": {
            "cost": 10.0, "emission": 20.0, "makespan": 30.0}}
    row = validation.comparison_row(source, (8.0, 25.0, 30.0))
    assert row["q90_30_cost"] == 10.0
    assert row["q90_30_emission"] == 20.0
    assert row["q90_30_makespan"] == 30.0
    assert row["relative_error_cost_pct"] == 25.0
    assert row["relative_error_emission_pct"] == 20.0
    assert row["relative_error_makespan_pct"] == 0.0


def test_oos_q90_requires_5000_outcomes_and_uses_ccp_aggregator(monkeypatch):
    calls = []
    def fake_aggregate(cost, emission, makespan, *means, **kwargs):
        calls.append((cost.size, emission.size, makespan.size, kwargs))
        return 1.0, 2.0, 3.0
    monkeypatch.setattr(base, "aggregate_scenario_objectives", fake_aggregate)
    values = np.arange(5000, dtype=float)
    assert validation.ccp_q90(values, values + 1, values + 2) == (1.0, 2.0, 3.0)
    assert calls == [(5000, 5000, 5000, {
        "mode": "ccp", "confidence_cost": 0.9,
        "confidence_emission": 0.9, "confidence_time": 0.9})]
    try:
        validation.ccp_q90(values[:-1], values, values)
    except RuntimeError as error:
        assert "5000 outcomes" in str(error)
    else:
        raise AssertionError("expected non-5000 OOS vectors to fail")


def test_empirical_q90_semantics_match_training_and_oos():
    training = np.arange(30, dtype=float)
    oos = np.arange(5000, dtype=float)
    assert base.empirical_ccp_quantile(training, 0.9) == 26.0
    expected = base.empirical_ccp_quantile(oos, 0.9)
    assert validation.ccp_q90(oos, oos, oos) == (expected, expected, expected)


def test_validation_constants_and_no_optimisation_or_pareto_operations():
    assert validation.OOS_SCENARIOS == 5000
    assert validation.OOS_SEED == 930001
    tree = ast.parse(inspect.getsource(validation))
    forbidden = {"run_nsga2", "crossover", "mutate", "mutation", "repair",
                 "fast_non_dominated_sort", "nondominated_mask"}
    called = {node.func.attr if isinstance(node.func, ast.Attribute)
              else node.func.id for node in ast.walk(tree)
              if isinstance(node, ast.Call)
              and isinstance(node.func, (ast.Attribute, ast.Name))}
    assert called.isdisjoint(forbidden)
