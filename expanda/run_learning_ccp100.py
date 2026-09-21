#!/usr/bin/env python3
"""Stage 1: Random-CCP100 baseline and mutation training-data collection."""
import argparse
import hashlib
from pathlib import Path
import random
import subprocess
import time

import numpy as np

import baseline_uncertainty as base
from mutation_logging import MutationLogger, write_json
from run_ev_ccp_oos_pilot import candidate_rows, json_candidate, scenario_digest, summarise

ROOT = Path(__file__).resolve().parent


def main(argv=None):
    total_started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/data_expanded.xlsx")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pop", type=int, default=100)
    parser.add_argument("--gens", type=int, default=100)
    parser.add_argument("--algorithm-seed", type=int, default=981101)
    parser.add_argument("--training-seed", type=int, default=981201)
    parser.add_argument("--path-seed", type=int, default=0)
    parser.add_argument("--evaluation-budget", type=int, default=None,
                        help="Hard cap on actual training evaluations, including initialisation and boost.")
    parser.add_argument("--validate-oos", action="store_true",
                        help="Optional final-only 5000-scenario evaluation; never used for learning.")
    parser.add_argument("--validation-seed", type=int, default=981999)
    args = parser.parse_args(argv)
    if args.pop < 2 or args.pop % 2 or args.gens < 1:
        parser.error("pop must be even and >=2; gens must be >=1")
    if args.evaluation_budget is not None and args.evaluation_budget < args.pop + 4:
        parser.error("evaluation-budget must be at least pop+4")
    if args.training_seed == args.validation_seed:
        parser.error("training and validation seeds must differ")
    if args.out.exists() and any(args.out.iterdir()):
        parser.error("output directory must be empty; use a new --out for each run")
    args.out.mkdir(parents=True, exist_ok=True)
    base.BORDER_EVENT_DEFINITIONS = base.load_border_event_definitions(base.DEFAULT_BORDER_EVENT_DATA_FILE)
    network = base.load_network_from_extended(str(args.data))
    (nodes, regions, hold, proc, trans_cost, arcs, timetables, batches,
     wait_cost, wait_emission, carbon, emission_factors, speeds, trans_map,
     border_delay, theta, _) = network
    if len(batches) != 40:
        raise ValueError(f"Expected 40 batches, got {len(batches)}")
    for batch in batches:
        batch.penalty_per_teu_h = base.DEFAULT_LATE_PENALTY_USD_PER_TEU_H
    tt, lookup = base.build_timetable_dict(timetables), base.build_arc_lookup(arcs)
    random.seed(args.path_seed)
    np.random.seed(args.path_seed)
    paths = base.build_path_library(nodes, regions, arcs, batches, tt, lookup)
    base.sanity_check_path_lib(batches, paths)
    ev = base.build_expected_value_scenario_set(arcs, border_delay, seed=0,
        border_event_definitions=base.BORDER_EVENT_DEFINITIONS)
    options = base.build_reliable_path_options(batches, paths, tt, trans_map, border_delay, ev, mode="ev")
    scenarios = base.build_scenario_set(arcs, border_delay, 100, args.training_seed,
        stochastic=True, border_event_definitions=base.BORDER_EVENT_DEFINITIONS)
    digest = scenario_digest(scenarios)
    base.ACTIVE_SCENARIO_SET = scenarios
    base._PATH_SCENARIO_CACHE = {}
    base.RISK_METRIC = "ccp"
    base.CONFIDENCE_COST = base.CONFIDENCE_EMISSION = base.CONFIDENCE_TIME = .90
    random.seed(args.algorithm_seed)
    np.random.seed(args.algorithm_seed)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    config = dict(schema_version=1, method="Random-CCP100", stage="logging pilot; no trained model",
        population=args.pop, generations=args.gens, alpha=.90, training_size=100,
        seeds=dict(algorithm=args.algorithm_seed, training=args.training_seed, path=args.path_seed,
                   validation=args.validation_seed), git_commit=commit,
        scenario_digest=digest, evaluation_budget=args.evaluation_budget,
        stopping_rule="generation cap OR insufficient budget for a worst-case four-evaluation offspring pair",
        budget_note="May leave fewer than 4 evaluations unused; boost needs a conservative reserve. No budget overrun.",
        operator_probabilities=dict(zip(base.OPS, base._FIXED_OP_PROBS)),
        crossover_rate=base.CROSSOVER_RATE, mutation_rate=base.MUTATION_RATE,
        repair="encoding only; invalid mutation rolled back; no capacity repair",
        capacity_semantics="nominal planning capacity, not scenario-wise capacity",
        objective_units=dict(cost="USD", emission="gCO2", makespan="h"),
        training_feature_note="No OOS features/labels; infeasible objective labels masked, violations retained",
        waiting=base.waiting_emission_configuration(wait_emission),
        uncertainty=dict(travel_cv=base.MODE_TIME_CV, border_cv=base.BORDER_DELAY_CV,
                         travel_caps=base.MODE_TIME_MAX_FACTOR, border_cap=base.BORDER_DELAY_MAX_FACTOR),
        input_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
            (args.data, Path(base.DEFAULT_BORDER_EVENT_DATA_FILE), ROOT/"baseline_uncertainty.py",
             ROOT/"learning_mutation.py", ROOT/"mutation_logging.py", Path(__file__))},
        final_oos_size=5000 if args.validate_oos else 0)
    write_json(args.out / "configuration.json", config)
    run_id = f"random-ccp100-a{args.algorithm_seed}-s{args.training_seed}"
    logger = MutationLogger(args.out, run_id, digest, args.evaluation_budget)
    preparation_seconds = time.perf_counter() - total_started
    started = time.perf_counter()
    base.ACTIVE_MUTATION_LOGGER = logger
    try:
        population = base.run_nsga2(nodes, regions, hold, proc, trans_cost,
            arcs, timetables, batches, wait_cost, wait_emission, carbon, emission_factors,
            speeds, trans_map, border_delay, theta, paths, options,
            pop_size=args.pop, generations=args.gens)[0]
        logger.flush_events(population)
    finally:
        base.ACTIVE_MUTATION_LOGGER = None
        logger.close()
    runtime = time.perf_counter() - started
    fronts = base.fast_non_dominated_sort(population)
    front = [population[i] for i in fronts[0] if population[i].feasible]
    rows = candidate_rows("Random-CCP100", run_id, front, config)
    write_json(args.out / "final_feasible_nondominated.json", [json_candidate(r) for r in rows])
    if not rows:
        write_json(args.out / "best_infeasible.json",
                   json_candidate(candidate_rows("Random-CCP100", run_id,
                        [min(population, key=base.constraint_sort_key)], config)[0]))
    oos_seconds = 0.0
    if args.validate_oos and rows:
        oos_start = time.perf_counter()
        oos = base.build_scenario_set(arcs, border_delay, 5000, args.validation_seed,
            stochastic=True, border_event_definitions=base.BORDER_EVENT_DEFINITIONS)
        base.ACTIVE_SCENARIO_SET = oos
        base._PATH_SCENARIO_CACHE = {}
        summaries = []
        for row in rows:
            ind = row["individual"]
            base.evaluate_individual(ind, batches, arcs, tt, wait_cost, wait_emission,
                node_hold_cost=hold, node_proc_cost=proc, carbon_tax_map=carbon,
                trans_map=trans_map, border_delay_map=border_delay, theta_rm=theta,
                node_trans_cost=trans_cost)
            result = dict(decision_fingerprint=row["decision_fingerprint"])
            for name, values in zip(("cost", "emission", "makespan"), (ind.cost_s, ind.emission_s, ind.makespan_s)):
                training = row["optimisation_objectives"][name]
                result[name] = dict(**summarise(values), training_q90=training,
                                    coverage=float(np.mean(values <= training)))
            summaries.append(result)
        write_json(args.out / "oos_summary.json", dict(size=5000, seed=args.validation_seed,
            scenario_digest=scenario_digest(oos), post_oos_filtering=False, candidates=summaries))
        oos_seconds = time.perf_counter() - oos_start
    summary = dict(completed=True, method="Random-CCP100", ccp_evaluation_count=logger.evaluations,
        identical_evaluation_reuses=logger.cache_hits, mutation_attempts=logger.total_attempts,
        effective_modifications=logger.total_effective, final_pareto_size=len(rows),
        preparation_seconds=preparation_seconds, optimisation_seconds=runtime, oos_seconds=oos_seconds,
        total_seconds=time.perf_counter()-total_started,
        stop_reason="evaluation_budget" if not logger.has_budget(4) else "generation_limit",
        evaluation_budget=args.evaluation_budget, generations_completed=logger.generations_completed)
    write_json(args.out / "COMPLETE.json", summary)
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
