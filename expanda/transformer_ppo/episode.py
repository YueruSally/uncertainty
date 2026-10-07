"""One CCP100 PPO training or frozen-policy optimization episode."""
from __future__ import annotations

import hashlib
from pathlib import Path
import random
import time

import numpy as np

import baseline_uncertainty as base
from multi_instance_logging import fingerprint, write_json
from run_ev_ccp_oos_pilot import candidate_rows, json_candidate, scenario_digest

from .archive import RewardArchive, RewardConfig
from .controller import PPOController
from .instance_catalog import read_instance


def run_episode(data: Path, instance_path: Path, out: Path, trainer, normalizer,
                algorithm_seed: int, evaluation_budget=15_000,
                pop_size=100, generations=1_000, reward_config=None):
    if pop_size < 2 or pop_size % 2 or generations < 1:
        raise ValueError("pop_size must be even and >=2; generations must be positive")
    if evaluation_budget is not None and evaluation_budget < pop_size + 4:
        raise ValueError("evaluation_budget must be at least pop_size+4")
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"episode output directory must be empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    instance, batches = read_instance(instance_path)
    if hashlib.sha256(data.read_bytes()).hexdigest() != instance["source_data_sha256"]:
        raise ValueError("network workbook changed after catalog generation")
    started = time.perf_counter()
    base.BORDER_EVENT_DEFINITIONS = base.load_border_event_definitions(
        base.DEFAULT_BORDER_EVENT_DATA_FILE)
    network = base.load_network_from_extended(str(data))
    (nodes, regions, hold, proc, trans_cost, arcs, timetables, _raw,
     wait_cost, wait_emission, carbon, emission_factors, speeds, trans_map,
     border_delay, theta, _) = network
    for batch in batches:
        batch.penalty_per_teu_h = base.DEFAULT_LATE_PENALTY_USD_PER_TEU_H
    timetable = base.build_timetable_dict(timetables)
    lookup = base.build_arc_lookup(arcs)
    random.seed(instance["path_seed"])
    np.random.seed(instance["path_seed"] % (2**32 - 1))
    paths = base.build_path_library(nodes, regions, arcs, batches, timetable, lookup)
    base.sanity_check_path_lib(batches, paths)
    expected = base.build_expected_value_scenario_set(
        arcs, border_delay, seed=0,
        border_event_definitions=base.BORDER_EVENT_DEFINITIONS)
    options = base.build_reliable_path_options(
        batches, paths, timetable, trans_map, border_delay, expected, mode="ev")
    scenarios = base.build_scenario_set(
        arcs, border_delay, 100, instance["ccp100_seed"], stochastic=True,
        border_event_definitions=base.BORDER_EVENT_DEFINITIONS)
    base.ACTIVE_SCENARIO_SET = scenarios
    base._PATH_SCENARIO_CACHE = {}
    base.RISK_METRIC = "ccp"
    base.CONFIDENCE_COST = base.CONFIDENCE_EMISSION = base.CONFIDENCE_TIME = 0.90
    random.seed(algorithm_seed)
    np.random.seed(algorithm_seed % (2**32 - 1))
    seeds = {"algorithm": algorithm_seed, "path": instance["path_seed"],
             "training": instance["ccp100_seed"]}
    objective_divisor = (
        max(1.0, sum(float(batch.quantity) for batch in batches)),
        max(1.0, sum(float(batch.quantity) for batch in batches)),
        max(1.0, max(float(batch.LT) for batch in batches)
            - min(float(batch.ET) for batch in batches)),
    )
    episode_normalizer = normalizer.with_divisor(objective_divisor)
    config = {
        "schema_version": 1, "method": trainer.name,
        "instance_id": instance["instance_id"], "configuration": instance["configuration"],
        "instance_digest": instance["instance_digest"], "K": instance["K"],
        "deadline_window_alpha": instance["deadline_window_alpha"],
        "population": pop_size, "generations": generations,
        "alpha": 0.90,
        "evaluation_budget": evaluation_budget, "training_size": 100,
        "scenario_digest": scenario_digest(scenarios), "seeds": seeds,
        "operator_probabilities": dict(zip(base.OPS, base._FIXED_OP_PROBS)),
        "crossover_rate": base.CROSSOVER_RATE, "mutation_rate": base.MUTATION_RATE,
        "external_archive_role": "reward-only; excluded from environmental selection",
        "policy_training": bool(trainer.training),
        "normalization": {"minimum": episode_normalizer.minimum,
                          "maximum": episode_normalizer.maximum,
                          "reference": episode_normalizer.reference,
                          "objective_divisor": objective_divisor,
                          "definition": "cost/total_teu, emission/total_teu, makespan/instance_horizon"},
    }
    write_json(out / "configuration.json", config)
    archive = RewardArchive(episode_normalizer)
    controller = PPOController(
        out, instance["instance_id"], config["scenario_digest"], evaluation_budget,
        trainer=trainer, archive=archive,
        reward_config=reward_config or RewardConfig(), batches=batches,
        generations=generations, pop_size=pop_size, method=trainer.name,
        instance=instance, seeds=seeds)
    base.ACTIVE_MUTATION_LOGGER = controller
    try:
        population = base.run_nsga2(
            nodes, regions, hold, proc, trans_cost, arcs, timetables, batches,
            wait_cost, wait_emission, carbon, emission_factors, speeds, trans_map,
            border_delay, theta, paths, options, pop_size=pop_size,
            generations=generations)[0]
        controller.flush_events(population)
        controller.finish_episode()
    finally:
        base.ACTIVE_MUTATION_LOGGER = None
        controller.close()
    fronts = base.fast_non_dominated_sort(population)
    front = [population[index] for index in fronts[0] if population[index].feasible]
    rows = candidate_rows(trainer.name, instance["instance_id"], front, config)
    write_json(out / "final_feasible_nondominated.json", [json_candidate(row) for row in rows])
    write_json(out / "final_archive_lineage.json", [{
        "source_solution_id": row["source_solution_id"],
        "decision_fingerprint": fingerprint(row["individual"]),
        "optimisation_objectives": row["optimisation_objectives"],
    } for row in rows])
    summary = {
        "completed": True, "method": trainer.name,
        "ccp_evaluation_count": controller.evaluations,
        "mutation_attempts": controller.total_attempts,
        "effective_modifications": controller.total_effective,
        "no_eligible_candidate": controller.no_candidate,
        "inference_seconds": controller.inference_seconds,
        "optimisation_seconds": time.perf_counter() - started,
        "stop_reason": "evaluation_budget" if not controller.has_budget(4) else "generation_limit",
        "evaluation_budget": evaluation_budget,
        "generations_completed": controller.generations_completed,
        **controller.summary(),
    }
    write_json(out / "COMPLETE.json", summary)
    return summary
