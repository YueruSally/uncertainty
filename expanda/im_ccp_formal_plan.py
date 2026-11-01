"""Frozen constants and dependency-light helpers for formal CCP I/M tests."""
from __future__ import annotations

import hashlib
import json


METHODS = ("BASE", "I_DIRECTED_BALANCED", "M_DIRECTED_BALANCED")
ALGORITHM_SEEDS = (
    79_000_001, 79_000_002, 79_000_003, 79_000_004, 79_000_005)
FINAL_INSTANCE_COUNT = 20
POPULATION = 250
GENERATIONS = 200
EVALUATION_BUDGET = POPULATION * (GENERATIONS + 1)
TRAINING_SCENARIO_COUNT = 200
OOS_SCENARIO_COUNT = 5_000
CCP_CONFIDENCE = 0.90
UNCERTAINTY_PROFILE = "transport_border_saa_v1"
I_PROFILE = "joint_balanced"
M_MINIMUM_GENERATION = 100
M_OPERATOR = "replace"
M_PROBABILITY = 1.0
M_SELECTOR = "joint_balanced"
FINAL_INSTANCE_MASTER_SEED = 71_000_000
FINAL_INSTANCE_CCP_SEED_BASE = 73_000_000
FINAL_INSTANCE_OOS_SEED_BASE = 74_000_000
SHARD_COUNT = len(ALGORITHM_SEEDS)
TASKS_PER_SHARD = FINAL_INSTANCE_COUNT * len(METHODS)
TOTAL_TASK_COUNT = TASKS_PER_SHARD * SHARD_COUNT


def canonical_digest(value) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_schedule(instance_ids):
    instance_ids = list(instance_ids)
    if (len(instance_ids) != FINAL_INSTANCE_COUNT
            or len(set(instance_ids)) != FINAL_INSTANCE_COUNT):
        raise ValueError(
            f"formal CCP I/M test requires {FINAL_INSTANCE_COUNT} "
            "unique instances")
    tasks = []
    for repeat, seed in enumerate(ALGORITHM_SEEDS, 1):
        for instance_id in instance_ids:
            for method in METHODS:
                tasks.append({
                    "task_id": (
                        f"{instance_id}__a{repeat:02d}__{method.lower()}"),
                    "instance_id": instance_id,
                    "algorithm_repeat": repeat,
                    "algorithm_seed": seed,
                    "method": method,
                    "execution_shard": repeat - 1,
                })
    tasks.sort(key=lambda row: row["task_id"])
    return tasks


def expected_rule_i_attempts(population=POPULATION):
    """Mirror the Base initializer partition without importing runtime code."""
    feasibility = int(round(population * 0.30))
    non_seed = max(0, population - feasibility)
    each_greedy = max(1, non_seed // 6) if non_seed else 0
    return population - feasibility - 3 * each_greedy
