"""Dependency-light helpers for the frozen S0 deterministic final test."""
from __future__ import annotations

import hashlib
import json


POLICY_SEEDS = (301, 302, 303)
ALGORITHM_REPEATS = 3
ALGORITHM_SEED_BASE = 93_000_000
ALGORITHM_SEED_STRIDE = 1_000
PRIMARY_METHODS = ("BASE", "I", "C", "M", "ICM")
AUXILIARY_METHODS = ("RULE-I",)
LEARNED_METHODS = ("I", "C", "M", "ICM")
METHOD_WEIGHTS = {
    "BASE": 1.0,
    "RULE-I": 1.0,
    "I": 1.1,
    "M": 1.5,
    "C": 6.5,
    "ICM": 7.5,
}


def canonical_digest(value) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def algorithm_seed(repeat: int, instance_index: int) -> int:
    if not 1 <= repeat <= ALGORITHM_REPEATS:
        raise ValueError("algorithm repeat is outside the frozen range")
    if not 0 <= instance_index < 10:
        raise ValueError("final-test instance index must be in [0, 9]")
    return (
        ALGORITHM_SEED_BASE
        + (repeat - 1) * ALGORITHM_SEED_STRIDE
        + instance_index
    )


def build_schedule(instance_ids, shard_count=2):
    """Build all predeclared runs and greedily balance operational shards."""
    instance_ids = list(instance_ids)
    if len(instance_ids) != 10 or len(set(instance_ids)) != 10:
        raise ValueError("the final test requires ten unique instances")
    if shard_count != 2:
        raise ValueError("the frozen execution plan uses exactly two shards")
    tasks = []
    for repeat in range(1, ALGORITHM_REPEATS + 1):
        for instance_index, instance_id in enumerate(instance_ids):
            seed = algorithm_seed(repeat, instance_index)
            for method in ("BASE", "RULE-I"):
                tasks.append({
                    "task_id": (
                        f"{instance_id}__a{repeat:02d}__{method.lower()}"),
                    "instance_id": instance_id,
                    "algorithm_repeat": repeat,
                    "algorithm_seed": seed,
                    "method": method,
                    "policy_seed": None,
                })
            for policy_seed in POLICY_SEEDS:
                for method in LEARNED_METHODS:
                    tasks.append({
                        "task_id": (
                            f"{instance_id}__a{repeat:02d}__p{policy_seed}__"
                            f"{method.lower()}"),
                        "instance_id": instance_id,
                        "algorithm_repeat": repeat,
                        "algorithm_seed": seed,
                        "method": method,
                        "policy_seed": policy_seed,
                    })

    # Longest-processing-time scheduling is operational only. It neither
    # changes seeds nor contributes to statistical analysis.
    ordered = sorted(
        tasks,
        key=lambda row: (
            -METHOD_WEIGHTS[row["method"]], row["task_id"]),
    )
    shard_loads = [0.0] * shard_count
    for row in ordered:
        shard = min(range(shard_count), key=lambda value: shard_loads[value])
        row["execution_shard"] = shard
        row["estimated_runtime_weight"] = METHOD_WEIGHTS[row["method"]]
        shard_loads[shard] += METHOD_WEIGHTS[row["method"]]
    tasks.sort(key=lambda row: row["task_id"])
    return tasks, shard_loads

