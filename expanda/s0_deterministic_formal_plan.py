"""Dependency-light helpers for the frozen S0 deterministic experiment."""
from __future__ import annotations

import hashlib
import json
import random


POLICY_SEEDS = (301, 302, 303)
ALGORITHM_SEED_BASE = 81_000_000
POPULATION = 250
EVALUATION_BUDGET = 30_000
GENERATION_CAP = 250
TRAINING_INSTANCE_COUNT = 10
EXPECTED_MINIMUM_PPO_UPDATES = {
    "I": 50,
    "C": 2_500,
    "M": 100,
}


def canonical_digest(value) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def evenly_spaced_ranks(total: int, count: int) -> list[int]:
    """Select one-based ranks including both endpoints without duplicates."""
    if count < 2 or total < count:
        raise ValueError("rank selection requires 2 <= count <= total")
    result = [1 + round(index * (total - 1) / (count - 1))
              for index in range(count)]
    if len(set(result)) != count:
        raise ValueError("evenly spaced rank selection produced duplicates")
    return result


def ordered_ids_by_seed(instance_ids, policy_seeds=POLICY_SEEDS):
    """Use the same frozen instances but a reproducible order for each seed."""
    result = {}
    for seed in policy_seeds:
        ordered = list(instance_ids)
        random.Random(seed).shuffle(ordered)
        result[str(seed)] = ordered
    return result
