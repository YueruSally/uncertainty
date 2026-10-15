#!/usr/bin/env python3
"""Collect exact immediate-reward labels under a logged exploratory policy."""
import argparse
import hashlib
import json
from pathlib import Path

from multi_instance_logging import write_json
from transformer_ppo.archive import ObjectiveNormalizer, RewardConfig
from transformer_ppo.episode import run_episode
from transformer_ppo.rf_reward import RFRewardCollectionPolicy


def _normalizer(path):
    values = json.loads(path.read_text(encoding="utf-8"))
    if values.get("source") != "training-only":
        raise ValueError("normalization must declare source=training-only")
    return ObjectiveNormalizer(tuple(values["minimum"]), tuple(values["maximum"]),
                               tuple(values.get("reference", (1.1, 1.1, 1.1))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path,
                        default=Path(__file__).parent / "data/data_expanded.xlsx")
    parser.add_argument("--instance", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--algorithm-seed", type=int, required=True)
    parser.add_argument("--policy-seed", type=int, required=True)
    parser.add_argument("--uniform-mixture", type=float, default=0.5)
    parser.add_argument("--evaluation-budget", type=int, default=5_000)
    parser.add_argument("--pop", type=int, default=100)
    parser.add_argument("--gens", type=int, default=1_000)
    args = parser.parse_args()
    policy = RFRewardCollectionPolicy(
        args.out, args.policy_seed, uniform_mixture=args.uniform_mixture)
    collection_config = {
        "schema_version": 1,
        "policy": "(1-rho)*p_rule+rho*uniform_eligible",
        "uniform_mixture": args.uniform_mixture,
        "algorithm_seed": args.algorithm_seed,
        "policy_seed": args.policy_seed,
        "evaluation_budget": args.evaluation_budget,
        "inverse_propensity_weighting": False,
        "weighting_reason": (
            "fit conditional mean reward directly; retain propensities for overlap and "
            "clipped-weight sensitivity diagnostics rather than imposing high-variance IPS"),
        "normalization_sha256": hashlib.sha256(args.normalization.read_bytes()).hexdigest(),
        "instance_sha256": hashlib.sha256(args.instance.read_bytes()).hexdigest(),
    }
    summary = run_episode(
        args.data, args.instance, args.out, policy, _normalizer(args.normalization),
        algorithm_seed=args.algorithm_seed, evaluation_budget=args.evaluation_budget,
        pop_size=args.pop, generations=args.gens, reward_config=RewardConfig(),
        configuration_extra={"rf_reward_collection": collection_config})
    write_json(args.out / "rf_collection_configuration.json", collection_config)
    print({**summary, "rf_reward_samples": policy.sample_count})


if __name__ == "__main__":
    main()
