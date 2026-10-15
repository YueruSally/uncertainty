#!/usr/bin/env python3
"""Run one frozen RF-Reward model with deterministic eligible-action argmax."""
import argparse
import json
from pathlib import Path

from transformer_ppo.archive import ObjectiveNormalizer, RewardConfig
from transformer_ppo.episode import run_episode
from transformer_ppo.rf_reward import RFRewardPolicy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path,
                        default=Path(__file__).parent / "data/data_expanded.xlsx")
    parser.add_argument("--instance", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--algorithm-seed", type=int, required=True)
    parser.add_argument("--evaluation-budget", type=int, default=15_000)
    parser.add_argument("--pop", type=int, default=100)
    parser.add_argument("--gens", type=int, default=1_000)
    args = parser.parse_args()
    values = json.loads(args.normalization.read_text(encoding="utf-8"))
    if values.get("source") != "training-only":
        parser.error("normalization must declare source=training-only")
    normalizer = ObjectiveNormalizer(
        tuple(values["minimum"]), tuple(values["maximum"]),
        tuple(values.get("reference", (1.1, 1.1, 1.1))))
    policy = RFRewardPolicy(args.model)
    summary = run_episode(
        args.data, args.instance, args.out, policy, normalizer,
        algorithm_seed=args.algorithm_seed, evaluation_budget=args.evaluation_budget,
        pop_size=args.pop, generations=args.gens, reward_config=RewardConfig())
    print(summary)


if __name__ == "__main__":
    main()
