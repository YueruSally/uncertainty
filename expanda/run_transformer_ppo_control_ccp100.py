#!/usr/bin/env python3
"""Run a frozen Random or Rule control on a new Transformer-PPO manifest."""
import argparse
from pathlib import Path

from run_multi_instance_ccp100 import main as run_control
from transformer_ppo.instance_catalog import read_instance


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path,
                        default=Path(__file__).parent / "data/data_expanded.xlsx")
    parser.add_argument("--instance", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--policy", choices=("random", "rule"), required=True)
    parser.add_argument("--algorithm-seed", type=int, required=True)
    parser.add_argument("--policy-seed", type=int)
    parser.add_argument("--evaluation-budget", type=int, default=15_000)
    parser.add_argument("--pop", type=int, default=100)
    parser.add_argument("--gens", type=int, default=1_000)
    args = parser.parse_args()
    instance, _ = read_instance(args.instance)
    policy_seed = (args.policy_seed if args.policy_seed is not None
                   else args.algorithm_seed + 10_000_000)
    command = [
        "--data", str(args.data), "--instance", str(args.instance),
        "--out", str(args.out), "--policy", args.policy,
        "--algorithm-seed", str(args.algorithm_seed),
        "--policy-seed", str(policy_seed),
        "--training-seed", str(instance["ccp100_seed"]),
        "--path-seed", str(instance["path_seed"]),
        "--evaluation-budget", str(args.evaluation_budget),
        "--pop", str(args.pop), "--gens", str(args.gens),
    ]
    raise SystemExit(run_control(command))
