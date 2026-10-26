#!/usr/bin/env python3
"""Run Base, one frozen component, or frozen I+C+M deterministically."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformer_ppo.archive import ObjectiveNormalizer, RewardConfig
from transformer_ppo.config import PPOConfig
from transformer_ppo.episode import run_episode
from transformer_ppo.ppo import PPOTrainer


def _normalizer(path):
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("source") != "training-only":
        raise ValueError("normalization must declare source=training-only")
    return ObjectiveNormalizer(
        tuple(value["minimum"]), tuple(value["maximum"]),
        tuple(value.get("reference", (1.1, 1.1, 1.1))))


def _load(path, component, device):
    if path is None or not path.is_file():
        raise ValueError(f"{component} checkpoint is missing: {path}")
    trainer = PPOTrainer.load(
        path, PPOConfig, lambda *_: None, device=device, training=False)
    if trainer.component != component:
        raise ValueError(
            f"checkpoint {path} is component {trainer.component}, expected {component}")
    return trainer


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path,
                        default=Path(__file__).parent / "data/data_expanded.xlsx")
    parser.add_argument("--instance", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--component", choices=(
        "BASE", "RULE-I", "I", "C", "M", "ICM"),
                        required=True)
    parser.add_argument("--initialization-rule-normalization", type=Path)
    parser.add_argument("--checkpoint", type=Path,
                        help="Checkpoint for a single I, C, or M run.")
    parser.add_argument("--i-checkpoint", type=Path)
    parser.add_argument("--c-checkpoint", type=Path)
    parser.add_argument("--m-checkpoint", type=Path)
    parser.add_argument("--algorithm-seed", type=int, required=True)
    parser.add_argument("--evaluation-budget", type=int, default=15_000)
    parser.add_argument("--pop", type=int, default=100)
    parser.add_argument("--gens", type=int, default=1_000)
    parser.add_argument("--device")
    args = parser.parse_args(argv)
    try:
        normalizer = _normalizer(args.normalization)
        rule_normalization = None
        if args.component == "RULE-I":
            if (args.initialization_rule_normalization is None
                    or not args.initialization_rule_normalization.is_file()):
                raise ValueError("RULE-I requires --initialization-rule-normalization")
            rule_normalization = json.loads(
                args.initialization_rule_normalization.read_text(encoding="utf-8"))
            trainer = None
        elif args.component == "BASE":
            trainer = None
        elif args.component == "ICM":
            trainer = {
                "I": _load(args.i_checkpoint, "I", args.device),
                "C": _load(args.c_checkpoint, "C", args.device),
                "M": _load(args.m_checkpoint, "M", args.device),
            }
        else:
            trainer = _load(args.checkpoint, args.component, args.device)
        summary = run_episode(
            args.data, args.instance, args.out, trainer, normalizer,
            algorithm_seed=args.algorithm_seed,
            evaluation_budget=args.evaluation_budget, pop_size=args.pop,
            generations=args.gens, reward_config=RewardConfig(),
            evaluation_mode="deterministic", component=args.component,
            rule_initialization_normalization=rule_normalization)
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
