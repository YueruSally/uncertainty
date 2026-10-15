#!/usr/bin/env python3
"""Run independent same-child/operator multi-position reward diagnostics."""
import argparse
import hashlib
import json
from pathlib import Path

from multi_instance_logging import write_json
from transformer_ppo.archive import ObjectiveNormalizer, RewardConfig
from transformer_ppo.episode import run_episode
from transformer_ppo.position_diagnostic import (
    DiagnosticRuleTrainer, PositionDiagnosticController,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path,
                        default=Path(__file__).parent / "data/data_expanded.xlsx")
    parser.add_argument("--instance", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--algorithm-seed", type=int, required=True)
    parser.add_argument("--policy-seed", type=int, required=True)
    parser.add_argument("--diagnostic-seed", type=int, required=True)
    parser.add_argument("--diagnostic-repeats", type=int, default=3)
    parser.add_argument("--diagnostic-max-events", type=int, default=8)
    parser.add_argument("--diagnostic-max-candidates", type=int, default=8)
    parser.add_argument("--evaluation-budget", type=int, default=300)
    parser.add_argument("--pop", type=int, default=20)
    parser.add_argument("--gens", type=int, default=1_000)
    args = parser.parse_args()
    values = json.loads(args.normalization.read_text(encoding="utf-8"))
    if values.get("source") != "training-only":
        parser.error("normalization must declare source=training-only")
    normalizer = ObjectiveNormalizer(
        tuple(values["minimum"]), tuple(values["maximum"]),
        tuple(values.get("reference", (1.1, 1.1, 1.1))))
    trainer = DiagnosticRuleTrainer(args.policy_seed)
    diagnostic_config = {
        "schema_version": 1,
        "same_pre_mutation_child": True,
        "same_operator": True,
        "common_new_value_seed_within_repeat": True,
        "archive_frozen_within_event": True,
        "diagnostic_evaluations_excluded_from_trajectory_budget": True,
        "diagnostic_seed": args.diagnostic_seed,
        "diagnostic_repeats": args.diagnostic_repeats,
        "diagnostic_max_events": args.diagnostic_max_events,
        "diagnostic_max_candidates": args.diagnostic_max_candidates,
        "normalization_sha256": hashlib.sha256(args.normalization.read_bytes()).hexdigest(),
        "instance_sha256": hashlib.sha256(args.instance.read_bytes()).hexdigest(),
    }
    summary = run_episode(
        args.data, args.instance, args.out, trainer, normalizer,
        algorithm_seed=args.algorithm_seed, evaluation_budget=args.evaluation_budget,
        pop_size=args.pop, generations=args.gens, reward_config=RewardConfig(),
        controller_class=PositionDiagnosticController,
        controller_kwargs={
            "diagnostic_seed": args.diagnostic_seed,
            "diagnostic_repeats": args.diagnostic_repeats,
            "diagnostic_max_events": args.diagnostic_max_events,
            "diagnostic_max_candidates": args.diagnostic_max_candidates,
        }, configuration_extra={"position_diagnostic": diagnostic_config})
    write_json(args.out / "position_diagnostic_configuration.json", {
        **diagnostic_config,
        "diagnostic_evaluation_count": summary["diagnostic_evaluation_count"],
    })
    print(summary)


if __name__ == "__main__":
    main()
