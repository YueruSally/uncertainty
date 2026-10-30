#!/usr/bin/env python3
"""Run paired Base/I/C/M/I+C+M under one frozen CCP/SAA scenario set."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from multi_instance_logging import write_json
from run_learning_components_deterministic import _load, _normalizer
from transformer_ppo.archive import RewardConfig
from transformer_ppo.episode import CCP_UNCERTAINTY_PROFILES, run_episode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path,
                        default=Path(__file__).parent / "data/data_expanded.xlsx")
    parser.add_argument("--instance", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--i-checkpoint", type=Path, required=True)
    parser.add_argument("--c-checkpoint", type=Path, required=True)
    parser.add_argument("--m-checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--algorithm-seed", type=int, required=True)
    parser.add_argument("--inference-action-mode", choices=("argmax", "sample"),
                        default="sample")
    parser.add_argument("--inference-policy-seed", type=int, default=0)
    parser.add_argument("--evaluation-budget", type=int, default=15_000)
    parser.add_argument("--scenario-count", type=int, default=200)
    parser.add_argument("--ccp-confidence", type=float, default=0.90)
    parser.add_argument(
        "--uncertainty-profile", choices=tuple(CCP_UNCERTAINTY_PROFILES),
        default="transport_border_saa_v1")
    parser.add_argument("--pop", type=int, default=100)
    parser.add_argument("--gens", type=int, default=1_000)
    parser.add_argument("--device")
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        parser.error(f"output directory must be empty: {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)
    try:
        normalizer = _normalizer(args.normalization)
        checkpoint_paths = {
            "I": args.i_checkpoint,
            "C": args.c_checkpoint,
            "M": args.m_checkpoint,
        }
        summaries = {}
        scenario_digests = {}
        for component in ("BASE", "I", "C", "M", "ICM"):
            if component == "BASE":
                trainer = None
            elif component == "ICM":
                trainer = {
                    name: _load(
                        path, name, args.device, args.inference_action_mode,
                        args.inference_policy_seed + "ICM".index(name) + 1)
                    for name, path in checkpoint_paths.items()
                }
            else:
                trainer = _load(
                    checkpoint_paths[component], component, args.device,
                    args.inference_action_mode,
                    args.inference_policy_seed)
            method_out = args.out / component
            summaries[component] = run_episode(
                args.data, args.instance, method_out, trainer, normalizer,
                algorithm_seed=args.algorithm_seed,
                evaluation_budget=args.evaluation_budget, pop_size=args.pop,
                generations=args.gens, reward_config=RewardConfig(),
                evaluation_mode="ccp", component=component,
                scenario_count=args.scenario_count,
                ccp_confidence=args.ccp_confidence,
                uncertainty_profile=args.uncertainty_profile)
            configuration = json.loads(
                (method_out / "configuration.json").read_text(encoding="utf-8"))
            scenario_digests[component] = configuration["scenario_digest"]
        if len(set(scenario_digests.values())) != 1:
            raise RuntimeError(
                f"paired methods did not share one scenario set: {scenario_digests}")
    except (KeyError, OSError, RuntimeError, TypeError, ValueError,
            json.JSONDecodeError) as error:
        parser.error(str(error))
    result = {
        "completed": True,
        "methods": ["BASE", "I", "C", "M", "ICM"],
        "algorithm_seed": args.algorithm_seed,
        "evaluation_budget": args.evaluation_budget,
        "scenario_count": args.scenario_count,
        "ccp_confidence": args.ccp_confidence,
        "uncertainty_profile": args.uncertainty_profile,
        "scenario_digest": next(iter(scenario_digests.values())),
        "summaries": summaries,
    }
    write_json(args.out / "COMPLETE.json", result)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
