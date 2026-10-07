#!/usr/bin/env python3
"""Train one Transformer-PPO or MLP-PPO seed on frozen CCP100 instances."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random

from multi_instance_logging import write_json
from transformer_ppo.archive import ObjectiveNormalizer, RewardConfig
from transformer_ppo.config import PPOConfig, episode_seeds
from transformer_ppo.episode import run_episode
from transformer_ppo.ppo import PPOTrainer


def _normalizer(path):
    value = json.loads(path.read_text())
    if value.get("source") != "training-only":
        raise ValueError("normalization must explicitly declare source=training-only")
    return ObjectiveNormalizer(tuple(value["minimum"]), tuple(value["maximum"]),
                               tuple(value.get("reference", (1.1, 1.1, 1.1))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path,
                        default=Path(__file__).parent / "data/data_expanded.xlsx")
    parser.add_argument("--instances", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--architecture", choices=("transformer", "mlp"), required=True)
    parser.add_argument("--policy-seed", type=int, required=True)
    parser.add_argument("--evaluation-budget", type=int, required=True,
                        help="Predeclared short-episode CCP100 evaluation budget.")
    parser.add_argument("--pop", type=int, default=100)
    parser.add_argument("--gens", type=int, default=1_000)
    parser.add_argument("--max-instances", type=int,
                        help="Development-only cap; omit for the frozen formal training pass.")
    parser.add_argument("--device")
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        parser.error("output directory must be empty")
    args.out.mkdir(parents=True, exist_ok=True)
    manifests = []
    for path in args.instances.glob("train-S*.json"):
        record = json.loads(path.read_text())
        if record.get("split") == "train":
            manifests.append(path)
    if len(manifests) != 450 and args.max_instances is None:
        parser.error(f"formal training requires exactly 450 manifests, found {len(manifests)}")
    random.Random(args.policy_seed).shuffle(manifests)
    if args.max_instances is not None:
        manifests = manifests[:args.max_instances]
    config = PPOConfig()
    normalizer = _normalizer(args.normalization)
    trainer = PPOTrainer(args.architecture, config, lambda *_: None,
                         device=args.device, policy_seed=args.policy_seed, training=True)
    run_config = {
        "schema_version": 1, "architecture": args.architecture,
        "policy_seed": args.policy_seed, "ppo": asdict(config),
        "ppo_digest": config.digest(), "episode_count": len(manifests),
        "evaluation_budget_per_episode": args.evaluation_budget,
        "normalization_sha256": hashlib.sha256(args.normalization.read_bytes()).hexdigest(),
        "oos_used_for_training": False,
        "checkpoint_rule": "final checkpoint after predeclared instance pass",
    }
    write_json(args.out / "training_configuration.json", run_config)
    summaries = []
    for index, manifest in enumerate(manifests):
        seeds = episode_seeds(index, args.policy_seed)
        episode_out = args.out / "episodes" / manifest.stem
        print(f"EPISODE {index + 1}/{len(manifests)} {manifest.stem}", flush=True)
        summary = run_episode(
            args.data, manifest, episode_out, trainer, normalizer,
            algorithm_seed=seeds["algorithm"], evaluation_budget=args.evaluation_budget,
            pop_size=args.pop, generations=args.gens, reward_config=RewardConfig())
        summaries.append({"instance": manifest.stem, **summary})
    checkpoint = args.out / "final_policy.pt"
    trainer.save(checkpoint, metadata=run_config)
    write_json(args.out / "training_summary.json", {
        "completed": True, "episodes": summaries,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "total_actual_evaluations": sum(x["ccp_evaluation_count"] for x in summaries),
    })


if __name__ == "__main__":
    main()
