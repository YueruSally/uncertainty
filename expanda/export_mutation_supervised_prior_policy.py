#!/usr/bin/env python3
"""Export an untrained zero-residual Transformer policy around a fitted prior."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from transformer_ppo.config import PPOConfig
from transformer_ppo.graph_features import GRAPH_MODEL_SPEC_V2
from transformer_ppo.observations import (
    CANDIDATE_PATH_V4_FEATURES, GLOBAL_FEATURES, PARENT_FEATURES,
)
from transformer_ppo.ppo import PPOTrainer


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--policy-seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if not args.prior.is_file():
        parser.error(f"prior does not exist: {args.prior}")
    if args.out.exists():
        parser.error(f"output already exists: {args.out}")
    try:
        prior = _read_json(args.prior)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    config = PPOConfig(
        dropout=0.0,
        actor_policy="base_residual",
        candidate_representation="path_v4",
        credit_assignment="operation_terminal",
        mutation_control="replace_path",
    )
    model_spec = {
        "parent_dim": len(PARENT_FEATURES),
        "candidate_dim": len(CANDIDATE_PATH_V4_FEATURES),
        "global_dim": len(GLOBAL_FEATURES),
        "action_count": 5,
        **GRAPH_MODEL_SPEC_V2,
    }
    try:
        trainer = PPOTrainer(
            "transformer", config, lambda *_: None,
            device=args.device, policy_seed=args.policy_seed,
            training=False, component="M", model_spec=model_spec,
            supervised_prior=prior)
        metadata = {
            "export_kind": "zero-residual-supervised-prior",
            "policy_seed": int(args.policy_seed),
            "episode_count": 0,
            "ppo_digest": config.digest(),
            "learning_control": {
                "protocol": "nsga2-learning-components-v2",
                "model_choice_probability": 1.0,
                "control_random_stream": "private-sha256-seeded",
                "mutation_supervised_prior": {
                    "path": str(args.prior.resolve()),
                    "sha256": _sha256(args.prior),
                },
            },
        }
        trainer.save(args.out, metadata=metadata)
    except (KeyError, OSError, TypeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps({
        "checkpoint": str(args.out),
        "checkpoint_sha256": _sha256(args.out),
        "prior": str(args.prior),
        "prior_sha256": _sha256(args.prior),
        "policy_seed": int(args.policy_seed),
        "ppo_updates": 0,
        "actor_residual": "zero",
    }, indent=2))


if __name__ == "__main__":
    main()
