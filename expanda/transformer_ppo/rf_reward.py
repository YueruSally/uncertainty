"""Random-forest policies and leakage-safe features for immediate mutation reward."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np

from multi_instance_policy import Decision, rule_probabilities

from .config import OPERATORS
from .observations import CANDIDATE_FEATURES, GLOBAL_FEATURES, PARENT_FEATURES


PARENT_STATISTICS = ("mean", "minimum", "maximum", "std")
CANDIDATE_SET_STATISTICS = ("mean", "std")
FEATURE_NAMES = tuple(
    [f"action_{name}" for name in CANDIDATE_FEATURES]
    + [f"global_{name}" for name in GLOBAL_FEATURES]
    + ["parent_token_count"]
    + [f"parent_{stat}_{name}" for stat in PARENT_STATISTICS for name in PARENT_FEATURES]
    + ["candidate_count", "eligible_candidate_count"]
    + [f"candidate_set_{stat}_{name}"
       for stat in CANDIDATE_SET_STATISTICS for name in CANDIDATE_FEATURES]
    + [f"operator_{operator}" for operator in OPERATORS]
)


def _finite_array(values, width):
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.shape[1] != width:
        raise ValueError(f"expected a two-dimensional array with width {width}")
    return np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)


def feature_values(observation: dict, candidate_index: int) -> list[float]:
    """Flatten only information visible immediately before the action."""
    candidates = _finite_array(observation["candidates"], len(CANDIDATE_FEATURES))
    parents = _finite_array(observation["parent"], len(PARENT_FEATURES))
    eligible = np.asarray(observation["eligible"], dtype=bool)
    if not 0 <= candidate_index < len(candidates) or not eligible[candidate_index]:
        raise ValueError("RF features require an eligible candidate index")
    operator_id = int(observation["operator_id"])
    if not 0 <= operator_id < len(OPERATORS):
        raise ValueError("invalid operator id")
    parent_stats = [parents.mean(axis=0), parents.min(axis=0),
                    parents.max(axis=0), parents.std(axis=0)]
    candidate_stats = [candidates.mean(axis=0), candidates.std(axis=0)]
    operator = [float(i == operator_id) for i in range(len(OPERATORS))]
    result = np.concatenate([
        candidates[candidate_index],
        np.nan_to_num(np.asarray(observation["global"], dtype=float)),
        np.asarray([len(parents)], dtype=float),
        *parent_stats,
        np.asarray([len(candidates), int(eligible.sum())], dtype=float),
        *candidate_stats,
        np.asarray(operator, dtype=float),
    ]).astype(float).tolist()
    if len(result) != len(FEATURE_NAMES) or not np.all(np.isfinite(result)):
        raise ValueError("RF feature vector does not match the frozen schema")
    return result


def feature_record(observation: dict, candidate_index: int) -> dict[str, float]:
    return dict(zip(FEATURE_NAMES, feature_values(observation, candidate_index)))


class RFRewardCollectionPolicy:
    """Rule-supported exploratory logger; it does not fit or update a model."""

    name = "rf-reward-exploration-collector"
    training = False
    updates = 0
    checkpoint_provenance = None

    def __init__(self, out: Path, policy_seed: int, uniform_mixture: float = 0.5):
        if not 0.0 < uniform_mixture <= 1.0:
            raise ValueError("uniform_mixture must be in (0, 1]")
        self.uniform_mixture = float(uniform_mixture)
        self.rng = random.Random(int(policy_seed))
        self.policy_seed = int(policy_seed)
        self.observation_provider = None
        self.pending_observation = None
        self.pending_action = None
        self.sample_count = 0
        self.path = Path(out) / "rf_reward_samples.jsonl"
        self.handle = None

    def choose(self, candidates, context=None):
        p_rule = rule_probabilities(candidates)
        eligible = [i for i, value in enumerate(p_rule) if value > 0.0]
        if not eligible:
            return Decision(None, p_rule, [None] * len(candidates), {
                "no_eligible_candidate": True, "fallback": None,
                "collection_uniform_mixture": self.uniform_mixture,
            })
        observation = self.observation_provider(candidates, context or {})
        uniform = 1.0 / len(eligible)
        probabilities = [
            ((1.0 - self.uniform_mixture) * p_rule[i]
             + self.uniform_mixture * uniform) if i in eligible else 0.0
            for i in range(len(candidates))
        ]
        action = self.rng.choices(
            range(len(probabilities)), weights=probabilities, k=1)[0]
        self.pending_observation = observation
        self.pending_action = action
        return Decision(action, probabilities, [None] * len(candidates), {
            "no_eligible_candidate": False, "fallback": None,
            "collection_policy": "(1-rho)*p_rule+rho*uniform_eligible",
            "collection_uniform_mixture": self.uniform_mixture,
            "logging_propensity": probabilities[action],
        })

    def record_sample(self, outcome: dict):
        if self.pending_observation is None or self.pending_action is None:
            raise RuntimeError("RF outcome arrived without a pending action")
        action = self.pending_action
        values = feature_values(self.pending_observation, action)
        row = {
            "schema_version": 1,
            "event_id": outcome["event_id"],
            "instance_id": outcome["instance_id"],
            "instance_digest": outcome["instance_digest"],
            "operator": outcome["operator"],
            "generation": outcome["generation"],
            "phase": outcome["phase"],
            "candidate_index": action,
            "feature_names": FEATURE_NAMES,
            "feature_values": values,
            "reward": float(outcome["reward"]),
            "reward_hv_delta": float(outcome["reward_hv_delta"]),
            "reward_archive_hv_before": float(outcome["reward_archive_hv_before"]),
            "reward_archive_hv_after": float(outcome["reward_archive_hv_after"]),
            "logging_propensity": float(outcome["selection_probability"]),
            "p_rule": float(values[FEATURE_NAMES.index("action_p_rule")]),
            "mutation_success": bool(outcome["mutation_success"]),
            "repair_success": bool(outcome["repair_success"]),
            "mutation_reverted": bool(outcome["mutation_reverted"]),
            "decision_changed": bool(outcome["decision_changed"]),
            "effective_mutation": bool(outcome["effective_mutation"]),
            "feasible_after": bool(outcome["feasible_after"]),
            "ccp_evaluation_count": int(outcome["ccp_evaluation_count"]),
        }
        if self.handle is None:
            self.handle = self.path.open("x", encoding="utf-8")
        self.handle.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
        self.handle.flush()
        self.sample_count += 1

    def observe_outcome(self, reward, done=False):
        self.pending_observation = None
        self.pending_action = None

    def finish(self):
        if self.pending_observation is not None:
            raise RuntimeError("collection ended with an unresolved action")
        if self.handle is not None and not self.handle.closed:
            self.handle.close()


class RFRewardPolicy:
    """Frozen deterministic argmax policy over the Rule-eligible action set."""

    name = "rf-reward-regressor"
    training = False
    updates = 0

    def __init__(self, model_path: Path, predict_jobs: int | None = None):
        import joblib

        self.model_path = Path(model_path)
        artifact = joblib.load(self.model_path)
        if artifact.get("schema_version") != 1:
            raise ValueError("unsupported RF-Reward artifact schema")
        if tuple(artifact.get("feature_names", ())) != FEATURE_NAMES:
            raise ValueError("RF-Reward feature schema mismatch")
        if artifact.get("target") != "ppo_immediate_mutation_reward":
            raise ValueError("RF artifact has the wrong target")
        self.artifact = artifact
        self.model = artifact["model"]
        if predict_jobs is not None:
            if predict_jobs < 1:
                raise ValueError("predict_jobs must be positive")
            if not hasattr(self.model, "n_jobs"):
                raise ValueError("RF-Reward model does not expose n_jobs")
            self.model.n_jobs = int(predict_jobs)
        self.observation_provider = None
        self.checkpoint_provenance = {
            "model_sha256": hashlib.sha256(self.model_path.read_bytes()).hexdigest(),
            "schema_version": artifact["schema_version"],
            "training_data_digest": artifact.get("training_data_digest"),
            "rf_seed": artifact.get("rf_seed"),
            "predict_jobs": getattr(self.model, "n_jobs", None),
        }

    def choose(self, candidates, context=None):
        eligible = [i for i, row in enumerate(candidates) if row.get("eligible")]
        if not eligible:
            return Decision(None, [0.0] * len(candidates), [None] * len(candidates), {
                "no_eligible_candidate": True, "fallback": None,
            })
        observation = self.observation_provider(candidates, context or {})
        started = time.perf_counter()
        matrix = np.asarray([feature_values(observation, i) for i in eligible], dtype=float)
        predictions = np.asarray(self.model.predict(matrix), dtype=float)
        if len(predictions) != len(eligible) or not np.all(np.isfinite(predictions)):
            raise ValueError("RF-Reward returned invalid predictions")
        p_rule = rule_probabilities(candidates)
        local = max(range(len(eligible)),
                    key=lambda j: (predictions[j], p_rule[eligible[j]], -eligible[j]))
        chosen = eligible[local]
        probabilities = [float(i == chosen) for i in range(len(candidates))]
        scores = [None] * len(candidates)
        for index, value in zip(eligible, predictions):
            scores[index] = float(value)
        return Decision(chosen, probabilities, scores, {
            "no_eligible_candidate": False, "fallback": None,
            "selection": "eligible predicted-reward argmax",
            "predicted_reward": float(predictions[local]),
            "inference_seconds": time.perf_counter() - started,
        })

    def observe_outcome(self, reward, done=False):
        return None

    def finish(self):
        return None
