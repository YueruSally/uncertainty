"""Random, eligibility-rule, and learned CCP100 mutation-location policies."""
from dataclasses import dataclass, field
import math
from pathlib import Path
import random

import joblib

from learning_features import feature_frame, feature_record


@dataclass
class PolicyDecision:
    chosen_index: int
    probabilities: list[float]
    scores: list[float | None]
    exploration: bool
    metadata: dict = field(default_factory=dict)


def _baseline_probabilities(candidates):
    values = [float(row["selection_probability"]) for row in candidates]
    total = sum(values)
    if not values or not math.isfinite(total) or total <= 0:
        raise ValueError("Candidate baseline probabilities must have positive finite mass")
    return [value / total for value in values]


def _sample(probabilities, rng):
    return rng.choices(range(len(probabilities)), weights=probabilities, k=1)[0]


class RandomLocationPolicy:
    name = "random"

    def __init__(self, rng=None):
        self.rng = rng or random

    def choose(self, candidates, context=None):
        probabilities = _baseline_probabilities(candidates)
        return PolicyDecision(_sample(probabilities, self.rng), probabilities,
                              [None] * len(candidates), True)


class EligibleRuleLocationPolicy:
    name = "rule-eligible-epsilon-greedy"

    def __init__(self, epsilon=.10, rng=None):
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be between zero and one")
        self.epsilon = float(epsilon)
        self.rng = rng or random

    def choose(self, candidates, context=None):
        baseline = _baseline_probabilities(candidates)
        eligible = [i for i, row in enumerate(candidates) if row["eligible"]]
        if not eligible:
            return PolicyDecision(_sample(baseline, self.rng), baseline,
                                  [None] * len(candidates), True,
                                  {"eligible_fallback": True, "epsilon": self.epsilon})
        mass = sum(baseline[i] for i in eligible)
        exploit = [baseline[i] / mass if i in eligible else 0.0
                   for i in range(len(candidates))]
        marginal = [self.epsilon * baseline[i] + (1.0 - self.epsilon) * exploit[i]
                    for i in range(len(candidates))]
        exploration = self.rng.random() < self.epsilon
        chosen = _sample(baseline if exploration else exploit, self.rng)
        return PolicyDecision(chosen, marginal,
                              [1.0 if i in eligible else None
                               for i in range(len(candidates))], exploration,
                              {"eligible_fallback": False, "epsilon": self.epsilon})


class LearningLocationPolicy:
    name = "learning-epsilon-greedy"

    def __init__(self, model_path, epsilon=.10, rng=None):
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be between zero and one")
        self.model_path = Path(model_path)
        self.epsilon = float(epsilon)
        self.rng = rng or random
        artifact = joblib.load(self.model_path)
        if artifact.get("schema_version") != 1:
            raise ValueError("Unsupported learning model schema")
        self.pipeline = artifact["pipeline"]
        self.artifact = artifact

    def choose(self, candidates, context=None):
        context = context or {}
        baseline = _baseline_probabilities(candidates)
        eligible = [i for i, row in enumerate(candidates) if row["eligible"]]
        if not eligible:
            return PolicyDecision(_sample(baseline, self.rng), baseline,
                                  [None] * len(candidates), True,
                                  {"eligible_fallback": True, "epsilon": self.epsilon})
        records = [feature_record(candidates[i], context) for i in eligible]
        frame = feature_frame(records)
        probability = self.pipeline.predict_proba(frame)
        classes = list(self.pipeline.classes_)
        positive_index = classes.index(1)
        eligible_scores = probability[:, positive_index].astype(float).tolist()
        scores = [None] * len(candidates)
        for index, score in zip(eligible, eligible_scores):
            scores[index] = score
        best = max(eligible_scores)
        winners = [index for index in eligible
                   if math.isclose(scores[index], best, rel_tol=0.0, abs_tol=1e-12)]
        exploit = [1.0 / len(winners) if i in winners else 0.0
                   for i in range(len(candidates))]
        marginal = [self.epsilon * baseline[i] + (1.0 - self.epsilon) * exploit[i]
                    for i in range(len(candidates))]
        exploration = self.rng.random() < self.epsilon
        chosen = _sample(baseline if exploration else exploit, self.rng)
        return PolicyDecision(chosen, marginal, scores, exploration,
                              {"eligible_fallback": False, "epsilon": self.epsilon,
                               "best_score": best, "winner_count": len(winners)})
