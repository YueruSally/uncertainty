"""Random, eligibility-rule, and learned CCP100 mutation-location policies."""
from dataclasses import dataclass, field
import math
from pathlib import Path
import random

import joblib

from learning_features import feature_frame, feature_record
from segment_learning_features import feature_frame as segment_feature_frame
from segment_learning_features import feature_record as segment_feature_record


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
    name = "learning-score-weighted-mixture"

    def __init__(self, model_path, epsilon=.10, rng=None):
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be between zero and one")
        self.model_path = Path(model_path)
        self.epsilon = float(epsilon)
        self.rng = rng or random
        artifact = joblib.load(self.model_path)
        if artifact.get("schema_version") != 1:
            raise ValueError("Unsupported learning model schema")
        if artifact.get("target") != "selection_survivor":
            raise ValueError("Learning policy requires a selection_survivor model")
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
        # Reweight the original hierarchical location probabilities rather than
        # collapsing 90% of the mass onto one argmax.  This keeps the diversity
        # that the Rule baseline preserved in the paired screening runs.
        weighted = [baseline[i] * max(0.0, scores[i]) if i in eligible else 0.0
                    for i in range(len(candidates))]
        weight_mass = sum(weighted)
        if weight_mass <= 0.0:
            eligible_mass = sum(baseline[i] for i in eligible)
            exploit = [baseline[i] / eligible_mass if i in eligible else 0.0
                       for i in range(len(candidates))]
        else:
            exploit = [value / weight_mass for value in weighted]
        marginal = [self.epsilon * baseline[i] + (1.0 - self.epsilon) * exploit[i]
                    for i in range(len(candidates))]
        exploration = self.rng.random() < self.epsilon
        chosen = _sample(baseline if exploration else exploit, self.rng)
        return PolicyDecision(chosen, marginal, scores, exploration,
                              {"eligible_fallback": False, "epsilon": self.epsilon,
                               "best_score": max(eligible_scores),
                               "score_weight_mass": weight_mass})


class SegmentObjectiveMixturePolicy:
    """Use learned objective signals only for mode-segment selection.

    All non-mode operators retain the eligibility Rule policy.  For a mode
    event, one objective head is sampled uniformly, and its improvement
    probability is discounted by predicted capacity-worsening risk.
    """

    name = "learning-segment-objective-mixture"
    objective_heads = ("cost_improved", "emission_improved", "makespan_improved")
    risk_head = "capacity_worsened"

    def __init__(self, model_path, epsilon=.10, rng=None):
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be between zero and one")
        self.model_path = Path(model_path)
        self.epsilon = float(epsilon)
        self.rng = rng or random
        artifact = joblib.load(self.model_path)
        if artifact.get("schema_version") != 2:
            raise ValueError("Segment policy requires model schema 2")
        if artifact.get("deployment_status") != "pilot_only":
            raise ValueError("Segment model must be explicitly promoted for pilot use")
        required = set(self.objective_heads) | {self.risk_head}
        missing = sorted(required - set(artifact.get("pipelines", {})))
        if missing:
            raise ValueError(f"Segment model is missing required heads: {missing}")
        policy = artifact.get("policy", {})
        if (policy.get("guided_operator") != "mode"
                or tuple(policy.get("objective_heads", ())) != self.objective_heads
                or policy.get("risk_head") != self.risk_head):
            raise ValueError("Segment model policy metadata does not match implementation")
        self.pipelines = artifact["pipelines"]
        self.artifact = artifact

    @staticmethod
    def _positive_probability(pipeline, frame):
        probability = pipeline.predict_proba(frame)
        positive_index = list(pipeline.classes_).index(1)
        return probability[:, positive_index].astype(float).tolist()

    def _mixture(self, candidates, baseline, eligible, scores, metadata):
        weighted = [baseline[i] * max(0.0, scores[i]) if i in eligible else 0.0
                    for i in range(len(candidates))]
        mass = sum(weighted)
        if mass <= 0.0:
            eligible_mass = sum(baseline[i] for i in eligible)
            exploit = [baseline[i] / eligible_mass if i in eligible else 0.0
                       for i in range(len(candidates))]
            metadata["zero_score_fallback"] = True
        else:
            exploit = [value / mass for value in weighted]
            metadata["zero_score_fallback"] = False
        marginal = [self.epsilon * baseline[i] + (1.0 - self.epsilon) * exploit[i]
                    for i in range(len(candidates))]
        exploration = self.rng.random() < self.epsilon
        chosen = _sample(baseline if exploration else exploit, self.rng)
        metadata.update(epsilon=self.epsilon, score_weight_mass=mass)
        return PolicyDecision(chosen, marginal, scores, exploration, metadata)

    def choose(self, candidates, context=None):
        context = context or {}
        baseline = _baseline_probabilities(candidates)
        eligible = [i for i, row in enumerate(candidates) if row["eligible"]]
        if not eligible:
            return PolicyDecision(_sample(baseline, self.rng), baseline,
                                  [None] * len(candidates), True,
                                  {"eligible_fallback": True, "epsilon": self.epsilon,
                                   "model_applied": False})
        if context.get("operator") != "mode":
            scores = [1.0 if i in eligible else None for i in range(len(candidates))]
            return self._mixture(candidates, baseline, eligible, scores, {
                "eligible_fallback": False, "model_applied": False,
                "fallback_policy": "rule-eligible"})
        records = [segment_feature_record(candidates[i], context) for i in eligible]
        frame = segment_feature_frame(records)
        objective = self.rng.choice(self.objective_heads)
        benefit = self._positive_probability(self.pipelines[objective], frame)
        risk = self._positive_probability(self.pipelines[self.risk_head], frame)
        eligible_scores = [p * (1.0 - min(1.0, max(0.0, r)))
                           for p, r in zip(benefit, risk)]
        scores = [None] * len(candidates)
        for index, score in zip(eligible, eligible_scores):
            scores[index] = score
        return self._mixture(candidates, baseline, eligible, scores, {
            "eligible_fallback": False, "model_applied": True,
            "guided_operator": "mode", "objective_head": objective,
            "risk_head": self.risk_head, "score_formula": "benefit*(1-capacity_risk)",
            "best_score": max(eligible_scores)})
