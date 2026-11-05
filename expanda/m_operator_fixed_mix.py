"""Predeclared Balanced/mode mixture, shared search execution, no learned model."""
from collections import Counter
import hashlib
import math
import random
import statistics

import numpy as np

from m_operator_closed_loop import FrozenPathPolicy, FrozenOperatorController
from multi_instance_policy import Decision
from transformer_ppo.config import PPOConfig

METHODS = ("HEURISTIC_BALANCED", "BALANCED80_MODE20")
INSTANCES = ("train-S0-021", "train-S0-022")
SEEDS = tuple(range(896001, 896021))
BOOTSTRAP_SEED = 896900


class FixedMixPolicy(FrozenPathPolicy):
    """Reuse only the frozen-policy protocol; never load weights or observations."""

    def __init__(self, instance, seed, mode_probability):
        if not 0 <= mode_probability <= 1:
            raise ValueError("mode probability must be within [0,1]")
        self.name = "fixed-balanced-mode-mixture"
        self.config = PPOConfig(candidate_representation="path_v3", mutation_control="operator",
                                credit_assignment="generation_terminal", dropout=0.)
        self.config.validate()
        self.mode_probability = float(mode_probability)
        namespace = f"fixed-balanced-mode-v1:{instance}:{seed}"
        self.selection_seed = int.from_bytes(hashlib.sha256(namespace.encode()).digest()[:8], "big")
        self.selection_rng = random.Random(self.selection_seed)
        self.checkpoint_provenance = {"kind": "fixed-mixture-no-checkpoint", "learning_enabled": False,
            "mode_probability": self.mode_probability, "selection_seed": self.selection_seed}
        self.sequence_depth = 0
        self.feature_construction_seconds = 0.
        self.observation_provider = None
        self.model_evaluations = self.switches = self.unsupported = 0
        self.requested_mode = self.mode_ineligible_fallbacks = self.decisions = 0
        self.action_counts = Counter()

    def choose(self, candidates, context=None, **kwargs):
        if len(candidates) != 5:
            raise ValueError("requires unchanged five-operator action order")
        legal = [bool(row["eligible"]) for row in candidates]
        # Separate from global search RNG and the executor's target RNG.
        draw = self.selection_rng.random()
        self.decisions += 1
        probabilities = [0.] * 5
        if not legal[4]:
            self.unsupported += 1
            action = None  # Same unsupported-state treatment in both methods.
        else:
            request_mode = draw < self.mode_probability
            self.requested_mode += int(request_mode)
            self.mode_ineligible_fallbacks += int(request_mode and not legal[3])
            action = 3 if request_mode and legal[3] else 4
            probabilities[3] = self.mode_probability if legal[3] else 0.
            probabilities[4] = 1. - probabilities[3]
            self.action_counts[str(action)] += 1
            self.switches += int(action == 3)
        return Decision(action, probabilities, [None] * 5,
            {"decision_kind": "fixed_mixture", "inference_seconds": 0., "switched": action == 3})


class FixedMixController(FrozenOperatorController):
    def __init__(self, *args, **kwargs):
        self.successful_actions = Counter()
        self.effective_actions = Counter()
        super().__init__(*args, **kwargs)

    def append(self, name, row):
        if name == "mutation_outcomes" and row.get("control_branch") == "operator_scheduler":
            if row.get("mutation_success"):
                self.successful_actions[row["operator"]] += 1
                self.effective_actions[row["operator"]] += int(row.get("decision_changed", False))
        # Aggregate statistics only; no repeated candidate/observation dumps.

    def summary(self):
        p = self.trainer
        selected = sum(p.action_counts.values())
        return {**super().summary(), "fixed_mix": {
            "mode_probability": p.mode_probability, "selection_seed": p.selection_seed,
            "post_g10_decisions": p.decisions, "supported_decisions": selected,
            "mode_requests": p.requested_mode, "mode_selected": p.switches,
            "actual_mode_fraction": p.switches / selected if selected else 0.,
            "mode_ineligible_fallbacks": p.mode_ineligible_fallbacks,
            "balanced_unavailable": p.unsupported,
            "successful_actions": dict(self.successful_actions), "effective_actions": dict(self.effective_actions)}}


def validate_pair(rows):
    if len(rows) != 2 or {r["strategy"] for r in rows} != set(METHODS):
        raise ValueError("requires one Balanced and one fixed-mixture run")
    by_method = {r["strategy"]: r for r in rows}
    h, m = [by_method[name] for name in METHODS]
    if h["instance"] != m["instance"] or h["algorithm_seed"] != m["algorithm_seed"]:
        raise ValueError("instance/seed pairing mismatch")
    if h["pairing"] != m["pairing"]:
        raise ValueError("initial population/scenarios/configuration mismatch")
    for r, probability in ((h, 0.), (m, .2)):
        if r["real_objective_evaluation_count"] != 400 or r["evaluation_budget"] != 400:
            raise ValueError("both methods must use exactly 400 objective evaluations")
        if r["ppo_updates"] or r.get("counterfactual_evaluation_count", 0) or r.get("candidate_audit_evaluations", 0):
            raise ValueError("fixed mixture must not train or evaluate alternatives")
        if r["minimum_control_generation"] != 10 or r["fixed_mix"]["mode_probability"] != probability:
            raise ValueError("fixed schedule/probability changed")
        if not math.isfinite(r["final_hypervolume"]) or r["final_hypervolume"] < 0:
            raise ValueError("invalid final HV")
    if h["final_hypervolume"] <= 0:
        raise ValueError("relative improvement undefined for zero Balanced HV")
    if h["fixed_mix"]["selection_seed"] != m["fixed_mix"]["selection_seed"]:
        raise ValueError("mixture random seed mismatch")
    return {"instance": h["instance"], "algorithm_seed": h["algorithm_seed"],
        "heuristic_hv": h["final_hypervolume"], "mixture_hv": m["final_hypervolume"],
        "paired_percent_vs_heuristic": 100 * (m["final_hypervolume"] / h["final_hypervolume"] - 1),
        "heuristic_seconds": h["optimisation_seconds"], "mixture_seconds": m["optimisation_seconds"],
        "mode_selected": m["fixed_mix"]["mode_selected"], "supported_decisions": m["fixed_mix"]["supported_decisions"],
        "mode_ineligible_fallbacks": m["fixed_mix"]["mode_ineligible_fallbacks"]}


def bootstrap(values, indices):
    values = np.asarray(values, dtype=float)
    means = values[indices].mean(axis=1)
    low, high = np.quantile(means, [.025, .975])
    return {"mean": float(values.mean()), "paired_seed_bootstrap_95ci_low": float(low),
            "paired_seed_bootstrap_95ci_high": float(high)}


def summarize(runs):
    groups = {}
    for row in runs:
        groups.setdefault((row["instance"], row["algorithm_seed"]), []).append(row)
    expected = {(instance, seed) for instance in INSTANCES for seed in SEEDS}
    if set(groups) != expected or len(runs) != 80:
        raise ValueError("summary requires all 40 predeclared pairs / 80 runs")
    pairs = [validate_pair(groups[key]) for key in sorted(expected)]
    indices = np.random.default_rng(BOOTSTRAP_SEED).integers(0, 20, size=(10000, 20))
    per_instance, gains = [], []
    for instance in INSTANCES:
        rows = [r for r in pairs if r["instance"] == instance]
        values = [r["paired_percent_vs_heuristic"] for r in rows]
        gains.append(np.asarray(values))
        stats = bootstrap(values, indices)
        effect = ("positive" if stats["paired_seed_bootstrap_95ci_low"] > 0 else
                  "negative" if stats["paired_seed_bootstrap_95ci_high"] < 0 else "inconclusive")
        selected = sum(r["mode_selected"] for r in rows)
        supported = sum(r["supported_decisions"] for r in rows)
        per_instance.append({"instance": instance, "paired_runs": len(rows),
            "mean_heuristic_hv": statistics.fmean(r["heuristic_hv"] for r in rows),
            "mean_mixture_hv": statistics.fmean(r["mixture_hv"] for r in rows),
            "mean_paired_percent_vs_heuristic": stats.pop("mean"), **stats,
            "mean_heuristic_seconds": statistics.fmean(r["heuristic_seconds"] for r in rows),
            "mean_mixture_seconds": statistics.fmean(r["mixture_seconds"] for r in rows),
            "mode_selected": selected, "actual_mode_fraction": selected / supported if supported else 0.,
            "mode_ineligible_fallbacks": sum(r["mode_ineligible_fallbacks"] for r in rows), "effect": effect})
    contrast = bootstrap(gains[1] - gains[0], indices)
    effects = [r["effect"] for r in per_instance]
    if set(effects) == {"positive", "negative"}:
        conclusion = "INSTANCE_SENSITIVE_FIXED_MIX"
        explanation = "固定组合在两个开发实例上呈现相反方向的收益；支持继续研究实例差异，但两个实例不足以建立可泛化的比例策略。"
    elif effects == ["positive", "positive"]:
        conclusion = "FIXED_MIX_BENEFITS_BOTH"
        explanation = "固定组合在两个开发实例上均显示正收益；应把该组合作为后续学习方法需要超过的对照。"
    elif "positive" in effects:
        conclusion = "FIXED_MIX_BENEFIT_ON_ONE_OTHER_UNCERTAIN"
        explanation = "固定组合在一个实例上显示正收益，另一实例证据不足；尚不能确定收益差异是否稳定。"
    elif effects == ["negative", "negative"]:
        conclusion = "FIXED_20_PERCENT_MODE_UNDERPERFORMS"
        explanation = "这一个固定比例在两个实例上均落后；不能据此排除其他比例，也不能断定原模型收益来自状态选择或噪声。"
    else:
        conclusion = "FIXED_MIX_GAIN_NOT_ESTABLISHED"
        explanation = "本轮未确立固定组合的正收益；现有证据不足以区分状态选择效果、比例不合适与随机波动。"
    return {"per_instance": per_instance, "instance_contrast_022_minus_021_percentage_points": contrast,
        "conclusion": conclusion, "explanation": explanation,
        "fixed_mix_benefit_detected": "positive" in effects, "automatic_learning_admission": False,
        "interval_scope": "paired percentile bootstrap over 20 algorithm seeds, conditional on each fixed development instance and its frozen CCP100 scenarios; not an unseen-instance/OOS guarantee",
        "pairs": pairs}
