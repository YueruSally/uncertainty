"""Fixed, training-normalized initialization rule control."""
from __future__ import annotations

import random

import baseline_uncertainty as base

from .comparison_controllers import BaseBudgetController
from .initialization_rule_features import (
    FEATURES,
    path_rule_score,
    validate_rule_normalization,
)


def _path_signature(path):
    return tuple(path.nodes), tuple(path.modes)


class InitializationRuleController(BaseBudgetController):
    """Replace only Base's random initialization path identities by Rule-I."""

    controls_initialization = True
    controls_crossover = False
    controls_mutation = False

    def __init__(self, *args, rule_normalization, **kwargs):
        self.rule_normalization = dict(rule_normalization)
        validate_rule_normalization(self.rule_normalization)
        self.total_attempts = 0
        self.total_effective = 0
        self.no_candidate = 0
        self.rule_choices = 0
        super().__init__(
            *args,
            log_names=("initialization_rule_actions", "generation_summary"),
            **kwargs,
        )

    def initial_individual(self, batches, path_lib, max_paths=3):
        individual = base.Individual()
        event_id = f"{self.run_id}:RULE-I:{self.total_attempts + 1}"
        self.total_attempts += 1
        differs_from_shadow = False
        actions = []
        for batch in batches:
            key = (batch.origin, batch.destination, batch.batch_id)
            paths = list(path_lib.get((batch.origin, batch.destination), ()))
            if not paths:
                self.no_candidate += 1
                continue
            k = random.randint(1, min(int(max_paths), len(paths)))
            shadow_paths = random.sample(paths, k)
            raw_shares = [random.random() for _ in range(k)]
            ranked = sorted(
                paths,
                key=lambda path: (
                    path_rule_score(path, self.rule_normalization),
                    int(path.path_id),
                    _path_signature(path),
                ),
            )
            selected = ranked[:k]
            self.rule_choices += k
            differs_from_shadow = differs_from_shadow or (
                tuple(_path_signature(path) for path in selected)
                != tuple(_path_signature(path) for path in shadow_paths)
            )
            individual.od_allocations[key] = base.merge_and_normalize([
                base.PathAllocation(path=path, share=raw_shares[index])
                for index, path in enumerate(selected)
            ])
            actions.append({
                "batch_id": batch.batch_id,
                "required_path_count": k,
                "chosen_path_ids": [path.path_id for path in selected],
                "chosen_scores": [
                    path_rule_score(path, self.rule_normalization)
                    for path in selected
                ],
            })
        self.total_effective += int(differs_from_shadow)
        self.append("initialization_rule_actions", {
            "event_id": event_id,
            "different_from_shadow_base": bool(differs_from_shadow),
            "actions": actions,
        })
        return individual

    def summary(self):
        result = super().summary()
        result.update({
            "initialization_attempts": self.total_attempts,
            "initializations_different_from_shadow_base": self.total_effective,
            "no_eligible_candidate": self.no_candidate,
            "rule_choices": self.rule_choices,
        })
        return result
