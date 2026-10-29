"""PPO controller implementing the frozen NSGA-II mutation logger hook."""
from __future__ import annotations

import random
import time

import baseline_uncertainty as base
from multi_instance_logging import MutationLogger, fingerprint

from .archive import RewardArchive, RewardConfig, hv_reward
from .observations import build_observation
from .v2_controls import private_seed, validate_model_choice_probability


class PPOController(MutationLogger):
    """Collect action rewards without changing environmental selection."""

    def __init__(self, *args, trainer, archive: RewardArchive,
                 reward_config: RewardConfig, batches, generations, pop_size,
                 graph_observation=None, model_choice_probability=1.0,
                 **kwargs):
        self.trainer = trainer
        self._ppo_updates_at_start = int(trainer.updates)
        self.archive = archive
        self.reward_config = reward_config
        self.batches_for_observation = batches
        self.generations = int(generations)
        self.pop_size = int(pop_size)
        self.graph_observation = graph_observation
        self.current_population = []
        self.pending_offspring = []
        self._pending_fingerprints = set()
        self.current_individual = None
        self._suppress_archive = False
        self._inside_mutation = False
        self.reward_total = 0.0
        self.positive_reward_actions = 0
        self.model_control_events = 0
        self.base_control_events = 0
        self.model_choice_probability = validate_model_choice_probability(
            model_choice_probability)
        self.control_rng = random.Random(private_seed(
            "M-mixture-v2",
            kwargs.get("run_id", args[1] if len(args) > 1 else ""),
            (kwargs.get("seeds") or {}).get("algorithm"),
        ))
        super().__init__(
            *args, policy=trainer, evaluate_before_mutation=False,
            preserve_base_position_rng=True, normalizer=archive.normalizer, **kwargs)
        if self.model_choice_probability < 1.0:
            self.decision_counts["base_mixture_choice"] = 0
        trainer.observation_provider = self._observation

    def evaluate(self, function, ind, *args, **kwargs):
        previous = self.evaluations
        result = super().evaluate(function, ind, *args, **kwargs)
        if not self._suppress_archive:
            self.archive.observe(ind.objectives, bool(ind.feasible), fingerprint(ind))
        if self.generation < 0 and self.evaluations > previous and len(self.current_population) < self.pop_size:
            self.current_population.append(ind)
        elif (self.generation >= 0 and self.phase == "offspring"
              and self.evaluations > previous and not self._inside_mutation):
            self._record_pending_offspring(ind)
        return result

    def _record_pending_offspring(self, ind):
        key = fingerprint(ind)
        if key not in self._pending_fingerprints:
            self._pending_fingerprints.add(key)
            self.pending_offspring.append(ind)

    def _observation(self, candidates, context):
        if self.current_individual is None:
            raise RuntimeError("policy requested an observation outside a mutation decision")
        enriched = dict(context)
        enriched["offspring_fraction"] = (
            len(self.pending_offspring) / max(1, self.pop_size))
        # The decision object is a newly crossed child, so any rank/crowding
        # copied from a mating parent is stale and must not enter the state.
        enriched["parent_rank"] = -1
        enriched["parent_crowding"] = 0.0
        return build_observation(
            self.current_individual, self.batches_for_observation, candidates, enriched,
            self.current_population, self.pending_offspring,
            self.archive.summary(), self.archive.normalizer, self.budget,
            self.evaluations, self.generations, self.graph_observation,
            getattr(getattr(self.trainer, "config", None),
                    "candidate_representation", "legacy"))

    def mutate(self, ind, batches, path_lib, tt_dict, arc_lookup,
               arcs, waiting_cost, waiting_emission, reliable_options=None, **kwargs):
        # Preserve the original crossover -> mutation -> repair -> evaluation
        # order.  In particular, do not evaluate the crossed child here.
        use_model = (
            True if self.model_choice_probability == 1.0 else
            False if self.model_choice_probability == 0.0 else
            self.control_rng.random() < self.model_choice_probability
        )
        if not use_model:
            self.base_control_events += 1
            before = fingerprint(ind)
            self._inside_mutation = True
            # Re-enter Base's exact implementation without recursively routing
            # it back to this controller.  Its random batch, operator, target,
            # repair, and evaluation stream are therefore unchanged.
            self.controls_mutation = False
            try:
                result = base.mutate_fixed(
                    ind, batches, path_lib, tt_dict, arc_lookup, arcs,
                    waiting_cost, waiting_emission,
                    reliable_options=reliable_options, **kwargs)
            finally:
                self.controls_mutation = True
                self._inside_mutation = False
            changed = fingerprint(ind) != before
            self.total_attempts += 1
            self.total_effective += int(changed)
            self.decision_counts["base_mixture_choice"] += 1
            self.append("mutation_outcomes", {
                "run_id": self.run_id,
                "instance_id": self.instance.get("instance_id"),
                "generation": self.generation,
                "evaluation": self.evaluations,
                "decision_kind": "base_mixture_choice",
                "control_branch": "paired_base",
                "model_choice_probability": self.model_choice_probability,
                "operator": result[0],
                "decision_changed": bool(changed),
                "reward_recorded_for_policy": False,
            })
            if self.phase == "offspring":
                self._record_pending_offspring(ind)
            return result

        self.model_control_events += 1
        self._inside_mutation = True
        pending_before = len(self.pending)
        self.current_individual = ind
        self._suppress_archive = True
        started = time.perf_counter()
        try:
            result = super().mutate(
                ind, batches, path_lib, tt_dict, arc_lookup, arcs,
                waiting_cost, waiting_emission, reliable_options=reliable_options, **kwargs)
        finally:
            self._suppress_archive = False
            self.current_individual = None
            self._inside_mutation = False
        if len(self.pending) == pending_before:
            # no eligible candidate: no policy action and no PPO update
            if self.phase == "offspring":
                self._record_pending_offspring(ind)
            return result
        row = self.pending[-1][1]
        update = self.archive.observe_many([
            (ind.objectives, bool(ind.feasible), fingerprint(ind)),
        ])
        delta_hv = update.delta_hv
        reward = hv_reward(delta_hv, self.reward_config)
        row.update(reward=float(reward), reward_hv_delta=float(delta_hv),
                   reward_archive_hv_before=update.hv_before,
                   reward_archive_hv_after=update.hv_after,
                   reward_archive_size=update.archive_size,
                   reward_recorded_for_policy=(
                       row.get("decision_kind") == "model_choice"),
                   action_wall_seconds=time.perf_counter() - started)
        if row.get("decision_kind") == "model_choice":
            self.reward_total += reward
            self.positive_reward_actions += int(reward > 0)
        self.trainer.observe_outcome(reward, done=False)
        if self.phase == "offspring":
            self._record_pending_offspring(ind)
        return result

    def flush_events(self, population):
        # This hook is called immediately after P+Q environmental selection and
        # after boost replacement.  It is the exact point at which pending Q is
        # folded into the next population state.
        super().flush_events(population)
        self.current_population = list(population)
        self.pending_offspring.clear()
        self._pending_fingerprints.clear()

    def log_generation(self, population, boost):
        self.current_population = list(population)
        super().log_generation(population, boost)
        self.log_search_progress(population, "environmental_selection")

    def finish_episode(self):
        self.trainer.finish()
        self.log_policy_updates()

    def summary(self):
        return {"reward_total": self.reward_total,
                "positive_reward_actions": self.positive_reward_actions,
                "reward_archive_hv": self.archive.hypervolume,
                "reward_archive_size": len(self.archive.points),
                "decision_counts": dict(self.decision_counts),
                "ppo_updates": self.trainer.updates - self._ppo_updates_at_start,
                "ppo_updates_before_run": self._ppo_updates_at_start,
                "ppo_updates_total": self.trainer.updates,
                "model_choice_probability": self.model_choice_probability,
                "model_control_events": self.model_control_events,
                "base_control_events": self.base_control_events,
                "feature_construction_seconds": self.feature_construction_seconds,
                "ppo_update_seconds": self.ppo_update_seconds}
