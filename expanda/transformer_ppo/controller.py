"""PPO controller implementing the frozen NSGA-II mutation logger hook."""
from __future__ import annotations

import time

import baseline_uncertainty as base
from multi_instance_logging import MutationLogger, fingerprint

from .archive import RewardArchive, RewardConfig, mutation_reward
from .observations import build_observation


class PPOController(MutationLogger):
    """Collect action rewards without changing environmental selection."""

    def __init__(self, *args, trainer, archive: RewardArchive,
                 reward_config: RewardConfig, batches, generations, pop_size, **kwargs):
        self.trainer = trainer
        self.archive = archive
        self.reward_config = reward_config
        self.batches_for_observation = batches
        self.generations = int(generations)
        self.pop_size = int(pop_size)
        self.current_population = []
        self.pending_offspring = []
        self._pending_fingerprints = set()
        self.current_individual = None
        self._suppress_archive = False
        self._inside_mutation = False
        self.reward_total = 0.0
        self.positive_reward_actions = 0
        super().__init__(*args, policy=trainer, **kwargs)
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
            self.evaluations, self.generations)

    def mutate(self, ind, batches, path_lib, tt_dict, arc_lookup,
               arcs, waiting_cost, waiting_emission, reliable_options=None, **kwargs):
        # Freeze crossover contribution in the archive before the policy acts.
        self._inside_mutation = True
        base.evaluate_individual(ind, batches, arcs, tt_dict, waiting_cost,
                                 waiting_emission, **kwargs)
        hv_before = self.archive.hypervolume
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
        self.archive.observe(ind.objectives, bool(ind.feasible), fingerprint(ind))
        delta_hv = max(0.0, self.archive.hypervolume - hv_before)
        reward = mutation_reward(row, delta_hv, self.reward_config)
        row.update(reward=float(reward), reward_hv_delta=float(delta_hv),
                   reward_archive_hv_before=float(hv_before),
                   reward_archive_hv_after=float(self.archive.hypervolume),
                   reward_archive_size=len(self.archive.points),
                   action_wall_seconds=time.perf_counter() - started)
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

    def finish_episode(self):
        self.trainer.finish()

    def summary(self):
        return {"reward_total": self.reward_total,
                "positive_reward_actions": self.positive_reward_actions,
                "reward_archive_hv": self.archive.hypervolume,
                "reward_archive_size": len(self.archive.points),
                "ppo_updates": self.trainer.updates}
