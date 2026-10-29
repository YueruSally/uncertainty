"""PPO controller implementing the frozen NSGA-II mutation logger hook."""
from __future__ import annotations

import random
import time
import math

import baseline_uncertainty as base
from multi_instance_logging import MutationLogger, fingerprint
from multi_instance_mutation import candidates_for

from .archive import RewardArchive, RewardConfig, hv_reward
from .config import OPERATORS
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
        self.mutation_control = getattr(
            getattr(trainer, "config", None), "mutation_control", "location")
        self.control_rng = random.Random(private_seed(
            "M-mixture-v2",
            kwargs.get("run_id", args[1] if len(args) > 1 else ""),
            (kwargs.get("seeds") or {}).get("algorithm"),
        ))
        super().__init__(
            *args, policy=trainer, evaluate_before_mutation=False,
            preserve_base_position_rng=True, normalizer=archive.normalizer,
            log_names=("mutation_outcomes", "mutation_candidates",
                       "mutation_operator_candidates", "mutation_skips",
                       "generation_summary"), **kwargs)
        if self.model_choice_probability < 1.0:
            self.decision_counts["base_mixture_choice"] = 0
        trainer.observation_provider = self._observation

    @staticmethod
    def _mean(rows, name):
        values = []
        for row in rows:
            try:
                value = float(row.get(name, 0.0))
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                values.append(value)
        return sum(values) / len(values) if values else 0.0

    def _operator_candidates(self, ind, batches, path_lib, tt_dict,
                             arc_lookup, reliable_options):
        """Aggregate legal target sets into five operator-level actions."""
        numeric = (
            "batch_quantity", "path_count", "share", "path_length",
            "alternative_mode_count", "available_reliable_options",
            "unallocated_path_count", "path_cost_per_teu",
            "path_emission_per_teu", "path_nominal_time_h",
            "path_total_distance", "path_bottleneck_capacity",
            "alternative_min_cost", "alternative_min_emission",
            "alternative_min_time", "alternative_mean_cost",
            "alternative_mean_emission", "alternative_mean_time",
            "reliable_best_on_time_probability",
            "reliable_mean_on_time_probability",
            "reliable_min_max_lateness_h", "reliable_mean_max_lateness_h",
        )
        result = []
        for operator, prior in zip(OPERATORS, base._FIXED_OP_PROBS):
            targets = candidates_for(
                ind, batches, operator, path_lib, tt_dict, arc_lookup,
                reliable_options)
            legal = [row for row in targets if row["eligible"]]
            source = legal or targets
            candidate = {
                "target": {}, "eligible": bool(legal),
                "selection_probability": float(prior),
                "candidate_operator": operator,
                "operator_stage": True,
                "operator_legal_target_count": len(legal),
                "operator_legal_target_fraction": (
                    len(legal) / max(1, len(targets))),
                "operator_target_count": len(targets),
            }
            candidate.update({name: self._mean(source, name) for name in numeric})
            result.append(candidate)
        return result

    def _choose_operator(self, ind, batches, path_lib, tt_dict, arc_lookup,
                         reliable_options):
        candidates = self._operator_candidates(
            ind, batches, path_lib, tt_dict, arc_lookup, reliable_options)
        context = {
            "component": "M", "operator": "add",
            "decision_stage": "operator", "generation": self.generation,
            "phase": self.phase, "K": self.instance.get("K"),
            "deadline_window_alpha": self.instance.get("deadline_window_alpha"),
            # The crossed child has not been evaluated yet.  Any feasibility
            # fields copied from a mating parent are stale and must not be used.
            "feasible_before": None, "violation_before": None,
        }
        decision = self.trainer.choose(candidates, context)
        self.inference_seconds += float(
            decision.metadata.get("inference_seconds", 0.0))
        event_id = f"{self.run_id}:M-operator:{self.event_count + 1}"
        for index, candidate in enumerate(candidates):
            self.append("mutation_operator_candidates", {
                "event_id": event_id, "candidate_id": index,
                "generation": self.generation, "evaluation": self.evaluations,
                "operator": candidate["candidate_operator"],
                "eligible": candidate["eligible"],
                "operator_target_count": candidate["operator_target_count"],
                "operator_legal_target_count": candidate[
                    "operator_legal_target_count"],
                "operator_legal_target_fraction": candidate[
                    "operator_legal_target_fraction"],
                "baseline_selection_probability": candidate[
                    "selection_probability"],
                "selection_probability": decision.probabilities[index],
                "chosen": index == decision.chosen_index,
                "decision_kind": decision.metadata.get("decision_kind"),
            })
        return decision, candidates

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
        hierarchical = self.mutation_control == "operator_location"
        selected_operator = None
        operator_decision_kind = None
        try:
            if hierarchical:
                self.trainer.begin_sequence()
                operator_decision, operator_candidates = self._choose_operator(
                    ind, batches, path_lib, tt_dict, arc_lookup,
                    reliable_options)
                operator_decision_kind = operator_decision.metadata.get(
                    "decision_kind")
                if operator_decision.chosen_index is None:
                    # Match the two Base stream draws that the delegated
                    # location mutation would otherwise consume.
                    random.choice(batches)
                    base.sample_operator()
                    self.trainer.end_sequence(0.0, done=True)
                    self.no_candidate += 1
                    return "none", False
                selected_operator = operator_candidates[
                    operator_decision.chosen_index]["candidate_operator"]
                if self.trainer.pending is not None:
                    self.trainer.observe_outcome(0.0, done=False)
            result = super().mutate(
                ind, batches, path_lib, tt_dict, arc_lookup, arcs,
                waiting_cost, waiting_emission, reliable_options=reliable_options,
                operator_override=selected_operator, **kwargs)
        except Exception:
            if hierarchical and self.trainer.sequence_depth:
                self.trainer.sequence_depth = 0
                self.trainer.sequence_policy_actions = 0
                self.trainer.pending = None
            raise
        finally:
            self._suppress_archive = False
            self.current_individual = None
            self._inside_mutation = False
        if len(self.pending) == pending_before:
            # no eligible candidate: no policy action and no PPO update
            if hierarchical and self.trainer.sequence_depth:
                self.trainer.end_sequence(0.0, done=True)
            if self.phase == "offspring":
                self._record_pending_offspring(ind)
            return result
        row = self.pending[-1][1]
        update = self.archive.observe_many([
            (ind.objectives, bool(ind.feasible), fingerprint(ind)),
        ])
        delta_hv = update.delta_hv
        reward = hv_reward(delta_hv, self.reward_config)
        policy_actions = (
            self.trainer.sequence_policy_actions if hierarchical else
            int(row.get("decision_kind") == "model_choice"))
        row.update(reward=float(reward), reward_hv_delta=float(delta_hv),
                   reward_archive_hv_before=update.hv_before,
                   reward_archive_hv_after=update.hv_after,
                   reward_archive_size=update.archive_size,
                   reward_recorded_for_policy=bool(policy_actions),
                   mutation_control=self.mutation_control,
                   operator_decision_kind=operator_decision_kind,
                   policy_actions_in_operation=policy_actions,
                   action_wall_seconds=time.perf_counter() - started)
        if policy_actions:
            self.reward_total += reward
            self.positive_reward_actions += int(reward > 0)
        if hierarchical:
            self.trainer.end_sequence(reward, done=True)
        else:
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
                "mutation_control": self.mutation_control,
                "model_choice_probability": self.model_choice_probability,
                "model_control_events": self.model_control_events,
                "base_control_events": self.base_control_events,
                "feature_construction_seconds": self.feature_construction_seconds,
                "ppo_update_seconds": self.ppo_update_seconds}
