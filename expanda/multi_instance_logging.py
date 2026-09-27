"""Append-only post-V1 CCP100 candidate and attempted-outcome logs."""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import math
import random
import time

import numpy as np

import baseline_uncertainty as base
from learning_mutation import repair_after_mutation
from multi_instance_mutation import candidates_for
from multi_instance_policy import RandomPolicy
from multi_instance_features import feature_record


def fingerprint(ind):
    # Exact float shares, not the rounded export signature, for safe memoisation.
    data = [(key, [(tuple((a.from_node, a.to_node, a.mode) for a in x.path.arcs),
                    float(x.share).hex()) for x in value])
            for key, value in sorted(ind.od_allocations.items())]
    return hashlib.sha256(repr(data).encode()).hexdigest()


def clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    return value


def write_json(path, value):
    path.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _scenario_improvement_rate(before, after):
    """Fraction of paired training scenarios improved by a mutation."""
    old = np.asarray(before, dtype=float)
    new = np.asarray(after, dtype=float)
    if old.shape != new.shape or old.size == 0:
        return None
    finite = np.isfinite(old) & np.isfinite(new)
    if not finite.any():
        return None
    return float(np.mean(new[finite] < old[finite]))


def _violation_flag(breakdown, *names):
    return any(float(breakdown.get(name, 0.0) or 0.0) > 0.0 for name in names)


class MutationLogger:
    def __init__(self, out, run_id, scenario_id, budget=None, policy=None,
                 method="Random-CCP100", instance=None, seeds=None):
        self.out, self.run_id, self.scenario_id, self.budget = out, run_id, scenario_id, budget
        self.policy = policy or RandomPolicy()
        self.method = method
        self.instance = instance or {}
        self.seeds = seeds or {}
        self.evaluations = self.cache_hits = self.event_count = 0
        self.generation, self.phase = -1, "initialisation"
        self.generations_completed = 0
        self.pending = []
        self.start = time.perf_counter()
        self.total_attempts = self.total_effective = self.total_repaired = 0
        self.no_candidate = self.fallbacks = self.inference_seconds = 0
        self._last_generation_time = self.start
        self.files = {name: (out / (name + ".jsonl")).open("x", encoding="utf-8")
                      for name in ("mutation_outcomes", "mutation_candidates",
                                   "mutation_skips", "generation_summary")}

    def append(self, name, row):
        self.files[name].write(json.dumps(clean(row), allow_nan=False) + "\n")
        self.files[name].flush()

    def close(self):
        for handle in self.files.values():
            handle.close()

    def has_budget(self, required):
        return self.budget is None or self.evaluations + required <= self.budget

    def evaluate(self, function, ind, *args, **kwargs):
        key = (self.scenario_id, fingerprint(ind))
        if getattr(ind, "_learning_eval_key", None) == key:
            self.cache_hits += 1
            return
        if not self.has_budget(1):
            raise RuntimeError("CCP100 evaluation budget exhausted")
        function(ind, *args, **kwargs)
        self.evaluations += 1
        ind._learning_eval_key = key
        ind._learning_eval_id = self.evaluations

    def mutate(self, ind, batches, path_lib, tt_dict, arc_lookup,
               arcs, waiting_cost, waiting_emission, reliable_options=None, **kwargs):
        # Evaluate the post-crossover child, NOT either mating parent.
        base.evaluate_individual(ind, batches, arcs, tt_dict, waiting_cost, waiting_emission, **kwargs)
        before = deepcopy(ind)
        before_hash = fingerprint(before)
        before_count = self.evaluations
        op = base.sample_operator()
        candidates = candidates_for(ind, batches, op, path_lib, tt_dict, arc_lookup,
                                    reliable_options)
        context = dict(operator=op, generation=self.generation, phase=self.phase,
            K=self.instance.get("K"),
            deadline_window_alpha=self.instance.get("deadline_window_alpha"),
            feasible_before=bool(before.feasible), violation_before=before.normalized_violation,
            violation_breakdown_before=before.vio_breakdown,
            candidate_count=len(candidates),
            q90_cost_before=before.objectives[0], q90_emission_before=before.objectives[1],
            q90_makespan_before=before.objectives[2])
        decision = self.policy.choose(candidates, context)
        self.inference_seconds += decision.metadata.get("inference_seconds", 0.0)
        self.fallbacks += int(decision.metadata.get("fallback") is not None)
        self.event_count += 1
        event_id = f"{self.run_id}:{self.event_count}"
        for i, candidate in enumerate(candidates):
            baseline_probability = candidate["selection_probability"]
            self.append("mutation_candidates", dict(event_id=event_id, candidate_id=i,
                instance_id=self.instance.get("instance_id"),
                instance_digest=self.instance.get("instance_digest"),
                ccp_scenario_id=self.seeds.get("training"),
                ccp_scenario_digest=self.scenario_id,
                run_id=self.run_id, generation=self.generation,
                evaluation=self.evaluations, seeds=self.seeds,
                K=context["K"], deadline_window_alpha=context["deadline_window_alpha"],
                origin=next(b.origin for b in batches if b.batch_id == candidate["target"].batch_id),
                destination=next(b.destination for b in batches if b.batch_id == candidate["target"].batch_id),
                operator=op, phase=self.phase,
                **asdict(candidate["target"]),
                **{k: v for k, v in candidate.items()
                   if k not in ("target", "selection_probability")},
                pre_mutation_features=feature_record(candidate, context),
                baseline_selection_probability=baseline_probability,
                selection_probability=decision.probabilities[i],
                policy_score=decision.scores[i],
                logging_policy_probability=decision.probabilities[i],
                chosen=i == decision.chosen_index))
        self.total_attempts += 1
        if decision.chosen_index is None:
            self.no_candidate += 1
            self.append("mutation_skips", dict(event_id=event_id, run_id=self.run_id,
                instance_id=self.instance.get("instance_id"), generation=self.generation,
                evaluation=self.evaluations, operator=op, candidate_count=len(candidates),
                reason="no_eligible_candidate", policy=self.policy.name))
            return op, False
        chosen = candidates[decision.chosen_index]
        if self.policy.name == "rule-masked-model-reweighted-selection":
            p_rule = [c["p_rule"] for c in candidates]
            shadow_seed = int(hashlib.sha256(event_id.encode()).hexdigest()[:16], 16)
            shadow_choice = random.Random(shadow_seed).choices(
                range(len(candidates)), weights=p_rule, k=1)[0]
            decision.metadata["shadow_rule_choice"] = shadow_choice
            decision.metadata["different_from_shadow_rule"] = int(
                shadow_choice != decision.chosen_index)
        target = chosen["target"]
        batch = next(b for b in batches if b.batch_id == target.batch_id)
        ok = base.apply_mutation_op(ind, op, batch, path_lib, tt_dict, arc_lookup,
                                    reliable_options=reliable_options, target=target)
        raw_hash = fingerprint(ind)
        repair = repair_after_mutation(ind, before, batches, path_lib, tt_dict, arc_lookup)
        after_hash = fingerprint(ind)
        raw_changed = raw_hash != before_hash
        repair_changed = after_hash != raw_hash
        changed = after_hash != before_hash
        repair_only_change = not raw_changed and changed
        effective_mutation = bool(ok and raw_changed and changed)
        outcome_attributable = not repair_only_change and (bool(ok) or not raw_changed)
        base.evaluate_individual(ind, batches, arcs, tt_dict, waiting_cost, waiting_emission, **kwargs)
        finite = all(math.isfinite(v) for v in (*before.objectives, *ind.objectives))
        row = dict(run_id=self.run_id, scenario_id=self.scenario_id, event_id=event_id,
            instance_id=self.instance.get("instance_id"),
            ccp_scenario_id=self.seeds.get("training"),
            instance_digest=self.instance.get("instance_digest"), seeds=self.seeds,
            K=context["K"], deadline_window_alpha=context["deadline_window_alpha"],
            parent_id=before_hash, child_id=after_hash,
            method=self.method, generation=self.generation, phase=self.phase,
            operator=op, operator_probability=0.2, selection_policy=self.policy.name,
            baseline_selection_probability=chosen["selection_probability"],
            selection_probability=decision.probabilities[decision.chosen_index],
            selected_policy_score=decision.scores[decision.chosen_index],
            policy_metadata=decision.metadata,
            **asdict(target), path_id=chosen["path_id"], candidate_count=len(candidates),
            from_node=chosen.get("from_node"), to_node=chosen.get("to_node"),
            current_mode=chosen.get("current_mode"),
            batch_quantity=chosen.get("batch_quantity"),
            path_count=chosen.get("path_count"), share=chosen.get("share"),
            path_cost_per_teu=chosen.get("path_cost_per_teu"),
            path_emission_per_teu=chosen.get("path_emission_per_teu"),
            path_nominal_time_h=chosen.get("path_nominal_time_h"),
            alternative_mode_count=chosen.get("alternative_mode_count"),
            selected_target_eligible=bool(chosen["eligible"]), mutation_success=bool(ok),
            raw_mutation_changed=raw_changed, repair_changed_decision=repair_changed,
            repair_only_change=repair_only_change, effective_mutation=effective_mutation,
            outcome_attributable_to_selected_target=outcome_attributable,
            decision_changed=changed, decision_before=before_hash,
            decision_raw_mutation=raw_hash, decision_after=after_hash,
            **repair, feasible_before=bool(before.feasible), feasible_after=bool(ind.feasible),
            violation_before=before.normalized_violation, violation_after=ind.normalized_violation,
            violation_breakdown_before=before.vio_breakdown, violation_breakdown_after=ind.vio_breakdown,
            schedule_failure_before=_violation_flag(before.vio_breakdown, "miss_tt"),
            schedule_failure_after=_violation_flag(ind.vio_breakdown, "miss_tt"),
            capacity_infeasible_before=_violation_flag(
                before.vio_breakdown, "cap_excess", "border_cap_excess"),
            capacity_infeasible_after=_violation_flag(
                ind.vio_breakdown, "cap_excess", "border_cap_excess"),
            evaluation_before=before._learning_eval_id, evaluation_after=ind._learning_eval_id,
            extra_after_evaluations=self.evaluations-before_count,
            ccp_evaluation_count=self.evaluations, finite_objective_label=finite,
            objective_label_eligible=(finite and before.feasible and ind.feasible
                                      and outcome_attributable),
            child_dominates_before=bool(base.dominates(ind, before)),
            before_dominates_child=bool(base.dominates(before, ind)))
        deltas = []
        for name, old, new in zip(("cost", "emission", "makespan"), before.objectives, ind.objectives):
            delta = old-new if math.isfinite(old) and math.isfinite(new) else None
            row[f"q90_{name}_before"], row[f"q90_{name}_after"] = old, new
            row[f"delta_{name}"] = delta
            deltas.append(delta)
        row["tradeoff_move"] = finite and any(d > 0 for d in deltas) and any(d < 0 for d in deltas)
        for name, old, new in zip(
                ("cost", "emission", "makespan"),
                (before.cost_s, before.emission_s, before.makespan_s),
                (ind.cost_s, ind.emission_s, ind.makespan_s)):
            row[f"scenario_improvement_rate_{name}"] = _scenario_improvement_rate(old, new)
        self.pending.append((ind, row))
        self.total_effective += int(effective_mutation)
        self.total_repaired += int(repair["repair_action_count"] > 0 or repair["mutation_reverted"])
        return op, bool(ok)

    def flush_events(self, population):
        for ind, row in self.pending:
            survived = (any(ind is p for p in population)
                        if row["phase"] == "offspring" else None)
            row["survived_environmental_selection"] = survived
            row["retained_after_boost"] = (any(ind is p for p in population)
                                           if row["phase"] == "boost" else None)
            # Environmental selection sorts the entire P+Q and assigns rank
            # to rejected offspring too. Read it without changing selection.
            combined_rank = (int(ind.rank) if row["phase"] == "offspring"
                             and hasattr(ind, "rank") else None)
            row["combined_population_rank"] = combined_rank
            row["first_front_member"] = (bool(ind.feasible and combined_rank == 0)
                                          if combined_rank is not None else None)
            row["rank_after_selection"] = (int(ind.rank) if survived else None)
            crowding = getattr(ind, "crowding_distance", None)
            row["crowding_after_selection"] = (float(crowding)
                                                if survived and crowding is not None else None)
            row["nondominated_after_selection"] = bool(
                survived and ind.feasible and ind.rank == 0)
            valid = bool(row["effective_mutation"] and row["feasible_after"]
                         and row["outcome_attributable_to_selected_target"]
                         and row["finite_objective_label"] and row["phase"] == "offspring")
            row["selection_survivor"] = bool(valid and survived) if survived is not None else None
            row["first_front_member_label"] = (
                bool(valid and row["first_front_member"])
                if row["first_front_member"] is not None else None)
            if (valid and row["feasible_before"] and
                    tuple(row[f"q90_{name}_before"] for name in ("cost", "emission", "makespan")) !=
                    tuple(row[f"q90_{name}_after"] for name in ("cost", "emission", "makespan"))):
                row["parent_relation"] = (
                    "dominates" if row["child_dominates_before"] else
                    "dominated" if row["before_dominates_child"] else "incomparable")
            else:
                row["parent_relation"] = "invalid"
            self.append("mutation_outcomes", row)
        self.pending.clear()

    def log_generation(self, population, boost):
        self.generations_completed += 1
        now = time.perf_counter()
        self.append("generation_summary", dict(run_id=self.run_id, generation=self.generation,
            ccp_evaluation_count=self.evaluations, identical_evaluation_reuses=self.cache_hits,
            mutation_attempts_cumulative=self.total_attempts,
            effective_modification_rate_cumulative=self.total_effective/max(1, self.total_attempts),
            repair_rate_cumulative=self.total_repaired/max(1, self.total_attempts),
            feasible_ratio=sum(p.feasible for p in population)/len(population),
            training_front=[list(p.objectives) for p in population if p.feasible and p.rank == 0],
            boost_triggered=bool(boost), runtime_seconds=now-self.start,
            generation_seconds=now-self._last_generation_time,
            inference_seconds_cumulative=self.inference_seconds,
            no_eligible_candidate_cumulative=self.no_candidate,
            fallback_cumulative=self.fallbacks))
        self._last_generation_time = now
