"""Append-only CCP100 training logs; no OOS data enter features or labels."""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import math
import time

import baseline_uncertainty as base
from learning_mutation import RandomLocationPolicy, enumerate_targets, repair_after_mutation


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


class MutationLogger:
    def __init__(self, out, run_id, scenario_id, budget=None):
        self.out, self.run_id, self.scenario_id, self.budget = out, run_id, scenario_id, budget
        self.policy = RandomLocationPolicy()
        self.evaluations = self.cache_hits = self.event_count = 0
        self.generation, self.phase = -1, "initialisation"
        self.generations_completed = 0
        self.pending = []
        self.start = time.perf_counter()
        self.total_attempts = self.total_effective = self.total_repaired = 0
        self.files = {name: (out / (name + ".jsonl")).open("x", encoding="utf-8")
                      for name in ("mutation_events", "mutation_candidates", "generation_summary")}

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
        candidates = enumerate_targets(ind, batches, op, path_lib, tt_dict, arc_lookup)
        chosen = self.policy.choose(candidates)
        target = chosen["target"]
        batch = next(b for b in batches if b.batch_id == target.batch_id)
        self.event_count += 1
        event_id = f"{self.run_id}:{self.event_count}"
        for i, row in enumerate(candidates):
            self.append("mutation_candidates", dict(event_id=event_id, candidate_id=i,
                **asdict(row["target"]), **{k: v for k, v in row.items() if k != "target"},
                chosen=row is chosen))
        ok = base.apply_mutation_op(ind, op, batch, path_lib, tt_dict, arc_lookup,
                                    reliable_options=reliable_options, target=target)
        raw_hash = fingerprint(ind)
        repair = repair_after_mutation(ind, before, batches, path_lib, tt_dict, arc_lookup)
        changed = fingerprint(ind) != before_hash
        base.evaluate_individual(ind, batches, arcs, tt_dict, waiting_cost, waiting_emission, **kwargs)
        finite = all(math.isfinite(v) for v in (*before.objectives, *ind.objectives))
        row = dict(run_id=self.run_id, scenario_id=self.scenario_id, event_id=event_id,
            method="Random-CCP100", generation=self.generation, phase=self.phase,
            operator=op, operator_probability=0.2, selection_policy=self.policy.name,
            selection_probability=chosen["selection_probability"], exploration=True,
            **asdict(target), path_id=chosen["path_id"], candidate_count=len(candidates),
            mutation_success=bool(ok), decision_changed=changed,
            decision_before=before_hash, decision_raw_mutation=raw_hash, decision_after=fingerprint(ind),
            **repair, feasible_before=bool(before.feasible), feasible_after=bool(ind.feasible),
            violation_before=before.normalized_violation, violation_after=ind.normalized_violation,
            violation_breakdown_before=before.vio_breakdown, violation_breakdown_after=ind.vio_breakdown,
            evaluation_before=before._learning_eval_id, evaluation_after=ind._learning_eval_id,
            extra_after_evaluations=self.evaluations-before_count,
            ccp_evaluation_count=self.evaluations, finite_objective_label=finite,
            objective_label_eligible=finite and before.feasible and ind.feasible,
            child_dominates_before=bool(base.dominates(ind, before)),
            before_dominates_child=bool(base.dominates(before, ind)))
        deltas = []
        for name, old, new in zip(("cost", "emission", "makespan"), before.objectives, ind.objectives):
            delta = old-new if math.isfinite(old) and math.isfinite(new) else None
            row[f"q90_{name}_before"], row[f"q90_{name}_after"] = old, new
            row[f"delta_{name}"] = delta
            deltas.append(delta)
        row["tradeoff_move"] = finite and any(d > 0 for d in deltas) and any(d < 0 for d in deltas)
        self.pending.append((ind, row))
        self.total_attempts += 1
        self.total_effective += int(changed)
        self.total_repaired += int(repair["repair_action_count"] > 0 or repair["mutation_reverted"])
        return op, bool(ok)

    def flush_events(self, population):
        for ind, row in self.pending:
            row["survived_environmental_selection"] = (any(ind is p for p in population)
                                                       if row["phase"] == "offspring" else None)
            row["retained_after_boost"] = (any(ind is p for p in population)
                                           if row["phase"] == "boost" else None)
            self.append("mutation_events", row)
        self.pending.clear()

    def log_generation(self, population, boost):
        self.generations_completed += 1
        self.append("generation_summary", dict(run_id=self.run_id, generation=self.generation,
            ccp_evaluation_count=self.evaluations, identical_evaluation_reuses=self.cache_hits,
            mutation_attempts_cumulative=self.total_attempts,
            effective_modification_rate_cumulative=self.total_effective/max(1, self.total_attempts),
            repair_rate_cumulative=self.total_repaired/max(1, self.total_attempts),
            feasible_ratio=sum(p.feasible for p in population)/len(population),
            training_front=[list(p.objectives) for p in population if p.feasible and p.rank == 0],
            boost_triggered=bool(boost), runtime_seconds=time.perf_counter()-self.start))
