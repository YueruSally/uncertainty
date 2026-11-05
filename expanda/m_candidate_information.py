"""Matched operator information experiment: concrete proposals, no online PPO.

Candidates are generated with the existing legal-target/operator executor. Only
encoding and nominal path attributes are exposed. CCP outcomes are labels only.
"""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import random
import time

import numpy as np
import torch

import baseline_uncertainty as base
from m_operator_fixed_mix import FixedMixPolicy, FixedMixController
from m_operator_closed_loop import FrozenOperatorController
from m_operator_offline import (OPERATORS, BALANCED, TinyOperatorTransformer,
    preaction_features, transform, choose, digest, read_json)
from multi_instance_logging import fingerprint
from multi_instance_policy import Decision
from transformer_ppo.archive import exact_hv_3d

METHODS = ("HEURISTIC_BALANCED", "STATE_ONLY", "STATE_PLUS_CANDIDATE")
TRAIN_INSTANCES = ("train-S0-002", "train-S0-005", "train-S0-006", "train-S0-008")
VALID_INSTANCE = "train-S0-011"
TEST_INSTANCES = ("train-S0-021", "train-S0-022")
DATA_SEEDS = (897001, 897002)
TRAIN_SEEDS = (897101, 897102, 897103)
TEST_SEEDS = tuple(range(897201, 897221))
THRESHOLD_SCALED = .02  # Frozen before data/results; no threshold search.
MAX_STATES = 32


def private_seed(*parts):
    value = json.dumps(parts, separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "big")


@contextmanager
def proposal_randomness(seed):
    """Local construction cannot consume the real search's random stream."""
    py, np_state = random.getstate(), np.random.get_state()
    try:
        random.seed(seed)
        np.random.seed(seed % (2**32 - 1))
        yield
    finally:
        random.setstate(py)
        np.random.set_state(np_state)


METRICS = ("cost", "emission", "nominal_time", "distance", "capacity_per_quantity",
           "path_count", "weighted_arc_count", "road_share", "rail_share", "water_share", "border_share")
SPECIFIC_NAMES = ("batch_position", "quantity_fraction", "deadline_window_relative",
                  "allocation_position", "arc_position", "constructor_success", "encoding_changed") + tuple(
    f"{prefix}_{name}" for prefix in ("before", "after", "delta") for name in METRICS)


def allocation_metrics(allocations, quantity):
    v = np.zeros(len(METRICS))
    v[5] = len(allocations)
    for allocation in allocations:
        p, w = allocation.path, float(allocation.share)
        n = max(1, len(p.arcs))
        v[:5] += w * np.array([p.base_cost_per_teu, p.base_emission_per_teu,
            p.base_travel_time_h, sum(a.distance for a in p.arcs),
            min((a.capacity for a in p.arcs), default=0.) / max(quantity, 1e-8)])
        v[6] += w * len(p.arcs)
        v[7:10] += w * np.array([sum(a.mode == mode for a in p.arcs) / n for mode in ("road", "rail", "water")])
        v[10] += w * sum(bool(a.is_border_arc) for a in p.arcs) / n
    return v


def proposal_features(parent, child, batch, target, batches, ok):
    """No objectives, feasibility, scenario samples, rank or HV may enter here."""
    if batch is None:
        return np.zeros(len(SPECIFIC_NAMES))
    key = (batch.origin, batch.destination, batch.batch_id)
    old = parent.od_allocations.get(key, [])
    before = allocation_metrics(old, batch.quantity)
    after = allocation_metrics(child.od_allocations.get(key, []), batch.quantity)
    # Same pre-action denominator for both before/after. No CCP evaluation.
    population_scale = np.mean([allocation_metrics(parent.od_allocations.get(
        (b.origin, b.destination, b.batch_id), []), b.quantity) for b in batches], axis=0)
    scale = np.maximum(np.abs(population_scale), 1.)
    index = None if target is None else target.allocation_index
    arc_index = None if target is None else target.arc_index
    arc_count = len(old[index].path.arcs) if index is not None and index < len(old) else 0
    prefix = [next(i for i, b in enumerate(batches) if b.batch_id == batch.batch_id) / max(1, len(batches)-1),
        batch.quantity / max(sum(b.quantity for b in batches), 1e-8),
        max(0., batch.LT-batch.ET) / max(1., before[2]),
        -1. if index is None else index / max(1, len(old)-1),
        -1. if arc_index is None else arc_index / max(1, arc_count-1),
        float(ok), float(fingerprint(parent) != fingerprint(child))]
    value = np.concatenate([prefix, before/scale, after/scale, (after-before)/scale])
    if value.shape != (len(SPECIFIC_NAMES),) or not np.isfinite(value).all():
        raise ValueError("invalid pre-evaluation candidate features")
    return value


def build_proposals(controller, parent, batches, path_lib, tt_dict, arc_lookup, reliable_options, candidates):
    """The same proposals serve both information arms and Balanced."""
    result = []
    for action, op in enumerate(OPERATORS):
        child, batch, target, ok, metadata = deepcopy(parent), None, None, False, {}
        seed = private_seed("candidate-info-v1", controller.run_id, controller.seeds["algorithm"],
                            controller.generation, controller._active_mixture_event_index, op)
        if candidates[action]["eligible"]:
            with proposal_randomness(seed):
                # Explicit base-class call prevents recursively selecting an already frozen child.
                ok, batch, target, metadata = FixedMixController._execute_operator_action(
                    controller, child, op, batches, path_lib, tt_dict, arc_lookup, reliable_options,
                    target_rng=random.Random(private_seed(seed, "target")))
        result.append({"operator": op, "child": child, "batch": batch, "target": target,
            "ok": ok, "metadata": metadata, "seed": seed, "fingerprint": fingerprint(child),
            "features": proposal_features(parent, child, batch, target, batches, ok)})
    return result


class CandidatePolicy(FixedMixPolicy):
    def __init__(self, instance, seed, method, bundle=None, collect=False):
        super().__init__(instance, seed, 0.)
        self.method, self.collect = method, collect
        self.models, self.scaler, self.summary_dim = [], None, None
        self.threshold = None
        self.name = "frozen-concrete-candidate-information"
        if method != METHODS[0]:
            info = read_json(bundle)
            if info["protocol"] != "candidate-information-v1" or info["threshold_scaled"] != THRESHOLD_SCALED:
                raise ValueError("incorrect frozen model bundle")
            self.scaler, self.summary_dim = info["scaler"], info["summary_dim"]
            self.threshold = THRESHOLD_SCALED * self.scaler["target_scale"]
            for path in info["checkpoints"][method]:
                if digest(path) != info["checkpoint_sha256"][path]:
                    raise ValueError("checkpoint changed after freezing")
                try:
                    ckpt = torch.load(path, map_location="cpu", weights_only=True)
                except TypeError as exc:
                    if "weights_only" not in str(exc):
                        raise
                    ckpt = torch.load(path, map_location="cpu")
                if ckpt["method"] != method:
                    raise ValueError("wrong information arm")
                model = TinyOperatorTransformer(len(self.scaler["mean"]))
                model.load_state_dict(ckpt["model"], strict=True)
                model.eval()
                for p in model.parameters():
                    p.requires_grad_(False)
                self.models.append(model)
        self.checkpoint_provenance = {"method": method, "bundle_sha256": digest(bundle) if bundle else None,
            "candidate_contract": "concrete-proposal-v1", "training": False}

    def decide(self, summary, specific, mask):
        scores, elapsed = np.zeros(5), 0.
        if not mask[BALANCED]:
            self.unsupported += 1
            action = None
        elif not self.models:
            action = BALANCED
        else:
            started = time.perf_counter()
            x = transform(np.concatenate([summary, specific], axis=1)[None], self.scaler)
            if summary.shape[-1] != self.summary_dim:
                raise ValueError("online summary schema changed")
            if self.method == "STATE_ONLY":
                x[:, :, self.summary_dim:] = 0.
            with torch.no_grad():
                preds = [m(torch.as_tensor(x), torch.as_tensor(mask[None])).numpy()[0] for m in self.models]
            scores = np.mean(preds, axis=0) * self.scaler["target_scale"]
            action = int(choose(scores[None], mask[None], self.threshold)[0])
            elapsed = time.perf_counter() - started
            self.model_evaluations += 1
        probabilities = [0.] * 5
        if action is not None:
            probabilities[action] = 1.
            self.action_counts[str(action)] += 1
            self.switches += int(action != BALANCED)
        return Decision(action, probabilities, scores.tolist(), {"decision_kind": "concrete_candidate_inference",
            "inference_seconds": elapsed, "threshold": self.threshold, "switched": action not in (None, BALANCED)})


class CandidateController(FixedMixController):
    def __init__(self, *args, **kwargs):
        self.feedback_rows, self.feedback_evaluations, self.execution_checks = [], 0, 0
        self.proposal_seconds = 0.
        self.proposals, self.proposal_parent = None, None
        super().__init__(*args, **kwargs)

    def _choose_operator(self, ind, batches, path_lib, tt_dict, arc_lookup, reliable_options):
        started = time.perf_counter()
        candidates = self._operator_candidates(ind, batches, path_lib, tt_dict, arc_lookup, reliable_options)
        context = {"component": "M", "operator": "add", "decision_stage": "operator",
            "generation": self.generation, "phase": self.phase, "K": self.instance.get("K"),
            "deadline_window_alpha": self.instance.get("deadline_window_alpha"),
            "feasible_before": None, "violation_before": None}
        observation = self._observation(candidates, context)
        summary, mask = preaction_features(observation)
        self.proposal_parent = fingerprint(ind)
        self.proposals = build_proposals(self, ind, batches, path_lib, tt_dict, arc_lookup, reliable_options, candidates)
        specific = np.stack([p["features"] for p in self.proposals])
        self.proposal_seconds += time.perf_counter() - started
        decision = self.trainer.decide(summary, specific, mask)
        self.inference_seconds += decision.metadata["inference_seconds"]
        # Freeze inputs before any objective evaluation. Kept separate from labels.
        self.feedback_pending = None
        if self.trainer.collect and mask[BALANCED] and mask[:4].any() and len(self.feedback_rows) < MAX_STATES:
            self.feedback_pending = {"instance": self.run_id, "algorithm_seed": self.seeds["algorithm"],
                "generation": self.generation, "evaluation": self.evaluations,
                "event": self._active_mixture_event_index, "parent_fingerprint": self.proposal_parent,
                "summary": summary.tolist(), "specific": specific.tolist(), "eligible": mask.tolist()}
        return decision, candidates

    def _execute_operator_action(self, ind, operator, batches, path_lib, tt_dict, arc_lookup,
                                 reliable_options, target_rng=None, trace=None):
        if fingerprint(ind) != self.proposal_parent or self.proposals is None:
            raise RuntimeError("candidate parent changed before execution")
        p = self.proposals[OPERATORS.index(operator)]
        ind.od_allocations = deepcopy(p["child"].od_allocations)
        if fingerprint(ind) != p["fingerprint"]:
            raise RuntimeError("selected candidate was not executed exactly")
        self.execution_checks += 1
        return p["ok"], p["batch"], p["target"], {**p["metadata"], "proposal_seed": p["seed"], "proposal_exact": True}

    def _mutate_operator_only(self, ind, batches, path_lib, tt_dict, arc_lookup, arcs,
                              waiting_cost, waiting_emission, reliable_options, mixture_event_index, evaluation_kwargs):
        result = super()._mutate_operator_only(ind, batches, path_lib, tt_dict, arc_lookup, arcs,
            waiting_cost, waiting_emission, reliable_options, mixture_event_index, evaluation_kwargs)
        row = self.feedback_pending
        if row is not None:
            points = [self.archive.normalizer.transform(p.objectives) for p in self.current_population if p.feasible]
            hv = exact_hv_3d(points, self.archive.normalizer.reference)
            q, details = [None]*5, []
            for a, p in enumerate(self.proposals):
                if not row["eligible"][a]:
                    continue
                child = deepcopy(p["child"])
                # Deliberately outside the search budget and archive, counted as training cost.
                base._evaluate_individual_impl(child, batches, arcs, tt_dict, waiting_cost, waiting_emission, **evaluation_kwargs)
                self.feedback_evaluations += 1
                point = [self.archive.normalizer.transform(child.objectives)] if child.feasible else []
                q[a] = exact_hv_3d(points + point, self.archive.normalizer.reference) - hv
                details.append({"operator": p["operator"], "proposal_seed": p["seed"],
                    "target": asdict(p["target"]) if p["target"] is not None else None,
                    "constructor_success": bool(p["ok"]), "fingerprint": p["fingerprint"],
                    "objectives": list(child.objectives), "feasible": bool(child.feasible)})
            row.update(one_step_hv=q, relative_gain=[None if value is None else value-q[BALANCED] for value in q],
                       proposals=details, label_definition="HV(P union child)-HV(P union Balanced child)")
            self.feedback_rows.append(row)
            print(f"[M-CAND-DATA] {self.run_id} states={len(self.feedback_rows)}/{MAX_STATES} extra_evaluations={self.feedback_evaluations}", flush=True)
        return result

    def summary(self):
        return {**FrozenOperatorController.summary(self), "candidate_contract": "concrete-proposal-v1",
            "successful_actions": dict(self.successful_actions), "effective_actions": dict(self.effective_actions),
            "candidate_construction_seconds": self.proposal_seconds,
            "selected_candidate_execution_checks": self.execution_checks,
            "training_feedback_evaluations": self.feedback_evaluations,
            "training_feedback_states": len(self.feedback_rows)}
