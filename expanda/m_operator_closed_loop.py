"""Frozen offline path policy in the existing shared mutation executor.

No gradients, audits, candidate lookahead, new labels or policy updates. The
saved ensemble/threshold is applied exactly; unsupported Balanced states abstain.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import time

import numpy as np
import torch

from m_operator_offline import (
    BALANCED, choose, digest, load_panel, preaction_features, read_json, transform, write_json,
)
from m_operator_path_representation import (
    MatchedPathTransformer, build_parent_inputs, pack_paths, parent_arrays,
)
from multi_instance_logging import fingerprint
from multi_instance_policy import Decision
from transformer_ppo.comparison_controllers import BaseBudgetController
from transformer_ppo.config import PPOConfig
from transformer_ppo.controller import PPOController


def export_frozen_policy(path_root, collection, destination):
    """Choose an existing fold using INNER validation only, before search.

    Rank frozen policies by validation gain/train-target-RMS, then fewer
    switches and lexical fold ID. Do not change checkpoints or thresholds.
    S0-021/022 are absent from every source training/validation fold.
    """
    plan = read_json(path_root / "PLAN.json")
    if not (path_root / "REPRESENTATION_COMPLETE").exists():
        raise ValueError("path representation experiment must be complete")
    if plan["protocol"] != "operator-path-representation-v1":
        raise ValueError("unexpected checkpoint protocol")
    admission = path_root / "QUICK_RESULT.json"
    if not admission.exists() or read_json(admission).get("representation_screen_pass") is not True:
        raise ValueError("offline representation screen did not pass; closed-loop deployment is blocked")
    rows, _, mask, y, _, _ = load_panel(collection)
    dataset = collection / "batch_2/DATASET.json"
    if plan["inputs_sha256"].get(str(dataset.resolve())) != digest(dataset):
        raise ValueError("checkpoint/source dataset mismatch")
    names = np.array([r["instance"] for r in rows])
    rankings = []
    sources = {str((path_root / "PLAN.json").resolve()): digest(path_root / "PLAN.json"),
               str(dataset.resolve()): digest(dataset)}
    for fold in plan["folds"]:
        folder = path_root / fold["test"]
        f = read_json(folder / "FOLD_COMPLETE.json")
        sources[str((folder / "FOLD_COMPLETE.json").resolve())] = digest(folder / "FOLD_COMPLETE.json")
        pred = []
        checkpoints = []
        for seed in plan["seeds"]:
            meta = folder / "PATH_TOKEN" / f"seed_{seed}.json"
            checkpoint = meta.with_suffix(".pt")
            saved = read_json(meta)
            if digest(checkpoint) != saved["checkpoint_sha256"]:
                raise ValueError("frozen path checkpoint changed")
            sources[str(meta.resolve())] = digest(meta)
            sources[str(checkpoint.resolve())] = digest(checkpoint)
            pred.append(np.asarray(saved["predictions"]))
            checkpoints.append(str(checkpoint.resolve()))
        predictions = np.mean(pred, axis=0) - np.std(pred, axis=0)
        predictions[:, BALANCED] = 0.
        validation = np.flatnonzero(names == fold["validation"])
        threshold = f["methods"]["PATH_TOKEN"]["threshold"]
        selected = choose(predictions[validation], mask[validation], threshold)
        gain = float(y[validation, selected].mean())
        scale = f["preprocessing"]["common_candidate_context"]["target_scale"]
        rankings.append({"source_fold": fold["test"], "validation_instance": fold["validation"],
            "training_instances": fold["training"], "frozen_threshold": threshold,
            "validation_net_gain": gain, "validation_scaled_gain": gain / scale,
            "validation_switches": int((selected != BALANCED).sum()), "checkpoints": checkpoints})
    selected = sorted(rankings, key=lambda r: (-r["validation_scaled_gain"], r["validation_switches"], r["source_fold"]))[0]
    if selected["frozen_threshold"] is None or selected["frozen_threshold"] <= 0:
        raise ValueError("deployment requires a positive threshold frozen in inner validation; do not retune on search results")
    sources[str(admission.resolve())] = digest(admission)
    result = {"protocol": "frozen-path-closed-loop-v1", "selected": selected,
        "selection_rule": "max inner-validation net gain / training target scale; fewer switches then lexical fold; no outer/search scores",
        "selection_candidates": rankings, "input_sha256": sources, "seeds": plan["seeds"],
        "new_training_steps": 0, "threshold_changed": False,
        "source_instances": sorted(set(names.tolist()))}
    if destination.exists() and read_json(destination) != result:
        raise ValueError("frozen deployment selection changed; use a new output root")
    write_json(destination, result)
    return result


class FrozenPathPolicy:
    component = "M"
    training = False
    updates = 0
    ppo_update_seconds = 0.
    update_records = ()
    pending = None
    sequence_policy_actions = 0
    inference_action_mode = "argmax"
    policy_seed = None

    def __init__(self, bundle, heuristic=False):
        self.name = "frozen-balanced-M" if heuristic else "frozen-path-transformer-M"
        self.config = PPOConfig(candidate_representation="path_v3", mutation_control="operator",
                                credit_assignment="generation_terminal", dropout=0.)
        self.config.validate()
        self.heuristic, self.models = bool(heuristic), []
        self.sequence_depth = 0
        self.feature_construction_seconds = 0.
        self.observation_provider = None
        self.model_evaluations, self.switches, self.unsupported = 0, 0, 0
        self.action_counts = Counter()
        info = read_json(bundle)
        self.threshold = info["selected"]["frozen_threshold"]
        self.checkpoint_provenance = {"bundle": str(bundle), "sha256": digest(bundle),
            "source_fold": info["selected"]["source_fold"], "threshold": self.threshold,
            "offline_regression_only": True, "training": False}
        if not heuristic:
            for path in info["selected"]["checkpoints"]:
                if digest(Path(path)) != info["input_sha256"][path]:
                    raise ValueError("checkpoint hash mismatch at inference")
                try:
                    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
                except TypeError as exc:
                    if "weights_only" not in str(exc):
                        raise
                    # Older server PyTorch accepts the saved, hash-verified
                    # checkpoint but does not implement weights_only loading.
                    checkpoint = torch.load(path, map_location="cpu")
                if checkpoint["arm"] != "PATH_TOKEN":
                    raise ValueError("wrong representation checkpoint")
                model = MatchedPathTransformer(checkpoint["feature_count"])
                model.load_state_dict(checkpoint["model"], strict=True)
                model.eval()
                for parameter in model.parameters():
                    parameter.requires_grad_(False)
                if self.models and self.preprocessing != checkpoint["preprocessing"]:
                    raise ValueError("ensemble preprocessing differs")
                self.preprocessing = checkpoint["preprocessing"]
                self.feature_count = checkpoint["feature_count"]
                self.models.append(model)

    def predict_observation(self, observation):
        started = time.perf_counter()
        x, eligible = preaction_features(observation)
        if x.shape[1] != self.feature_count:
            raise ValueError("online feature dimensions disagree with offline checkpoint")
        scaler = self.preprocessing["common_candidate_context"]
        xx = torch.as_tensor(transform(x[None], scaler))
        mm = torch.as_tensor(eligible[None])
        parents, groups = parent_arrays([{"observation": observation}])
        inputs = build_parent_inputs(parents, groups, self.preprocessing["shared_parent_columns"], "PATH_TOKEN")
        packed = pack_paths(inputs, [0])
        self.feature_construction_seconds += time.perf_counter() - started
        started = time.perf_counter()
        with torch.inference_mode():
            values = np.array([m(xx, mm, *packed).numpy()[0] for m in self.models], dtype=float)
        values *= scaler["target_scale"]
        conservative = values.mean(0) - values.std(0)
        conservative[BALANCED] = 0.
        if not np.isfinite(conservative).all():
            raise FloatingPointError("non-finite frozen policy inference")
        self.model_evaluations += 1
        return conservative, time.perf_counter() - started

    def choose(self, candidates, context=None, **kwargs):
        legal = np.array([r["eligible"] for r in candidates], dtype=bool)
        if len(legal) != 5:
            raise ValueError("expected the five frozen operator actions")
        scores, inference_seconds = np.zeros(5), 0.
        if not legal[BALANCED]:
            self.unsupported += 1
            action = None  # Identical no-op in both arms; no untrained fallback.
        elif self.heuristic:
            action = BALANCED
        else:
            started = time.perf_counter()
            observation = self.observation_provider(candidates, context or {})
            self.feature_construction_seconds += time.perf_counter() - started
            scores, inference_seconds = self.predict_observation(observation)
            action = int(choose(scores[None], legal[None], self.threshold)[0])
        probabilities = [0.] * 5
        if action is not None:
            probabilities[action] = 1.
            self.action_counts[str(action)] += 1
            self.switches += int(action != BALANCED)
        return Decision(action, probabilities, scores.tolist(), {
            "decision_kind": "frozen_inference", "inference_seconds": inference_seconds,
            "threshold": self.threshold, "switched": action is not None and action != BALANCED})

    def begin_sequence(self):
        self.sequence_depth = 1

    def end_sequence(self, reward, done=False):
        self.sequence_depth = 0

    def finish(self):
        self.sequence_depth = 0


class TimedArchive:
    def __init__(self, archive, owner):
        self.wrapped, self.owner = archive, owner

    def __getattr__(self, name):
        return getattr(self.wrapped, name)

    def observe(self, *args, **kwargs):
        started = time.perf_counter()
        try:
            return self.wrapped.observe(*args, **kwargs)
        finally:
            self.owner.hv_seconds += time.perf_counter() - started


class CompactProfile:
    """Keep only initial fingerprints and aggregate timing, no per-action dumps."""
    def append(self, name, row):
        return None

    def log_initial_population(self, population):
        self.initial_fingerprints = [fingerprint(p) for p in population]
        super().log_initial_population(population)

    def log_search_progress(self, population, checkpoint):
        started = time.perf_counter()
        try:
            return super().log_search_progress(population, checkpoint)
        finally:
            self.hv_seconds += time.perf_counter() - started

    def summary(self):
        return {**super().summary(), "initial_population_fingerprints": self.initial_fingerprints,
                "profile_hv_seconds": self.hv_seconds, "profile_log_mode": "aggregate-only"}


class ProfiledBase(CompactProfile, BaseBudgetController):
    def __init__(self, *args, **kwargs):
        self.hv_seconds, self.initial_fingerprints = 0., []
        super().__init__(*args, **kwargs)

    def _flush_archive(self):
        started = time.perf_counter()
        try:
            return super()._flush_archive()
        finally:
            self.hv_seconds += time.perf_counter() - started


class FrozenOperatorController(CompactProfile, PPOController):
    def __init__(self, *args, **kwargs):
        self.hv_seconds, self.initial_fingerprints = 0., []
        super().__init__(*args, **kwargs)
        if self.candidate_audit_states:
            raise ValueError("counterfactual auditing is disabled in closed-loop inference")
        self.archive = TimedArchive(self.archive, self)
        self.decision_counts["base_mixture_choice"] = 0

    def _next_control_branch(self):
        self.mixture_event_count += 1
        return self.generation >= 10, self.mixture_event_count

    def _operator_candidates(self, *args, **kwargs):
        started = time.perf_counter()
        try:
            return super()._operator_candidates(*args, **kwargs)
        finally:
            self.trainer.feature_construction_seconds += time.perf_counter() - started

    def _population_hypervolume(self, population):
        started = time.perf_counter()
        try:
            return super()._population_hypervolume(population)
        finally:
            self.hv_seconds += time.perf_counter() - started

    def summary(self):
        return {**super().summary(), "minimum_control_generation": 10,
                "closed_loop_model_evaluations": self.trainer.model_evaluations,
                "closed_loop_switches": self.trainer.switches,
                "closed_loop_action_counts": dict(self.trainer.action_counts),
                "closed_loop_unsupported_states": self.trainer.unsupported,
                "execution_contract": "operator_shared_v1"}
