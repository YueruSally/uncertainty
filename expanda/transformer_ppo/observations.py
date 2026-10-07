"""Permutation-aware observations for mutation-location policies."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
import hashlib
import math
from typing import Iterable

import numpy as np

from .config import OPERATOR_TO_ID
from .archive import exact_hv_3d


PARENT_FEATURES = (
    "batch_quantity", "share", "path_cost", "path_emission", "path_time",
    "arc_distance", "arc_cost_rate", "arc_emission_rate", "arc_speed",
    "arc_position", "path_length", "is_arc", "origin_code_0", "origin_code_1",
    "destination_code_0", "destination_code_1", "from_node_code_0",
    "from_node_code_1", "to_node_code_0", "to_node_code_1",
    "mode_code_0", "mode_code_1",
)
CANDIDATE_FEATURES = (
    "batch_quantity", "share", "path_count", "path_cost", "path_emission",
    "path_time", "alternative_modes", "reliable_options", "unallocated_paths",
    "allocation_present", "arc_present", "arc_position", "p_rule",
    "origin_code_0", "origin_code_1", "destination_code_0", "destination_code_1",
    "from_node_code_0", "from_node_code_1", "to_node_code_0", "to_node_code_1",
    "mode_code_0", "mode_code_1",
)
GLOBAL_FEATURES = (
    "K", "deadline_window_alpha", "generation_fraction", "evaluation_fraction",
    "remaining_fraction", "offspring_fraction", "phase_is_boost",
    "parent_feasible", "parent_violation", "parent_rank", "parent_crowding",
    "parent_objective_0", "parent_objective_1", "parent_objective_2",
    "feasible_ratio", "front_size_fraction", "population_hv",
    "objective_min_0", "objective_min_1", "objective_min_2",
    "objective_span_0", "objective_span_1", "objective_span_2",
    "pending_size_fraction", "pending_feasible_ratio", "pending_front_size_fraction",
    "pending_hv", "pending_objective_min_0", "pending_objective_min_1",
    "pending_objective_min_2", "pending_objective_span_0",
    "pending_objective_span_1", "pending_objective_span_2",
    "archive_size_fraction", "archive_hv",
)


def _finite(value, default=0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)


def _category_code(value) -> tuple[float, float]:
    """Stable categorical signature, never a list/order positional encoding."""
    if value is None:
        return 0.0, 0.0
    digest = hashlib.sha256(str(value).encode("utf-8")).digest()
    first = int.from_bytes(digest[:4], "big") / (2**32 - 1)
    second = int.from_bytes(digest[4:8], "big") / (2**32 - 1)
    return 2.0 * first - 1.0, 2.0 * second - 1.0


def parent_tokens(ind, batches) -> list[list[float]]:
    by_id = {batch.batch_id: batch for batch in batches}
    tokens = []
    for key in sorted(ind.od_allocations):
        for alloc in ind.od_allocations[key]:
            batch = by_id.get(key[2])
            quantity = _finite(getattr(batch, "quantity", 0.0))
            path = alloc.path
            arcs = list(path.arcs)
            origin = _category_code(key[0])
            destination = _category_code(key[1])
            # Allocation token has no artificial list position.
            tokens.append([
                quantity, _finite(alloc.share), _finite(path.base_cost_per_teu),
                _finite(path.base_emission_per_teu), _finite(path.base_travel_time_h),
                0.0, 0.0, 0.0, 0.0, 0.0, float(len(arcs)), 0.0,
                *origin, *destination, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ])
            for position, arc in enumerate(arcs):
                denominator = max(1, len(arcs) - 1)
                from_node = _category_code(arc.from_node)
                to_node = _category_code(arc.to_node)
                mode = _category_code(arc.mode)
                tokens.append([
                    quantity, _finite(alloc.share), _finite(path.base_cost_per_teu),
                    _finite(path.base_emission_per_teu), _finite(path.base_travel_time_h),
                    _finite(arc.distance), _finite(arc.cost_per_teu_km),
                    _finite(arc.emission_per_teu_km), _finite(arc.speed_kmh),
                    position / denominator, float(len(arcs)), 1.0,
                    *origin, *destination, *from_node, *to_node, *mode,
                ])
    return tokens or [[0.0] * len(PARENT_FEATURES)]


def candidate_tokens(candidates: Iterable[dict], operator: str, batches=()) -> tuple[list[list[float]], list[bool]]:
    by_id = {batch.batch_id: batch for batch in batches}
    rows, eligible = [], []
    for candidate in candidates:
        target = candidate.get("target", {})
        if is_dataclass(target):
            target = asdict(target)
        arc_index = target.get("arc_index")
        batch = by_id.get(target.get("batch_id"))
        origin = _category_code(getattr(batch, "origin", None))
        destination = _category_code(getattr(batch, "destination", None))
        from_node = _category_code(candidate.get("from_node"))
        to_node = _category_code(candidate.get("to_node"))
        mode = _category_code(candidate.get("current_mode"))
        rows.append([
            _finite(candidate.get("batch_quantity")), _finite(candidate.get("share")),
            _finite(candidate.get("path_count")), _finite(candidate.get("path_cost_per_teu")),
            _finite(candidate.get("path_emission_per_teu")),
            _finite(candidate.get("path_nominal_time_h")),
            _finite(candidate.get("alternative_mode_count")),
            _finite(candidate.get("available_reliable_options")),
            _finite(candidate.get("unallocated_path_count")),
            float(target.get("allocation_index") is not None),
            float(arc_index is not None), _finite(arc_index), _finite(candidate.get("p_rule")),
            *origin, *destination, *from_node, *to_node, *mode,
        ])
        eligible.append(bool(candidate.get("eligible")))
    if not rows:
        rows, eligible = [[0.0] * len(CANDIDATE_FEATURES)], [False]
    if operator not in OPERATOR_TO_ID:
        raise ValueError(f"unknown operator: {operator}")
    return rows, eligible


def population_summary(population, normalizer=None) -> dict:
    if not population:
        return {"population_size": 0, "feasible_ratio": 0.0, "front_size": 0,
                "objective_min": [0.0] * 3, "objective_span": [0.0] * 3,
                "hypervolume": 0.0}
    feasible = [ind for ind in population if getattr(ind, "feasible", False)]
    front = [ind for ind in feasible if getattr(ind, "rank", None) == 0]
    source = feasible or population
    values = np.asarray([ind.objectives for ind in source], dtype=float)
    finite_rows = values[np.all(np.isfinite(values), axis=1)]
    if len(finite_rows):
        minimum, maximum = finite_rows.min(axis=0), finite_rows.max(axis=0)
    else:
        minimum = maximum = np.zeros(3, dtype=float)
    hv = 0.0
    if normalizer is not None:
        normalized = []
        for individual in feasible:
            try:
                normalized.append(normalizer.transform(individual.objectives))
            except ValueError:
                pass
        hv = exact_hv_3d(normalized, normalizer.reference)
    return {"population_size": len(population), "feasible_ratio": len(feasible) / len(population),
            "front_size": len(front), "objective_min": np.nan_to_num(minimum).tolist(),
            "objective_span": np.nan_to_num(maximum - minimum).tolist(),
            "hypervolume": hv}


def build_observation(ind, batches, candidates, context, population, pending_offspring,
                      archive_summary, normalizer, evaluation_budget,
                      evaluation_count, generations):
    operator = context["operator"]
    candidate_values, eligible = candidate_tokens(candidates, operator, batches)
    pop = population_summary(population, normalizer)
    pending = population_summary(pending_offspring, normalizer)
    budget = max(1, int(evaluation_budget or 1))
    generation = max(0, int(context.get("generation", 0)))
    mins, spans = pop["objective_min"], pop["objective_span"]
    pending_mins, pending_spans = pending["objective_min"], pending["objective_span"]
    global_values = [
        _finite(context.get("K")), _finite(context.get("deadline_window_alpha")),
        generation / max(1, generations - 1), evaluation_count / budget,
        max(0.0, 1.0 - evaluation_count / budget),
        _finite(context.get("offspring_fraction")), float(context.get("phase") == "boost"),
        float(bool(context.get("feasible_before"))), _finite(context.get("violation_before")),
        _finite(context.get("parent_rank"), -1.0), _finite(context.get("parent_crowding")),
        _finite(context.get("q90_cost_before")), _finite(context.get("q90_emission_before")),
        _finite(context.get("q90_makespan_before")), pop["feasible_ratio"],
        pop["front_size"] / max(1, pop["population_size"]), pop["hypervolume"],
        *[_finite(v) for v in mins], *[_finite(v) for v in spans],
        pending["population_size"] / max(1, len(population)), pending["feasible_ratio"],
        pending["front_size"] / max(1, pending["population_size"]), pending["hypervolume"],
        *[_finite(v) for v in pending_mins], *[_finite(v) for v in pending_spans],
        _finite(archive_summary.get("archive_size")) / max(1, budget),
        _finite(archive_summary.get("archive_hv")),
    ]
    return {
        "parent": parent_tokens(ind, batches),
        "candidates": candidate_values,
        "eligible": eligible,
        "operator_id": OPERATOR_TO_ID[operator],
        "global": global_values,
    }


def pad_observations(observations, device=None):
    """Convert variable token lists to padded tensors; torch is imported lazily."""
    import torch

    batch = len(observations)
    max_parent = max(len(item["parent"]) for item in observations)
    max_candidates = max(len(item["candidates"]) for item in observations)
    parent = torch.zeros(batch, max_parent, len(PARENT_FEATURES), dtype=torch.float32, device=device)
    candidates = torch.zeros(batch, max_candidates, len(CANDIDATE_FEATURES), dtype=torch.float32, device=device)
    parent_mask = torch.ones(batch, max_parent, dtype=torch.bool, device=device)
    candidate_mask = torch.ones(batch, max_candidates, dtype=torch.bool, device=device)
    eligible = torch.zeros(batch, max_candidates, dtype=torch.bool, device=device)
    for row, item in enumerate(observations):
        p, c = len(item["parent"]), len(item["candidates"])
        parent[row, :p] = torch.as_tensor(item["parent"], dtype=torch.float32, device=device)
        candidates[row, :c] = torch.as_tensor(item["candidates"], dtype=torch.float32, device=device)
        parent_mask[row, :p] = False
        candidate_mask[row, :c] = False
        eligible[row, :c] = torch.as_tensor(item["eligible"], dtype=torch.bool, device=device)
    return {
        "parent": parent, "candidates": candidates,
        "parent_padding_mask": parent_mask, "candidate_padding_mask": candidate_mask,
        "eligible_mask": eligible,
        "operator_id": torch.as_tensor([x["operator_id"] for x in observations],
                                       dtype=torch.long, device=device),
        "global": torch.as_tensor([x["global"] for x in observations],
                                   dtype=torch.float32, device=device),
    }
