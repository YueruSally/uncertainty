#!/usr/bin/env python3
"""Analyse one-shot mutation counterfactuals at matched search states.

The panel directory determines the state set.  For every
``INSTANCE/seed_SEED/event_EVENT`` directory, the script combines the original
one-shot run with all panel policy runs, deduplicates identical executed
replacement paths, and compares short-horizon paired HV advantages with the
final paired HV advantage.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path


EPSILON = 1e-12
HORIZONS = (
    "immediate_archive",
    "post_selection_archive",
    "plus_1_generation_archive",
    "plus_3_generations_archive",
    "post_selection_population",
    "plus_1_generation_population",
    "plus_3_generations_population",
)


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mean(values):
    return sum(values) / len(values) if values else None


def _sign(value):
    if value > EPSILON:
        return 1
    if value < -EPSILON:
        return -1
    return 0


def _pearson(xs, ys):
    if len(xs) < 2:
        return None
    x_mean, y_mean = _mean(xs), _mean(ys)
    x_centered = [value - x_mean for value in xs]
    y_centered = [value - y_mean for value in ys]
    denominator = math.sqrt(
        sum(value * value for value in x_centered)
        * sum(value * value for value in y_centered)
    )
    if denominator <= EPSILON:
        return None
    return sum(
        x_value * y_value
        for x_value, y_value in zip(x_centered, y_centered)
    ) / denominator


def _ranks(values):
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while (end < len(order)
               and abs(values[order[end]] - values[order[start]]) <= EPSILON):
            end += 1
        average_rank = 0.5 * (start + end - 1)
        for index in order[start:end]:
            ranks[index] = average_rank
        start = end
    return ranks


def _spearman(xs, ys):
    if len(xs) < 2:
        return None
    return _pearson(_ranks(xs), _ranks(ys))


def _progress(run: Path):
    rows = [
        row for row in _jsonl(run / "search_progress.jsonl")
        if row.get("checkpoint") == "environmental_selection"
    ]
    result = {}
    for row in rows:
        generation = int(row["generation"])
        if generation in result:
            raise ValueError(f"duplicate generation {generation}: {run}")
        result[generation] = row
    return result


def _paired_progress_delta(base_progress, run_progress, generation, field):
    if generation not in base_progress or generation not in run_progress:
        return None
    base = base_progress[generation].get(field)
    value = run_progress[generation].get(field)
    if base is None or value is None:
        return None
    return float(value) - float(base)


def _event_row(run: Path, event: int):
    rows = [
        row for row in _jsonl(run / "mutation_outcomes.jsonl")
        if row.get("mixture_event_index") == event
        and row.get("decision_kind") == "model_choice"
    ]
    if len(rows) != 1:
        raise ValueError(
            f"expected one model-choice row for event {event}, "
            f"found {len(rows)}: {run}")
    return rows[0]


def _record(run, source, instance, seed, event, base_run, base_progress):
    event_row = _event_row(run, event)
    generation = int(event_row["generation"])
    run_progress = _progress(run)
    base_complete = _read(base_run / "COMPLETE.json")
    run_complete = _read(run / "COMPLETE.json")

    record = {
        "instance": instance,
        "seed": seed,
        "event": event,
        "source": source,
        "run": str(run),
        "generation": generation,
        "executed_replacement_path_id": str(
            event_row["executed_replacement_path_id"]),
        "survived_environmental_selection": bool(
            event_row.get("survived_environmental_selection")),
        "immediate_archive": float(event_row.get("reward_hv_delta", 0.0)),
        "final": (
            float(run_complete["final_hypervolume"])
            - float(base_complete["final_hypervolume"])
        ),
        "final_archive_sha256": _sha256(run / "final_archive_lineage.json"),
    }
    for offset, prefix in ((0, "post_selection"),
                           (1, "plus_1_generation"),
                           (3, "plus_3_generations")):
        for field, suffix in (("archive_hv", "archive"),
                              ("population_hv", "population")):
            record[f"{prefix}_{suffix}"] = _paired_progress_delta(
                base_progress, run_progress, generation + offset, field)
    return record


def _metric_summary(records, predictor):
    selected = [
        record for record in records
        if record.get(predictor) is not None
    ]
    xs = [record[predictor] for record in selected]
    ys = [record["final"] for record in selected]
    return {
        "sample_count": len(selected),
        "sign_matches": sum(
            _sign(x_value) == _sign(y_value)
            for x_value, y_value in zip(xs, ys)
        ),
        "sign_accuracy": (
            _mean([
                _sign(x_value) == _sign(y_value)
                for x_value, y_value in zip(xs, ys)
            ]) if selected else None
        ),
        "pearson_with_final_delta": _pearson(xs, ys),
        "spearman_with_final_delta": _spearman(xs, ys),
    }


def _state_ranking(records, predictor):
    usable = [record for record in records if record.get(predictor) is not None]
    if not usable:
        return None
    predictor_max = max(record[predictor] for record in usable)
    final_max = max(record["final"] for record in usable)
    predicted_top = [
        record for record in usable
        if abs(record[predictor] - predictor_max) <= EPSILON
    ]
    final_top_paths = {
        record["executed_replacement_path_id"]
        for record in usable
        if abs(record["final"] - final_max) <= EPSILON
    }
    comparable = 0
    correct = 0
    for left_index, left in enumerate(usable):
        for right in usable[left_index + 1:]:
            predictor_sign = _sign(left[predictor] - right[predictor])
            final_sign = _sign(left["final"] - right["final"])
            if predictor_sign == 0 or final_sign == 0:
                continue
            comparable += 1
            correct += predictor_sign == final_sign
    chosen_final = _mean([record["final"] for record in predicted_top])
    return {
        "candidate_count": len(usable),
        "final_informative": (
            max(record["final"] for record in usable)
            - min(record["final"] for record in usable) > EPSILON
        ),
        "predictor_informative": (
            max(record[predictor] for record in usable)
            - min(record[predictor] for record in usable) > EPSILON
        ),
        "top_1_matches_final": any(
            record["executed_replacement_path_id"] in final_top_paths
            for record in predicted_top
        ),
        "predicted_top_tie_count": len(predicted_top),
        "predicted_top_mean_final_delta": chosen_final,
        "sampled_oracle_final_delta": final_max,
        "final_regret": final_max - chosen_final,
        "comparable_pairs": comparable,
        "correct_pairs": correct,
        "pairwise_accuracy": correct / comparable if comparable else None,
        "spearman": _spearman(
            [record[predictor] for record in usable],
            [record["final"] for record in usable]),
    }


def _ranking_summary(records, predictor):
    by_state = defaultdict(list)
    for record in records:
        by_state[(record["instance"], record["seed"], record["event"])].append(
            record)
    states = []
    for (instance, seed, event), state_records in sorted(by_state.items()):
        metrics = _state_ranking(state_records, predictor)
        if metrics is not None:
            states.append({
                "instance": instance,
                "seed": seed,
                "event": event,
                **metrics,
            })
    informative = [state for state in states if state["final_informative"]]
    comparable_pairs = sum(state["comparable_pairs"] for state in states)
    correct_pairs = sum(state["correct_pairs"] for state in states)
    return {
        "state_count": len(states),
        "final_informative_state_count": len(informative),
        "top_1_matches": sum(
            state["top_1_matches_final"] for state in informative),
        "top_1_accuracy": (
            _mean([state["top_1_matches_final"] for state in informative])
            if informative else None
        ),
        "comparable_pairs": comparable_pairs,
        "correct_pairs": correct_pairs,
        "pairwise_accuracy": (
            correct_pairs / comparable_pairs if comparable_pairs else None
        ),
        "mean_state_spearman": _mean([
            state["spearman"] for state in states
            if state["spearman"] is not None
        ]),
        "mean_final_regret": _mean([
            state["final_regret"] for state in informative
        ]),
        "per_state": states,
    }


def _discover_states(panel_root):
    states = []
    for event_dir in sorted(panel_root.glob("S0_*/seed_*/event_*")):
        try:
            instance = event_dir.parent.parent.name
            seed = int(event_dir.parent.name.removeprefix("seed_"))
            event = int(event_dir.name.removeprefix("event_"))
        except ValueError as error:
            raise ValueError(f"invalid panel state path: {event_dir}") from error
        states.append((instance, seed, event, event_dir))
    if not states:
        raise ValueError(f"no panel states found under {panel_root}")
    return states


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-root", type=Path, required=True)
    parser.add_argument("--original-root", type=Path, required=True)
    parser.add_argument("--panel-root", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    raw_records = []
    for instance, seed, event, panel_state in _discover_states(args.panel_root):
        base_run = args.base_root / instance / f"seed_{seed}" / "P000"
        original_run = (
            args.original_root / instance / f"seed_{seed}" / f"EVENT_{event}")
        if not original_run.is_dir():
            parser.error(f"missing original one-shot run: {original_run}")
        base_progress = _progress(base_run)
        raw_records.append(_record(
            original_run, "original", instance, seed, event,
            base_run, base_progress))
        panel_runs = sorted(panel_state.glob("policy_*"))
        if not panel_runs:
            parser.error(f"no policy runs found: {panel_state}")
        for run in panel_runs:
            raw_records.append(_record(
                run, run.name, instance, seed, event,
                base_run, base_progress))

    grouped = defaultdict(list)
    for record in raw_records:
        key = (
            record["instance"], record["seed"], record["event"],
            record["executed_replacement_path_id"],
        )
        grouped[key].append(record)

    consistency_failures = []
    for key, records in grouped.items():
        first = records[0]
        for other in records[1:]:
            fields = ("final_archive_sha256", "final", *HORIZONS)
            for field in fields:
                left, right = first.get(field), other.get(field)
                same = (
                    left == right if isinstance(left, str)
                    else left is None and right is None
                    or (left is not None and right is not None
                        and abs(left - right) <= EPSILON)
                )
                if not same:
                    consistency_failures.append({
                        "key": key,
                        "field": field,
                        "first": left,
                        "other": right,
                    })

    unique_records = []
    for records in grouped.values():
        record = dict(records[0])
        record["replicate_count"] = len(records)
        record["sources"] = [item["source"] for item in records]
        unique_records.append(record)
    unique_records.sort(key=lambda record: (
        record["instance"], record["seed"], record["event"],
        record["executed_replacement_path_id"]))

    report = {
        "base_root": str(args.base_root),
        "original_root": str(args.original_root),
        "panel_root": str(args.panel_root),
        "state_count": len({
            (record["instance"], record["seed"], record["event"])
            for record in raw_records
        }),
        "raw_run_count": len(raw_records),
        "unique_state_action_count": len(unique_records),
        "duplicate_run_count": len(raw_records) - len(unique_records),
        "duplicate_replay_consistent": not consistency_failures,
        "consistency_failures": consistency_failures,
        "zero_immediate_reward_count": sum(
            abs(record["immediate_archive"]) <= EPSILON
            for record in unique_records),
        "horizon_summary": {
            horizon: _metric_summary(unique_records, horizon)
            for horizon in HORIZONS
        },
        "within_state_ranking": {
            horizon: _ranking_summary(unique_records, horizon)
            for horizon in HORIZONS
        },
        "records": unique_records,
    }
    encoded = json.dumps(report, indent=2, allow_nan=False)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
