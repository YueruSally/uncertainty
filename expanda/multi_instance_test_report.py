#!/usr/bin/env python3
"""Five per-instance paired CCP100 test reports with frozen normalisation."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from multi_instance_analysis import REF, hv
from multi_instance_logging import write_json
from postprocess_formal_ev_vs_ccp import exact_hv_3d


def _read(path):
    return json.loads(path.read_text())


def _jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def _coverage(a, b):
    """Fraction of B points weakly dominated by at least one A point."""
    if not b:
        return None
    return sum(any(all(x <= y for x, y in zip(p, q)) for p in a) for q in b)/len(b)


def report(root, bounds_file, lock_file, out):
    if out.exists():
        raise ValueError("test report already exists")
    bounds, lock = _read(bounds_file), _read(lock_file)
    if lock["status"] != "frozen":
        raise ValueError("test requires frozen validation strategy")
    result = dict(schema_version=1, model_target=lock["label"], epsilon=lock["epsilon"],
                  reference_point=list(REF), bounds_digest=bounds["bounds_digest"],
                  instances={}, win_tie_loss={"learning_vs_rule": [0, 0, 0],
                                              "learning_vs_random": [0, 0, 0]})
    for sid in (f"T{i}" for i in range(1, 6)):
        rows = {method: [] for method in ("random", "rule", "learning")}
        for repetition in range(1, 11):
            controls = root / "test_controls" / sid
            paths = {
                "random": controls / "random" / f"run{repetition:02d}",
                "rule": controls / "rule" / f"run{repetition:02d}",
                "learning": root / "test_learning" / sid /
                    f"{lock['label']}_eps{lock['epsilon']:g}" / f"run{repetition:02d}",
            }
            for method, path in paths.items():
                complete = _read(path / "COMPLETE.json")
                config = _read(path / "configuration.json")
                oos = _read(path / "oos_5000" / "oos_5000.json")
                outcomes = [row for row in _jsonl(path / "mutation_outcomes.jsonl")
                            if row["phase"] == "offspring"]
                eligible = [r for r in outcomes if r["selected_target_eligible"]]
                learning_events = [r for r in outcomes if
                    r["selection_policy"] == "rule-masked-model-reweighted-selection"]
                generations = _jsonl(path / "generation_summary.jsonl")
                b = bounds["instances"][sid]["in_sample"]
                lo, hi = np.asarray(b["minimum"]), np.asarray(b["maximum"])
                curve = []
                for gen in generations:
                    front = np.asarray(gen["training_front"], dtype=float).reshape((-1, 3))
                    h = exact_hv_3d((front-lo)/(hi-lo), reference=REF) if len(front) else 0.
                    curve.append(dict(evaluations=gen["ccp_evaluation_count"],
                                      wall_seconds=gen["runtime_seconds"], in_sample_hv=h,
                                      generation_seconds=gen["generation_seconds"]))
                oos_points = [tuple(c[x]["q90"] for x in ("cost", "emission", "makespan"))
                              for c in oos["candidates"]]
                rows[method].append(dict(repetition=repetition, seeds=config["seeds"],
                    instance_digest=config["instance_digest"],
                    ccp_scenario_digest=config["scenario_digest"],
                    oos_scenario_digest=oos["oos_master_digest"],
                    actual_evaluations=complete["ccp_evaluation_count"],
                    stop_reason=complete["stop_reason"],
                    in_sample_hv=hv(path, bounds, sid, "in_sample"),
                    oos_hv=hv(path, bounds, sid, "oos", 5000),
                    effective_modification_rate=complete["effective_modifications"] /
                        max(1, complete["mutation_attempts"]),
                    first_front_entry_rate=sum(r["first_front_member_label"] is True for r in eligible) /
                        max(1, len(eligible)),
                    offspring_survival_rate=sum(r["selection_survivor"] is True for r in eligible) /
                        max(1, len(eligible)),
                    eligible_attempts=len(eligible), mutation_attempts=complete["mutation_attempts"],
                    no_eligible_candidate=complete["no_eligible_candidate"],
                    fallback_count=complete["fallback_count"],
                    fallback_rate=complete["fallback_count"] / max(1, complete["mutation_attempts"]),
                    mean_tv_distance=(float(np.mean([r["policy_metadata"].get("tv_distance", 0.)
                        for r in learning_events])) if learning_events else None),
                    different_argmax_rate=(float(np.mean([r["policy_metadata"].get("different_argmax", 0)
                        for r in learning_events])) if learning_events else None),
                    different_from_shadow_rule_rate=(float(np.mean([
                        r["policy_metadata"].get("different_from_shadow_rule", 0)
                        for r in learning_events])) if learning_events else None),
                    selected_model_score_mean=(float(np.mean([r["selected_policy_score"]
                        for r in learning_events if r["selected_policy_score"] is not None]))
                        if any(r["selected_policy_score"] is not None for r in learning_events)
                        else None),
                    eligible_candidate_counts=dict(Counter(
                        r["candidate_count"] for r in eligible)),
                    inference_seconds=complete["inference_seconds"],
                    optimisation_seconds=complete["optimisation_seconds"],
                    total_seconds=complete["total_seconds"],
                    oos_seconds=oos["elapsed_seconds"],
                    mean_generation_seconds=float(np.mean([g["generation_seconds"] for g in generations])),
                    objective_solution_median={name: float(np.median([c[name]["median"] for c in oos["candidates"]]))
                                               if oos["candidates"] else None
                                               for name in ("cost", "emission", "makespan")},
                    objective_solution_q90={name: float(np.median([c[name]["q90"] for c in oos["candidates"]]))
                                            if oos["candidates"] else None
                                            for name in ("cost", "emission", "makespan")},
                    oos_points=oos_points, curves=curve))
            ref = rows["rule"][-1]
            for method in ("random", "learning"):
                other = rows[method][-1]
                if (other["instance_digest"] != ref["instance_digest"] or
                        other["ccp_scenario_digest"] != ref["ccp_scenario_digest"] or
                        other["oos_scenario_digest"] != ref["oos_scenario_digest"] or
                        any(other["seeds"][key] != ref["seeds"][key]
                            for key in ("algorithm", "training", "path", "policy"))):
                    raise ValueError(f"unpaired run: {sid} #{repetition} {method}")
        contrasts = {}
        for control in ("rule", "random"):
            differences = [l["oos_hv"] - c["oos_hv"]
                           for l, c in zip(rows["learning"], rows[control])]
            statistic = (wilcoxon(differences).pvalue
                         if any(abs(x) > 1e-12 for x in differences) else 1.0)
            mean_delta = float(np.mean(differences))
            outcome = 0 if mean_delta > 1e-12 else 2 if mean_delta < -1e-12 else 1
            result["win_tie_loss"][f"learning_vs_{control}"][outcome] += 1
            contrasts[f"learning_vs_{control}"] = dict(
                paired_oos_hv_differences=differences,
                mean_difference=mean_delta, wilcoxon_p_value=float(statistic),
                set_coverage_learning_over_control=[_coverage(l["oos_points"], c["oos_points"])
                    for l, c in zip(rows["learning"], rows[control])],
                set_coverage_control_over_learning=[_coverage(c["oos_points"], l["oos_points"])
                    for l, c in zip(rows["learning"], rows[control])])
        result["instances"][sid] = dict(runs=rows, contrasts=contrasts)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--bounds", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = report(args.root, args.bounds, args.lock, args.out)
    print(result["win_tie_loss"])
