#!/usr/bin/env python3
"""Immutable post-processing of the completed formal EV/CCP30 experiment."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median, stdev

import numpy as np
from scipy.stats import rankdata, wilcoxon

ROOT = Path("formal_ev_vs_ccp_s30_30runs")
EXPECTED_SEED = 930001
EXPECTED_DIGEST = "9219266ad001420fffdcb93b3d135774ab893c9c8357fcdfbcf11520e29788cb"
EXPECTED_CANDIDATES = 15664
METHODS = ("EV", "CCP30")
VIEWS = {"OOS-MEAN": "mean", "OOS-Q90": "q90"}
OBJECTIVES = ("cost", "emission", "makespan")
HV_REFERENCE = (1.2, 1.2, 1.2)
EPSILON = 1e-12
ALPHA = 0.05


def read_json(path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path, rows, fields=None):
    rows = list(rows)
    fields = fields or list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def objectives(row, statistic):
    return tuple(float(row[name][statistic]) for name in OBJECTIVES)


def nondominated_mask(points):
    """Exact minimisation mask in 3-D, O(n log n), retaining objective duplicates."""
    pts = np.asarray(points, dtype=float)
    n = len(pts)
    order = sorted(range(n), key=lambda i: (pts[i, 0], pts[i, 1], pts[i, 2]))
    ys = sorted(set(float(x) for x in pts[:, 1]))
    positions = {y: i + 1 for i, y in enumerate(ys)}
    tree = [math.inf] * (len(ys) + 1)
    keep = np.zeros(n, dtype=bool)
    prior_exact = {}

    def query(i):
        result = math.inf
        while i:
            result = min(result, tree[i]); i -= i & -i
        return result

    def update(i, value):
        while i < len(tree):
            tree[i] = min(tree[i], value); i += i & -i

    for idx in order:
        x, y, z = map(float, pts[idx])
        exact = (x, y, z)
        if exact in prior_exact:
            keep[idx] = prior_exact[exact]
            continue
        best_z = query(positions[y])
        # Any earlier non-identical point attaining z or lower is a strict dominator.
        keep[idx] = best_z > z
        prior_exact[exact] = bool(keep[idx])
        update(positions[y], z)
    return keep


def nondominated_records(rows, statistic):
    mask = nondominated_mask([objectives(r, statistic) for r in rows])
    return [r for r, keep in zip(rows, mask) if keep]


def unique_points(rows, statistic):
    return sorted(set(objectives(r, statistic) for r in rows))


def normalise(points, mins, ranges):
    return [tuple((p[i] - mins[i]) / ranges[i] for i in range(3)) for p in points]


def exact_hv_3d(points, reference=HV_REFERENCE):
    ref = np.asarray(reference, dtype=float)
    pts = np.asarray(points, dtype=float).reshape((-1, 3))
    pts = pts[np.all(pts <= ref, axis=1)]
    if not len(pts): return 0.0
    xs = sorted(set(float(x) for x in pts[:, 0] if x < ref[0])) + [float(ref[0])]
    volume = 0.0
    for left, right in zip(xs[:-1], xs[1:]):
        active = pts[pts[:, 0] <= left + 1e-15]
        ys = sorted(set(float(y) for y in active[:, 1] if y < ref[1])) + [float(ref[1])]
        area = 0.0
        for low, high in zip(ys[:-1], ys[1:]):
            z = min((float(p[2]) for p in active if p[1] <= low + 1e-15), default=float(ref[2]))
            area += (high - low) * max(0.0, float(ref[2]) - z)
        volume += (right - left) * area
    return float(volume)


def igd_plus(reference, approximation):
    p, q = np.asarray(reference), np.asarray(approximation)
    return float(np.mean([np.min(np.sqrt(np.sum(np.maximum(q - x, 0.0) ** 2, axis=1))) for x in p]))


def spacing(points):
    q = np.asarray(points)
    if len(q) < 2: return 0.0
    distances = []
    for i in range(len(q)):
        d = np.sqrt(np.sum((q - q[i]) ** 2, axis=1)); d[i] = np.inf
        distances.append(float(np.min(d)))
    return float(np.std(distances, ddof=1))


def audit_inputs():
    completion = read_json(ROOT / "validation/COMPLETE.json")
    config = read_json(ROOT / "validation/configuration.json")
    seeds = read_json(ROOT / "FORMAL_SEEDS.json")
    assert config == {"size": 5000, "seed": EXPECTED_SEED, "digest": EXPECTED_DIGEST, "selection_sets_reused": False}
    assert seeds["final_validation_seed"] == EXPECTED_SEED
    assert completion["candidate_count"] == EXPECTED_CANDIDATES
    summary_path = ROOT / "validation/per_candidate_summary.json"
    assert sha256(summary_path) == completion["summary_sha256"]
    rows = read_json(summary_path)
    assert isinstance(rows, list) and len(rows) == EXPECTED_CANDIDATES
    assert all(set(r) >= {"run_id", "method", "source_solution_id", "training_objectives", *OBJECTIVES} for r in rows)
    ids = set()
    runtime_rows = []
    for run in range(1, 31):
        seed = seeds["replicates"][run - 1]
        assert seed["run_id"] == run
        for method in METHODS:
            base = ROOT / f"run_{run:02d}" / method
            marker, cfg = read_json(base / "COMPLETE.json"), read_json(base / "configuration.json")
            candidate_path = base / "final_feasible_nondominated.json"
            assert marker["completed"] and marker["run_id"] == run and marker["method"] == method
            assert marker["algorithm_seed"] == seed["algorithm_seed"] == cfg["algorithm_seed"]
            assert marker["candidate_sha256"] == sha256(candidate_path)
            assert marker["configuration_sha256"] == sha256(base / "configuration.json")
            candidate_rows = read_json(candidate_path)
            assert len(candidate_rows) == marker["candidate_count"]
            source_ids = {x["source_solution_id"] for x in candidate_rows}
            expected = {r["source_solution_id"] for r in rows if r["run_id"] == run and r["method"] == method}
            assert source_ids == expected and not ids.intersection(source_ids)
            ids.update(source_ids)
            runtime_rows.append((run, method, marker["runtime_seconds"], marker["candidate_count"]))
    assert len(ids) == EXPECTED_CANDIDATES
    assert completion["all_fingerprints_unchanged"] and all(r["decision_fingerprint"] == r["signature_after"] for r in rows)
    return rows, completion, runtime_rows


def descriptive(values):
    return dict(mean=mean(values), sd=stdev(values), median=median(values), minimum=min(values), maximum=max(values))


def main():
    rows, completion, runtimes = audit_inputs()
    metrics_dir, stats_dir = ROOT / "metrics", ROOT / "statistics"
    metrics_dir.mkdir(exist_ok=True); stats_dir.mkdir(exist_ok=True)
    norms, references, fronts = {}, {}, {}
    for view, statistic in VIEWS.items():
        all_pts = np.asarray([objectives(r, statistic) for r in rows])
        mins, maxs = np.min(all_pts, axis=0), np.max(all_pts, axis=0)
        raw_ranges = maxs - mins
        ranges = np.where(raw_ranges > EPSILON, raw_ranges, 1.0)
        norms[view] = {"objective_order": list(OBJECTIVES), "minima": mins.tolist(), "maxima": maxs.tolist(),
            "raw_ranges": raw_ranges.tolist(), "effective_ranges": ranges.tolist(), "range_epsilon": EPSILON,
            "zero_range_rule": "effective range 1.0 and normalized coordinate 0.0",
            "scope": "all 15,664 validated candidates pooled across all 60 runs",
            "hypervolume_reference_point": list(HV_REFERENCE)}
        front_rows = nondominated_records(rows, statistic)
        front_pts = unique_points(front_rows, statistic)
        references[view] = normalise(front_pts, mins, ranges)
        fronts[view] = front_rows
        payload = {"view": view, "statistic": statistic, "objective_order": list(OBJECTIVES),
            "provenance_record_count": len(front_rows), "unique_objective_count": len(front_pts),
            "method_provenance_counts": {m: sum(r["method"] == m for r in front_rows) for m in METHODS},
            "records": [{"run_id": r["run_id"], "method": r["method"], "source_solution_id": r["source_solution_id"],
                "objectives": dict(zip(OBJECTIVES, objectives(r, statistic)))} for r in front_rows]}
        write_json(metrics_dir / ("global_oos_mean_reference_front.json" if statistic == "mean" else "global_oos_q90_reference_front.json"), payload)
    write_json(metrics_dir / "normalisation.json", norms)

    metric_rows = []
    for run in range(1, 31):
        for method in METHODS:
            source = [r for r in rows if r["run_id"] == run and r["method"] == method]
            runtime = next(x[2] for x in runtimes if x[:2] == (run, method))
            for view, statistic in VIEWS.items():
                mins = np.asarray(norms[view]["minima"]); ranges = np.asarray(norms[view]["effective_ranges"])
                local_front = nondominated_records(source, statistic)
                local_pts = unique_points(local_front, statistic)
                norm = normalise(local_pts, mins, ranges)
                metric_rows.append({"run_id": run, "method": method, "view": view,
                    "hypervolume": exact_hv_3d(norm), "igd_plus": igd_plus(references[view], norm),
                    "spacing": spacing(norm), "nondominated_size": len(local_front),
                    "unique_objective_size": len(local_pts), "validated_candidate_count": len(source),
                    "optimisation_runtime_seconds": runtime})
    wide_metric_rows = []
    for run in range(1, 31):
        for method in METHODS:
            out = {"run_id": run, "method": method}
            for view in VIEWS:
                slug = view.lower().replace("oos-", "oos_").replace("-", "_")
                item = next(x for x in metric_rows if x["run_id"] == run and x["method"] == method and x["view"] == view)
                for key in ("hypervolume", "igd_plus", "spacing", "nondominated_size", "unique_objective_size"):
                    out[f"{slug}_{key}"] = item[key]
                out["validated_candidate_count"] = item["validated_candidate_count"]
                out["optimisation_runtime_seconds"] = item["optimisation_runtime_seconds"]
            wide_metric_rows.append(out)
    write_csv(metrics_dir / "run_level_metrics.csv", wide_metric_rows)

    paired_rows = []
    for run in range(1, 31):
        out = {"run_id": run}
        for view in VIEWS:
            slug = view.lower().replace("oos-", "oos_").replace("-", "_")
            a = next(x for x in metric_rows if x["run_id"] == run and x["method"] == "EV" and x["view"] == view)
            b = next(x for x in metric_rows if x["run_id"] == run and x["method"] == "CCP30" and x["view"] == view)
            for metric in ("hypervolume", "igd_plus", "spacing", "nondominated_size"):
                out[f"{slug}_EV_{metric}"] = a[metric]; out[f"{slug}_CCP30_{metric}"] = b[metric]
                out[f"{slug}_EV_minus_CCP30_{metric}"] = a[metric] - b[metric]
        paired_rows.append(out)
    write_csv(metrics_dir / "paired_ev_ccp_metrics.csv", paired_rows)

    gen_rows = []
    for run_scope in list(range(1, 31)) + ["ALL"]:
        for method in METHODS:
            subset = [r for r in rows if r["method"] == method and (run_scope == "ALL" or r["run_id"] == run_scope)]
            target = "mean" if method == "EV" else "q90"
            out = {"scope": "method" if run_scope == "ALL" else "run", "run_id": "" if run_scope == "ALL" else run_scope,
                "method": method, "comparison": "training direct-expected-input vs OOS mean" if method == "EV" else "training q90 vs OOS q90",
                "candidate_count": len(subset)}
            for obj in ("cost", "makespan", "emission"):
                signed = [100 * (r[obj][target] - r["training_objectives"][obj]) / r["training_objectives"][obj] for r in subset]
                absolute = [abs(x) for x in signed]
                for label, vals in (("signed_pct_delta", signed), ("absolute_percentage_error", absolute)):
                    out[f"{obj}_{label}_mean"] = mean(vals); out[f"{obj}_{label}_median"] = median(vals)
                    out[f"{obj}_{label}_sd"] = stdev(vals) if len(vals) > 1 else 0.0
            gen_rows.append(out)
    write_csv(metrics_dir / "generalisation_summary.csv", gen_rows)

    desc_rows, comp_rows, test_items = [], [], []
    directions = {"hypervolume": "higher", "igd_plus": "lower", "spacing": "lower", "nondominated_size": "descriptive"}
    for view in VIEWS:
        for metric in directions:
            ev = [r[metric] for r in metric_rows if r["view"] == view and r["method"] == "EV"]
            ccp = [r[metric] for r in metric_rows if r["view"] == view and r["method"] == "CCP30"]
            diff = np.asarray(ev) - np.asarray(ccp)
            for method, vals in (("EV", ev), ("CCP30", ccp)):
                desc_rows.append({"view": view, "metric": metric, "method": method, "n": 30, **descriptive(vals)})
            tol = 1e-12
            if directions[metric] == "higher": wins = int(sum(diff > tol)); losses = int(sum(diff < -tol))
            elif directions[metric] == "lower": wins = int(sum(diff < -tol)); losses = int(sum(diff > tol))
            else: wins = int(sum(diff > tol)); losses = int(sum(diff < -tol))
            ties = 30 - wins - losses
            nz = diff[np.abs(diff) > tol]
            if len(nz):
                ranks = rankdata(np.abs(nz)); rank_biserial = float((ranks[nz > 0].sum() - ranks[nz < 0].sum()) / ranks.sum())
                result = wilcoxon(diff, zero_method="pratt", alternative="two-sided", method="auto")
                statistic, p_value = float(result.statistic), float(result.pvalue)
            else: rank_biserial, statistic, p_value = 0.0, 0.0, 1.0
            dz = float(np.mean(diff) / np.std(diff, ddof=1)) if np.std(diff, ddof=1) > 0 else 0.0
            row = {"view": view, "metric": metric, "preferred_direction": directions[metric],
                "paired_difference_definition": "EV minus CCP30", "difference_mean": float(np.mean(diff)),
                "difference_median": float(np.median(diff)), "EV_wins": wins, "ties": ties, "CCP30_wins": losses,
                "wilcoxon_statistic": statistic, "raw_p_value": p_value,
                "matched_pairs_rank_biserial_EV_minus_CCP30": rank_biserial, "cohen_dz": dz}
            comp_rows.append(row); test_items.append(row)
    order = sorted(range(len(test_items)), key=lambda i: test_items[i]["raw_p_value"])
    adjusted = [0.0] * len(order); running = 0.0; m = len(order)
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * test_items[idx]["raw_p_value"]); adjusted[idx] = min(1.0, running)
    for row, adj in zip(comp_rows, adjusted): row["holm_adjusted_p_value"] = adj; row["reject_at_0.05"] = adj < ALPHA
    write_csv(stats_dir / "descriptive_statistics.csv", desc_rows)
    write_csv(stats_dir / "paired_comparison.csv", comp_rows)
    write_json(stats_dir / "statistical_tests.json", {"test": "two-sided paired Wilcoxon signed-rank",
        "predefinition_basis": "paired design; no inferential framework found in audited deterministic experiment code/paper artefacts",
        "null_hypothesis": "the paired EV-minus-CCP30 difference distribution is symmetric about zero",
        "alpha": ALPHA, "multiplicity": "Holm family-wise correction across 8 view/metric tests",
        "zero_method": "Pratt", "effect_sizes": ["matched-pairs rank-biserial correlation", "Cohen dz"], "results": comp_rows})

    def fmt(x): return f"{x:.6g}"
    summary_lines = ["# Concise statistical summary", "", "Thirty paired algorithm-seed replicates were analysed. Values are sample mean ± sample SD.", ""]
    for view in VIEWS:
        summary_lines.append(f"## {view}"); summary_lines.append("")
        for metric in directions:
            e = next(r for r in desc_rows if r["view"] == view and r["metric"] == metric and r["method"] == "EV")
            c = next(r for r in desc_rows if r["view"] == view and r["metric"] == metric and r["method"] == "CCP30")
            t = next(r for r in comp_rows if r["view"] == view and r["metric"] == metric)
            summary_lines.append(f"- {metric}: EV {fmt(e['mean'])} ± {fmt(e['sd'])}; CCP30 {fmt(c['mean'])} ± {fmt(c['sd'])}; W/T/L {t['EV_wins']}/{t['ties']}/{t['CCP30_wins']}; Holm p={fmt(t['holm_adjusted_p_value'])}; rank-biserial={fmt(t['matched_pairs_rank_biserial_EV_minus_CCP30'])}.")
        summary_lines.append("")
    (stats_dir / "STATISTICAL_SUMMARY.md").write_text("\n".join(summary_lines), encoding="utf-8")

    rt = [x[2] for x in runtimes]
    ev_gen = next(r for r in gen_rows if r["scope"] == "method" and r["method"] == "EV")
    ccp_gen = next(r for r in gen_rows if r["scope"] == "method" and r["method"] == "CCP30")
    emission_span = max(r["emission"]["maximum"] - r["emission"]["minimum"] for r in rows)
    punctuality = [r["punctuality_diagnostic_min_batch_on_time_probability"] for r in rows]
    report = ["# Final formal EV versus CCP30 report", "",
        "## Frozen formal experiment configuration", "", "The frozen design used 30 paired replicates, population 300, 500 generations, α=0.9, direct expected-input EV, and active quantile-objective CCP30 with separate empirical ceil(0.9×30)-th order-statistic objectives. EV_r and CCP30_r shared algorithm seed r.", "",
        "## Completion and validation", "", f"All 30 EV and all 30 CCP30 completion markers, configurations, candidate hashes, and paired seeds passed the immutable-input audit. Common held-out S5000 validation completed for {len(rows):,} candidates in {completion['runtime_seconds']:.3f} s; every decision fingerprint was unchanged. Seed: `{EXPECTED_SEED}`. Digest: `{EXPECTED_DIGEST}`.", "",
        "## Runtime summary", "", f"The 60 optimisation runs totalled {sum(rt):.3f} s ({mean(rt):.3f} ± {stdev(rt):.3f} s; range {min(rt):.3f}–{max(rt):.3f} s). Stage C took {completion['runtime_seconds']:.3f} s. No optimisation or validation was rerun during post-processing.", "",
        "## OOS run-level results and paired statistics", "", "OOS-MEAN represents expected/risk-neutral performance; OOS-Q90 represents upper-tail/risk-aware performance. HV uses one pooled-candidate normalization per view and the fixed normalized reference point (1.2, 1.2, 1.2). IGD+ uses the corresponding global held-out nondominated front. Spacing is computed in the same normalized objective space. Detailed 60-row results and 30-row paired contrasts are in `metrics/`; descriptive and paired tests are in `statistics/`.", ""]
    report += summary_lines[4:]
    report += ["## Generalisation", "", f"EV training direct-expected-input → OOS mean: cost signed delta mean {ev_gen['cost_signed_pct_delta_mean']:.4f}% (mean APE {ev_gen['cost_absolute_percentage_error_mean']:.4f}%); makespan signed delta mean {ev_gen['makespan_signed_pct_delta_mean']:.4f}% (mean APE {ev_gen['makespan_absolute_percentage_error_mean']:.4f}%).", "", f"CCP30 training q90 → OOS q90: cost signed delta mean {ccp_gen['cost_signed_pct_delta_mean']:.4f}% (mean APE {ccp_gen['cost_absolute_percentage_error_mean']:.4f}%); makespan signed delta mean {ccp_gen['makespan_signed_pct_delta_mean']:.4f}% (mean APE {ccp_gen['makespan_absolute_percentage_error_mean']:.4f}%). Per-run and method summaries are in `metrics/generalisation_summary.csv`.", "",
        "## Punctuality and emission diagnostics", "", f"Minimum-batch on-time probability relative to `Batches.LT` is retained only as a punctuality diagnostic (candidate median {median(punctuality):.4f}, range {min(punctuality):.4f}–{max(punctuality):.4f}); it is not CCP feasibility, reliability, or chance-constraint satisfaction. Emission is scenario-invariant under the current model (maximum within-candidate scenario span {emission_span:.6g}); its tiny floating-point training/OOS deltas are reported separately and are not treated as a generalisation-error discriminator.", "",
        "## Scientific interpretation", "", "The two held-out views answer different questions and must be interpreted separately. Under OOS-MEAN, CCP30 won 23/30 HV pairs and 26/30 IGD+ pairs; both differences survived Holm correction, so the evidence does not support an EV expected-performance advantage in this experiment. Under OOS-Q90, CCP30 won 21/30 pairs for both HV and IGD+, but neither comparison survived the eight-test Holm correction. Spacing was unresolved in both views. Thus the observed direction generally favours CCP30 for convergence/coverage, most clearly under mean evaluation, while tail-view inferential evidence is weaker. Nondominated-set size is descriptive, not intrinsically a quality score.", "",
        "## Limitations", "", "CCP30 uses active quantile-objective semantics: it minimizes three separate marginal empirical q90 objectives from only 30 training scenarios. It is not a joint chance constraint, and punctuality is not part of CCP feasibility. The global reference fronts are empirical fronts from the 15,664 validated formal candidates, not the unknown true Pareto fronts. The one common S5000 set supports paired comparisons but does not eliminate Monte Carlo error. Exact objective duplicates are deduplicated only for metric geometry; all nondominated provenance records are retained in the saved reference-front files.", "",
        "## Reproducibility and checkpoints", "", f"Validation summary SHA-256: `{completion['summary_sha256']}`. Scenario-level SHA-256 recorded by the completed checkpoint (not reread): `{completion['scenario_results_sha256']}`. Normalization bounds, epsilon handling, and HV reference are frozen in `metrics/normalisation.json`. Input audit checks candidate/configuration hashes for every run and verifies all 15,664 source IDs.", ""]
    (ROOT / "FINAL_FORMAL_REPORT.md").write_text("\n".join(report), encoding="utf-8")

    write_json(ROOT / "INPUT_AUDIT.json", {"status": "passed", "EV_completion_markers": 30,
        "CCP30_completion_markers": 30, "all_candidate_and_configuration_hashes_match": True,
        "validation_complete_marker": True, "validation_size": 5000, "validation_seed": EXPECTED_SEED,
        "validation_digest": EXPECTED_DIGEST, "validated_candidates": len(rows),
        "per_candidate_summary_readable_and_complete": True,
        "per_candidate_summary_sha256": completion["summary_sha256"],
        "all_source_solution_ids_mapped_exactly_once": True, "all_fingerprints_unchanged": True,
        "scenario_level_file_read": False})

    outputs = sorted([p for d in (metrics_dir, stats_dir) for p in d.iterdir() if p.is_file()] + [ROOT / "FINAL_FORMAL_REPORT.md", ROOT / "INPUT_AUDIT.json"])
    checkpoint = hashlib.sha256("".join(f"{p.relative_to(ROOT)}\0{sha256(p)}\n" for p in outputs).encode()).hexdigest()
    write_json(ROOT / "POSTPROCESSING_COMPLETE.json", {"status": "complete", "input_summary_sha256": completion["summary_sha256"],
        "validation_seed": EXPECTED_SEED, "validation_digest": EXPECTED_DIGEST, "candidate_count": len(rows),
        "scenario_level_file_read": False, "output_checkpoint_sha256": checkpoint,
        "checkpoint_definition": "SHA-256 of sorted relative-path, NUL, file-SHA256, newline entries; excludes this marker",
        "outputs": [str(p.relative_to(ROOT)) for p in outputs]})
    print(json.dumps({"checkpoint": checkpoint, "front_counts": {v: {m: sum(r['method'] == m for r in fronts[v]) for m in METHODS} for v in VIEWS}, "tests": comp_rows}, indent=2))


if __name__ == "__main__":
    main()
