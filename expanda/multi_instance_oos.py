#!/usr/bin/env python3
"""Evaluate frozen final decisions on a shared 5000-scenario OOS master."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

import baseline_uncertainty as base
from multi_instance_catalog import read_instance
from multi_instance_logging import write_json
from run_ev_ccp_oos_pilot import scenario_digest, scenario_prefix
from run_formal_ev_vs_ccp_s30_30runs import restore_individual


OBJECTIVES = ("cost", "emission", "makespan")


def interval(master, start, stop):
    if not 0 <= start < stop <= master.size:
        raise ValueError("invalid OOS scenario interval")
    return base.ScenarioSet(size=stop-start, seed=master.seed,
        travel_multiplier={k: np.asarray(v[start:stop]).copy()
                           for k, v in master.travel_multiplier.items()},
        border_delay_h={k: np.asarray(v[start:stop]).copy()
                        for k, v in master.border_delay_h.items()},
        arc_border_event=dict(master.arc_border_event),
        border_event_mean_h=dict(master.border_event_mean_h), stochastic=master.stochastic)


def run(args):
    start = time.perf_counter()
    if args.out.exists():
        raise ValueError("OOS output path already exists")
    instance, batches = read_instance(args.instance)
    if hashlib.sha256(args.data.read_bytes()).hexdigest() != instance["source_data_sha256"]:
        raise ValueError("network workbook changed after instance generation")
    config = json.loads((args.run / "configuration.json").read_text())
    if config["instance_digest"] != instance["instance_digest"]:
        raise ValueError("run and OOS instance digest differ")
    source = json.loads((args.run / "final_feasible_nondominated.json").read_text())
    base.BORDER_EVENT_DEFINITIONS = base.load_border_event_definitions(base.DEFAULT_BORDER_EVENT_DATA_FILE)
    (nodes, regions, hold, proc, trans_cost, arcs, timetables, _source_batches,
     wait_cost, wait_emission, carbon, emission_factors, speeds, trans_map,
     border_delay, theta, _) = base.load_network_from_extended(str(args.data))
    tt, lookup = base.build_timetable_dict(timetables), base.build_arc_lookup(arcs)
    master = base.build_scenario_set(arcs, border_delay, 5000, args.oos_seed,
        stochastic=True, border_event_definitions=base.BORDER_EVENT_DEFINITIONS)
    full_digest = scenario_digest(master)
    prefix_digest = scenario_digest(scenario_prefix(master, 1000))
    if args.extend_from:
        old = np.load(args.extend_from, allow_pickle=False)
        if str(old["master_digest"].item()) != full_digest or str(old["instance_digest"].item()) != instance["instance_digest"]:
            raise ValueError("prefix cache belongs to a different OOS master or instance")
        if list(old["fingerprints"].astype(str)) != [r["decision_fingerprint"] for r in source]:
            raise ValueError("OOS prefix decisions differ")
        if args.size != 5000:
            raise ValueError("extension requires --size 5000")
        base.ACTIVE_SCENARIO_SET = interval(master, 1000, 5000)
        previous = {name: old[name] for name in OBJECTIVES}
    else:
        base.ACTIVE_SCENARIO_SET = scenario_prefix(master, args.size)
        previous = None
    base._PATH_SCENARIO_CACHE = {}
    base.RISK_METRIC = "ccp"
    base.CONFIDENCE_COST = base.CONFIDENCE_EMISSION = base.CONFIDENCE_TIME = .90
    arrays = {name: [] for name in OBJECTIVES}
    summaries = []
    for i, row in enumerate(source):
        ind = restore_individual(row, {}, tt, lookup)
        base.evaluate_individual(ind, batches, arcs, tt, wait_cost, wait_emission,
            node_hold_cost=hold, node_proc_cost=proc, carbon_tax_map=carbon,
            trans_map=trans_map, border_delay_map=border_delay, theta_rm=theta,
            node_trans_cost=trans_cost)
        current = (ind.cost_s, ind.emission_s, ind.makespan_s)
        details = dict(decision_fingerprint=row["decision_fingerprint"],
                       source_solution_id=row["source_solution_id"])
        for name, values in zip(OBJECTIVES, current):
            all_values = (np.concatenate([previous[name][i], values])
                          if previous is not None else values)
            if len(all_values) != args.size or not np.all(np.isfinite(all_values)):
                raise ValueError("nonfinite or incomplete OOS objective samples")
            arrays[name].append(all_values)
            details[name] = dict(median=float(np.median(all_values)),
                q90=float(base.empirical_ccp_quantile(all_values, .90)))
        summaries.append(details)
    args.out.mkdir(parents=True)
    matrix = {name: np.asarray(rows, dtype=float).reshape(len(source), args.size)
              for name, rows in arrays.items()}
    np.savez_compressed(args.out / f"oos_{args.size}.npz", **matrix,
        fingerprints=np.asarray([r["decision_fingerprint"] for r in source]),
        master_digest=np.asarray(full_digest),
        instance_digest=np.asarray(instance["instance_digest"]))
    result = dict(instance_id=instance["instance_id"],
                  instance_digest=instance["instance_digest"],
                  run=str(args.run), size=args.size, oos_seed=args.oos_seed,
                  oos_master_digest=full_digest, oos_prefix_digest=prefix_digest,
                  source_sha256=hashlib.sha256((args.run / "final_feasible_nondominated.json").read_bytes()).hexdigest(),
                  evaluated_decisions=len(source), elapsed_seconds=time.perf_counter()-start,
                  candidates=summaries)
    write_json(args.out / f"oos_{args.size}.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(__file__).parent / "data/data_expanded.xlsx")
    parser.add_argument("--instance", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--size", type=int, choices=(1000, 5000), required=True)
    parser.add_argument("--oos-seed", type=int, required=True)
    parser.add_argument("--extend-from", type=Path)
    args = parser.parse_args()
    result = run(args)
    print({k: v for k, v in result.items() if k != "candidates"})
