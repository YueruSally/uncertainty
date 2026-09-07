#!/usr/bin/env python3
"""Resume-safe formal 30-pair EV versus CCP30 experiment."""
from __future__ import annotations
import argparse, csv, gzip, hashlib, json, os, random, tempfile, time
from copy import deepcopy
from pathlib import Path
import numpy as np
import baseline_uncertainty as base
from run_ccp_candidate_pool import decision_signature
from run_ev_ccp_oos_pilot import candidate_rows, json_candidate, scenario_digest, summarise

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "formal_ev_vs_ccp_s30_30runs"
SEEDS_FILE = OUT / "FORMAL_SEEDS.json"
PLAN_FILE = OUT / "FORMAL_EXPERIMENT_PLAN.json"
FORMULATION = "EV direct expectations / CCP30 separate empirical q90 objectives; punctuality diagnostic only"
WAITING_CONFIG = base.waiting_emission_configuration()

def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name+".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as h: json.dump(value,h,indent=2); h.flush(); os.fsync(h.fileno())
        os.replace(name,path)
    finally:
        if os.path.exists(name): os.unlink(name)

def digest(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""): h.update(block)
    return h.hexdigest()

def expected_meta(run_id, method, seed):
    return {"method":method,"run_id":run_id,"algorithm_seed":seed["algorithm_seed"],
      "training_seed":seed["ccp_training_seed"] if method=="CCP30" else None,
      "population":300,"generations":500,"alpha":.9,"formulation":FORMULATION,
      **WAITING_CONFIG}

def valid_completion(method_dir, expected):
    marker=method_dir/"COMPLETE.json"; candidates=method_dir/"final_feasible_nondominated.json"; config=method_dir/"configuration.json"
    if not all(p.is_file() for p in (marker,candidates,config)): return False
    try: m=json.loads(marker.read_text()); c=json.loads(config.read_text()); json.loads(candidates.read_text())
    except Exception: return False
    return (all(m.get(k)==v and c.get(k)==v for k,v in expected.items()) and
            m.get("candidate_sha256")==digest(candidates) and m.get("configuration_sha256")==digest(config))

def restore_individual(row, path_lib, tt_dict=None, arc_lookup=None,
                       restoration_stats=None):
    """Restore the exact persisted decision, including evolved paths.

    Crossover and mode mutation can create valid ``Path`` objects that are not
    members of the capped static path library.  Their complete decision data
    is persisted, so rebuild those paths from the underlying arcs rather than
    substituting a library path.  Road fallback is deliberately disabled:
    restoration must preserve every saved mode exactly.
    """
    lookup={(tuple(p.nodes),tuple(p.modes)):p for paths in path_lib.values() for p in paths}
    ind=base.Individual()
    used_fallback=False
    for block in row["decision"]:
        key=(block["origin"],block["destination"],int(block["batch_id"]))
        restored=[]
        for allocation in block["allocations"]:
            path_key=(tuple(allocation["nodes"]),tuple(allocation["modes"]))
            path=lookup.get(path_key)
            if path is None:
                if tt_dict is None or arc_lookup is None:
                    raise RuntimeError(
                        "saved path is absent from static lookup and exact "
                        "network reconstruction was not configured: "
                        f"{path_key}")
                path=base.rebuild_path_from_nodes_modes(
                    block["origin"],block["destination"],
                    allocation["nodes"],allocation["modes"],
                    tt_dict,arc_lookup,allow_road_fallback=False)
                if path is None or (tuple(path.nodes),tuple(path.modes)) != path_key:
                    raise RuntimeError(
                        "formal-data integrity failure: saved path cannot be "
                        f"reconstructed exactly from underlying arcs: {path_key}")
                lookup[path_key]=path
                used_fallback=True
                if restoration_stats is not None:
                    restoration_stats["fallback_allocations"] = (
                        restoration_stats.get("fallback_allocations",0)+1)
            restored.append(base.PathAllocation(path,float(allocation["share"])))
        ind.od_allocations[key]=restored
    if decision_signature(ind)!=row["decision_fingerprint"]: raise RuntimeError("restored decision fingerprint mismatch")
    if restoration_stats is not None:
        label="fallback_candidates" if used_fallback else "static_candidates"
        restoration_stats[label]=restoration_stats.get(label,0)+1
    return ind

def audit_restoration(rows, path_lib, tt_dict, arc_lookup):
    stats={"total_candidate_records":len(rows),
           "unique_fingerprints":len({row["decision_fingerprint"] for row in rows}),
           "static_candidates":0,"fallback_candidates":0,
           "fallback_allocations":0,"failures":0}
    for row in rows:
        try:
            restored=restore_individual(
                row,path_lib,tt_dict,arc_lookup,restoration_stats=stats)
            if decision_signature(restored)!=row["decision_fingerprint"]:
                raise RuntimeError("restored decision fingerprint mismatch")
        except Exception:
            stats["failures"]+=1
            raise
    return stats

def preflight():
    seeds=json.loads(SEEDS_FILE.read_text()); plan=json.loads(PLAN_FILE.read_text()); reps=seeds["replicates"]
    alg=[x["algorithm_seed"] for x in reps]; train=[x["ccp_training_seed"] for x in reps]
    checks={"30_seed_rows":len(reps)==30,"unique_algorithm_seeds":len(set(alg))==30,
      "unique_training_seeds":len(set(train))==30,"all_formal_seeds_unique":len(set(alg+train+[seeds["final_validation_seed"]]))==61,
      "path_seed_zero":seeds["path_library_seed"]==0,"methods_exact":plan["methods"]==["EV","CCP30"],
      "settings_exact":(plan["population"],plan["generations"],plan["alpha"],plan["replicates"])==(300,500,.9,30),
      "ccp_S30":plan["CCP30"].endswith("S=30"),"validation_S5000":plan["final_validation_scenarios"]==5000,
      "quantile_definition":base.empirical_ccp_quantile(np.arange(30),.9)==26.0}
    if not all(checks.values()): raise RuntimeError(f"preflight failed: {checks}")
    return checks

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--preflight-only",action="store_true"); ap.add_argument("--run-id",type=int,choices=range(1,31)); args=ap.parse_args()
    checks=preflight(); print("[FORMAL GATE] internal checks passed",checks,flush=True)
    if args.preflight_only:return 0
    seeds=json.loads(SEEDS_FILE.read_text()); (OUT/"validation").mkdir(exist_ok=True); (OUT/"metrics").mkdir(exist_ok=True); (OUT/"statistics").mkdir(exist_ok=True)
    base.BORDER_EVENT_DEFINITIONS=base.load_border_event_definitions(base.DEFAULT_BORDER_EVENT_DATA_FILE)
    net=base.load_network_from_extended(ROOT/"data/data_expanded.xlsx")
    (node_names,node_region,node_hold_cost,node_proc_cost,node_trans_cost,arcs,timetables,batches,waiting_cost,wait_emission,carbon_tax,emission_factors,mode_speeds,trans_map,border_delay_map,theta_rm,_)=net
    wait_emission=base.WAIT_EMISSION_gCO2_per_TEU_H_DEFAULT
    base.print_waiting_emission_configuration()
    for b in batches:b.penalty_per_teu_h=base.DEFAULT_LATE_PENALTY_USD_PER_TEU_H
    tt=base.build_timetable_dict(timetables); lookup=base.build_arc_lookup(arcs)
    random.seed(0);np.random.seed(0); paths=base.build_path_library(node_names,node_region,arcs,batches,tt,lookup);base.sanity_check_path_lib(batches,paths)
    ev=base.build_expected_value_scenario_set(arcs,border_delay_map,seed=0,border_event_definitions=base.BORDER_EVENT_DEFINITIONS)
    options=base.build_reliable_path_options(batches,paths,tt,trans_map,border_delay_map,ev,mode="ev")
    all_rows=[]
    selected_seeds=[seed for seed in seeds["replicates"] if args.run_id is None or seed["run_id"]==args.run_id]
    for seed in selected_seeds:
      rid=seed["run_id"]
      training=base.build_scenario_set(arcs,border_delay_map,30,seed["ccp_training_seed"],stochastic=True,border_event_definitions=base.BORDER_EVENT_DEFINITIONS)
      for method,scenarios,risk in (("EV",ev,"ev"),("CCP30",training,"ccp")):
        d=OUT/f"run_{rid:02d}"/method; d.mkdir(parents=True,exist_ok=True); meta=expected_meta(rid,method,seed)
        if valid_completion(d,meta): print(f"[RESUME] valid complete run={rid:02d} method={method}",flush=True); all_rows.extend(json.loads((d/"final_feasible_nondominated.json").read_text())); continue
        recovery=(d/"COMPLETE.json").exists() or any(d.iterdir())
        atomic_json(d/"configuration.json",{**meta,"training_scenarios":None if method=="EV" else 30,"scenario_digest":scenario_digest(scenarios),"recovery_of_incomplete_artifact":recovery})
        print(f"[FORMAL START] run={rid:02d} method={method} algorithm_seed={seed['algorithm_seed']} training_seed={meta['training_seed']} semantics={'direct_expected_inputs' if method=='EV' else 'empirical_q90_S30'}",flush=True)
        base.ACTIVE_SCENARIO_SET=scenarios;base._PATH_SCENARIO_CACHE={};base.RISK_METRIC=risk;base.CONFIDENCE_COST=base.CONFIDENCE_EMISSION=base.CONFIDENCE_TIME=.9
        random.seed(seed["algorithm_seed"]);np.random.seed(seed["algorithm_seed"]); started=time.perf_counter()
        pop=base.run_nsga2(node_names,node_region,node_hold_cost,node_proc_cost,node_trans_cost,arcs,timetables,batches,waiting_cost,wait_emission,carbon_tax,emission_factors,mode_speeds,trans_map,border_delay_map,theta_rm,paths,options,pop_size=300,generations=500)[0]
        front=base.fast_non_dominated_sort(pop)[0]; inds=[deepcopy(pop[i]) for i in front if pop[i].feasible]
        cfg={"population":300,"generations":500,"alpha":.9,"seeds":{"algorithm":seed["algorithm_seed"],"training":meta["training_seed"],"path_library":0}}
        rows=candidate_rows(method,f"formal-run-{rid:02d}-{method}",inds,cfg)
        serial=[]
        for x in rows:y=json_candidate(x);y["run_id"]=rid;serial.append(y)
        atomic_json(d/"final_feasible_nondominated.json",serial); cp=d/"final_feasible_nondominated.json"; conf=d/"configuration.json"
        atomic_json(d/"COMPLETE.json",{**meta,"candidate_count":len(serial),"runtime_seconds":time.perf_counter()-started,"candidate_sha256":digest(cp),"configuration_sha256":digest(conf),"completed":True})
        print(f"[FORMAL COMPLETE] run={rid:02d} method={method} candidates={len(serial)}",flush=True); all_rows.extend(serial)
    print(f"[FORMAL] selected optimisations complete: {len(selected_seeds) * 2}",flush=True)
    if args.run_id is not None:
      print("[FORMAL] validation deferred for run-specific optimisation",flush=True)
      return 0
    restoration=audit_restoration(all_rows,paths,tt,lookup)
    print(f"[FORMAL GATE] restoration audit passed {restoration}",flush=True)
    print("[FORMAL] starting common S5000 validation",flush=True)
    # Validation is intentionally deferred until all valid completion markers exist.
    validation=base.build_scenario_set(arcs,border_delay_map,5000,seeds["final_validation_seed"],stochastic=True,border_event_definitions=base.BORDER_EVENT_DEFINITIONS)
    atomic_json(OUT/"validation/configuration.json",{"size":5000,"seed":seeds["final_validation_seed"],"digest":scenario_digest(validation),"selection_sets_reused":False,**WAITING_CONFIG})
    base.ACTIVE_SCENARIO_SET=validation;base._PATH_SCENARIO_CACHE={};base.RISK_METRIC="ccp"
    summaries=[]; scenario_path=OUT/"validation/scenario_level_results.csv.gz"; started=time.perf_counter()
    with gzip.open(scenario_path,"wt",newline="",encoding="utf-8") as h:
      writer=csv.DictWriter(h,fieldnames=["run_id","method","source_solution_id","decision_fingerprint","scenario_id","cost","emission","makespan"]);writer.writeheader()
      for row in all_rows:
        ind=restore_individual(row,paths,tt,lookup); before=decision_signature(ind)
        base.evaluate_individual(ind,batches,arcs,tt,waiting_cost,wait_emission,node_hold_cost=node_hold_cost,node_proc_cost=node_proc_cost,carbon_tax_map=carbon_tax,trans_map=trans_map,border_delay_map=border_delay_map,theta_rm=theta_rm,node_trans_cost=node_trans_cost)
        after=decision_signature(ind)
        if before!=after: raise RuntimeError("Stage C changed fixed decision")
        for sid,(c,e,t) in enumerate(zip(ind.cost_s,ind.emission_s,ind.makespan_s)):writer.writerow({"run_id":row["run_id"],"method":row["method"],"source_solution_id":row["source_solution_id"],"decision_fingerprint":before,"scenario_id":sid,"cost":float(c),"emission":float(e),"makespan":float(t)})
        summaries.append({"run_id":row["run_id"],"method":row["method"],"source_solution_id":row["source_solution_id"],"decision_fingerprint":before,"signature_after":after,"training_objectives":row["optimisation_objectives"],"cost":summarise(ind.cost_s),"emission":summarise(ind.emission_s),"makespan":summarise(ind.makespan_s),"punctuality_diagnostic_min_batch_on_time_probability":float(min(ind.batch_on_time_prob.values())) if ind.batch_on_time_prob else None})
    atomic_json(OUT/"validation/per_candidate_summary.json",summaries)
    atomic_json(OUT/"validation/COMPLETE.json",{"size":5000,"seed":seeds["final_validation_seed"],"candidate_count":len(summaries),"runtime_seconds":time.perf_counter()-started,"scenario_results_sha256":digest(scenario_path),"summary_sha256":digest(OUT/"validation/per_candidate_summary.json"),"all_fingerprints_unchanged":all(x["decision_fingerprint"]==x["signature_after"] for x in summaries)})
    print(f"[FORMAL COMPLETE] common validation candidates={len(summaries)}",flush=True)
    return 0
if __name__=="__main__":raise SystemExit(main())
