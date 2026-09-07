import json
from pathlib import Path
import run_formal_ev_vs_ccp_s30_30runs as formal
import baseline_uncertainty as base

def test_formal_preflight_contract():
    checks=formal.preflight(); assert all(checks.values())

def test_completion_requires_marker_and_matching_hashes(tmp_path):
    expected=formal.expected_meta(1,"EV",{"algorithm_seed":910001,"ccp_training_seed":920001})
    assert not formal.valid_completion(tmp_path,expected)
    formal.atomic_json(tmp_path/"configuration.json",expected)
    formal.atomic_json(tmp_path/"final_feasible_nondominated.json",[])
    formal.atomic_json(tmp_path/"COMPLETE.json",{**expected,"candidate_sha256":formal.digest(tmp_path/"final_feasible_nondominated.json"),"configuration_sha256":formal.digest(tmp_path/"configuration.json")})
    assert formal.valid_completion(tmp_path,expected)
    bad=dict(expected);bad["algorithm_seed"]=1
    assert not formal.valid_completion(tmp_path,bad)

def test_restore_individual_preserves_fingerprint():
    path=base.Path(1,"A","B",["A","B"],["road"],[],1,2,3)
    ind=base.Individual({("A","B",1):[base.PathAllocation(path,1.0)]})
    from run_ccp_candidate_pool import decision_representation, decision_signature
    row={"decision":decision_representation(ind),"decision_fingerprint":decision_signature(ind)}
    restored=formal.restore_individual(row,{("A","B",1):[path]})
    assert decision_signature(restored)==row["decision_fingerprint"]

def test_restore_exact_evolved_path_absent_from_static_library():
    nodes=["Chongqing","Chengdu","Khorgos","Almaty","Minsk","Lodz"]
    modes=["rail","rail","rail","rail","road"]
    arcs=[]; tt={}
    for origin,destination,mode in zip(nodes,nodes[1:],modes):
        arc=base.Arc(origin,destination,mode,100.0,1000.0,0.3,250.0,100.0)
        arcs.append(arc)
        if mode=="rail": tt[(origin,destination,mode)]=[object()]
    evolved=base.path_from_arcs(
        arcs,nodes[0],nodes[-1],path_id=-1,tt_dict=tt)
    original=base.Individual({
        (nodes[0],nodes[-1],12):[base.PathAllocation(evolved,1.0)]})
    from run_ccp_candidate_pool import decision_representation, decision_signature
    row={"decision":decision_representation(original),
         "decision_fingerprint":decision_signature(original)}
    stats={}

    restored=formal.restore_individual(
        row,{},tt,base.build_arc_lookup(arcs),restoration_stats=stats)

    allocation=restored.od_allocations[(nodes[0],nodes[-1],12)][0]
    assert allocation.path.nodes==nodes
    assert allocation.path.modes==modes
    assert allocation.share==1.0
    assert decision_signature(restored)==row["decision_fingerprint"]
    assert stats=={"fallback_allocations":1,"fallback_candidates":1}
