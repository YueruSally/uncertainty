"""Immutable paired networks, paths and common random realizations; no search."""
import copy
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import shutil
import time

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
SEEDS = [993001, 993002, 993003]
BUDGETS = [400, 1000, 5000]
SHARED = 'ICM_SHARED_SUPERVISED'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+'.writing')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temp.replace(path)


def freeze(path, value):
    if Path(path).exists():
        if read(path) != value:
            raise ValueError('Frozen input differs; preserve this directory: '+str(path))
    else:
        write(path, value)


def csv_write(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError('Cannot write an empty table: '+str(path))
    temp = path.with_name(path.name+'.writing')
    with temp.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    temp.replace(path)


def verify(files):
    for path, expected in files.items():
        if digest(path) != expected:
            raise ValueError('File identity differs: '+path)


def copy_frozen(source, target):
    source, target = Path(source), Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        temp = target.with_name(target.name+'.copying'); shutil.copyfile(source, temp); temp.replace(target)
    if digest(source) != digest(target):
        raise ValueError('Frozen copy differs: '+str(target))


def path_key(path):
    return tuple(path.nodes), tuple(path.modes)


def dump_paths(paths):
    return [{'origin': od[0], 'destination': od[1], 'paths': [asdict(p) for p in ps]}
            for od, ps in paths.items()]


def load_paths(base, data, arcs):
    lookup = base.build_arc_lookup(arcs); result = {}
    for row in data:
        ps = []
        for raw in row['paths']:
            d = copy.deepcopy(raw); actual = []
            for a in d.pop('arcs'):
                arc = lookup[(a['from_node'], a['to_node'], a['mode'])]
                if asdict(arc) != a:
                    raise ValueError('Cached path uses a changed arc')
                actual.append(arc)
            ps.append(base.Path(arcs=actual, **d))
        result[(row['origin'], row['destination'])] = ps
    return result


def save_scenarios(np, root, scenarios):
    travel_keys = sorted(scenarios.travel_multiplier)
    event_keys = sorted(scenarios.border_delay_h)
    meta = dict(size=scenarios.size, seed=scenarios.seed, stochastic=scenarios.stochastic,
                travel_keys=[list(k) for k in travel_keys], event_keys=[list(k) for k in event_keys],
                arc_border_event=[[list(k), list(v)] for k, v in sorted(scenarios.arc_border_event.items())],
                event_means=[[list(k), v] for k, v in sorted(scenarios.border_event_mean_h.items())])
    freeze(root/'SCENARIOS.json', meta)
    arrays = dict(travel=np.asarray([scenarios.travel_multiplier[k] for k in travel_keys]),
                  border=np.asarray([scenarios.border_delay_h[k] for k in event_keys]).reshape(-1, scenarios.size))
    dest = root/'SCENARIOS.npz'
    if dest.exists():
        with np.load(dest, allow_pickle=False) as old:
            if set(old.files) != set(arrays) or any(not np.array_equal(old[k], v) for k, v in arrays.items()):
                raise ValueError('Frozen random realizations differ: '+str(dest))
    else:
        temp = dest.with_name('SCENARIOS.writing.npz'); np.savez_compressed(temp, **arrays); temp.replace(dest)


def load_scenarios(base, np, root):
    q = read(root/'SCENARIOS.json')
    with np.load(root/'SCENARIOS.npz', allow_pickle=False) as a:
        travel = {tuple(k): a['travel'][i].copy() for i, k in enumerate(q['travel_keys'])}
        border = {tuple(k): a['border'][i].copy() for i, k in enumerate(q['event_keys'])}
    for array in list(travel.values())+list(border.values()):
        array.setflags(write=False)
    return base.ScenarioSet(size=q['size'], seed=q['seed'], stochastic=q['stochastic'],
        travel_multiplier=travel, border_delay_h=border,
        arc_border_event={tuple(k): tuple(v) for k, v in q['arc_border_event']},
        border_event_mean_h={tuple(k): v for k, v in q['event_means']})


def scenario_kwargs(profile):
    return {k: profile[k] for k in ('mode_time_cv', 'mode_time_max_factor', 'border_delay_cv',
                                   'border_delay_max_factor', 'border_mean_source')}


def network(base, workbook, profile):
    data = base.load_network_from_extended(str(workbook))
    delay = base.load_border_delay_map_from_workbook(workbook, include_customs=profile['border_include_customs'])
    return data, delay


def prepare_inputs(root, base, np, profile):
    """All four gates must pass before PLAN.json can be created."""
    started = time.monotonic(); assets = HERE/'assets'
    manifest = read(assets/'ALIGNMENT.json')
    big_source = EXP/'formal_icm_s0_v1/assets/data/data_expanded.xlsx'
    verify({str(assets/'original_small.xlsx'): manifest['source_file_sha256'],
            str(assets/'small_aligned.xlsx'): manifest['aligned_file_sha256'],
            str(big_source): manifest['expanded_sha256']})
    books = {n: root/'inputs/networks'/f'{n}.xlsx' for n in ('small', 'large')}
    copy_frozen(assets/'small_aligned.xlsx', books['small']); copy_frozen(big_source, books['large'])
    base.BORDER_EVENT_DEFINITIONS = base.load_border_event_definitions(base.DEFAULT_BORDER_EVENT_DATA_FILE)
    original, _ = network(base, assets/'original_small.xlsx', profile)
    small, sd = network(base, books['small'], profile)
    small_caps, small_bg = dict(base.BORDER_CAPACITY), dict(base.BACKGROUND_FLOW)
    large, ld = network(base, books['large'], profile)
    bg = {k: base.BACKGROUND_FLOW.get(k, 0) for k in small_bg}
    if (len(set(small[0])), len(small[5]), len(set(large[0])), len(large[5])) != (38,708,54,978):
        raise ValueError('Unexpected effective network sizes')
    if small_bg != bg or any(small_bg.get(k) != v for k, v in manifest['background_flow_TEUday'].items()):
        raise ValueError('Shared background flows differ')
    if any(base.BORDER_CAPACITY.get(k) != v for k, v in small_caps.items()):
        raise ValueError('Shared border capacities differ')
    A = {(a.from_node,a.to_node,a.mode):asdict(a) for a in small[5]}
    B = {(a.from_node,a.to_node,a.mode):asdict(a) for a in large[5]}
    O = {(a.from_node,a.to_node,a.mode):asdict(a) for a in original[5]}
    if A != O or any(B.get(k) != v for k,v in A.items()):
        raise ValueError('Shared arc parameters differ')
    big_tt = {(x.from_node,x.to_node,x.mode):asdict(x) for x in large[6]}
    if any(big_tt.get((x.from_node,x.to_node,x.mode)) != asdict(x) for x in small[6]):
        raise ValueError('Shared timetables differ')
    if any(large[13].get(k) != v for k,v in small[13].items()):
        raise ValueError('Shared transfer parameters differ')
    # Keep the user's fixed 30-batch pool: membership in the ORIGINAL Nodes sheet,
    # not the two port nodes auto-added by the engine.
    listed = set(original[0])-{'Shanghai','Ningbo'}
    pool = sorted([copy.deepcopy(b) for b in large[7]
                   if b.origin in listed and b.destination in listed], key=lambda b:b.batch_id)
    if len(pool) != 30 or [b.batch_id for b in pool[:10]] != list(range(1,11)):
        raise ValueError('Expected original 10 plus 20 fixed common-OD batches')
    if [asdict(b) for b in pool[:10]] != [asdict(b) for b in original[7]]:
        raise ValueError('Original ten batch records differ')
    freeze(root/'inputs/BATCH_POOL.json', dict(batches=[asdict(b) for b in pool],
        selection='Ascending original BatchID; nested first 10 / first 20 / all 30; no redraw',
        alpha=1.0, quantities_and_ET_LT='Workbook values; unchanged between networks'))
    print('[NETWORK] check 1/4 passed: aligned 38/708 and 54/978 networks', flush=True)
    entries=[]; path_rows=[]; overlap_rows=[]; feasibility=[]; scenario_checks=[]
    for count in (10,20,30):
        cfg=f'B{count}'; pair=f'network-{cfg}-001'; batch_list=pool[:count]
        path_seed=371000000+count; scenario_seed=372000000+count
        # The only stochastic sampling call is on the large network.
        large, ld = network(base, books['large'], profile)
        master=base.build_scenario_set(large[5],ld,100,scenario_seed,stochastic=True,
            border_event_definitions=base.BORDER_EVENT_DEFINITIONS,**scenario_kwargs(profile))
        paths_by_net={}
        for net in ('small','large'):
            data, delay=network(base,books[net],profile)
            nodes, regions, _, _, _, arcs, tt, *_=data
            expected=base.build_scenario_set(arcs,delay,1,scenario_seed,stochastic=False,
                border_event_definitions=base.BORDER_EVENT_DEFINITIONS,**scenario_kwargs(profile))
            keys=sorted({(a.from_node,a.to_node,a.mode) for a in arcs})
            if any(master.arc_border_event.get(k) != expected.arc_border_event.get(k) for k in keys):
                raise ValueError('Common arc has a different crossing-event mapping')
            if any(master.border_event_mean_h.get(k) != v for k,v in expected.border_event_mean_h.items()):
                raise ValueError('Common crossing-event means differ')
            scenarios=base.ScenarioSet(size=100,seed=scenario_seed,
                travel_multiplier={k:master.travel_multiplier[k].copy() for k in keys},
                border_delay_h={k:master.border_delay_h[k].copy() for k in expected.border_delay_h},
                arc_border_event=expected.arc_border_event, border_event_mean_h=expected.border_event_mean_h,
                stochastic=True)
            folder=root/'inputs'/pair/net;folder.mkdir(parents=True,exist_ok=True)
            save_scenarios(np,folder,scenarios)
            saved=load_scenarios(base,np,folder)
            if any(not np.array_equal(saved.travel_multiplier[k],master.travel_multiplier[k]) for k in keys):
                raise ValueError('Shared arc draws differ after serialization')
            if any(not np.array_equal(v,master.border_delay_h[k]) for k,v in saved.border_delay_h.items()):
                raise ValueError('Shared crossing draws differ after serialization')
            scenario_checks.append(dict(configuration=cfg,network=net,arcs=len(keys),
                border_events=len(saved.border_delay_h),realizations=100,all_values_exactly_equal=True))
            timetable=base.build_timetable_dict(tt);lookup=base.build_arc_lookup(arcs)
            path_file=folder/'PATHS.json'
            if path_file.exists():
                paths=load_paths(base,read(path_file),arcs)
            else:
                random.seed(path_seed);np.random.seed(path_seed)
                paths=base.build_path_library(nodes,regions,arcs,batch_list,timetable,lookup)
                base.sanity_check_path_lib(batch_list,paths)
                freeze(path_file,dump_paths(paths))
                paths=load_paths(base,read(path_file),arcs)
            base.sanity_check_path_lib(batch_list,paths)
            paths_by_net[net]=paths
            # Same pre-screening as deployment (expected-input timetable checks).
            ev=base.build_expected_value_scenario_set(arcs,delay,seed=0,
                border_event_definitions=base.BORDER_EVENT_DEFINITIONS,border_mean_source=profile['border_mean_source'])
            base.ACTIVE_SCENARIO_SET=ev;base._PATH_SCENARIO_CACHE={}
            options=base.build_reliable_path_options(batch_list,paths,timetable,data[13],delay,ev,mode='ev')
            for b in batch_list:
                n=len(options.get((b.origin,b.destination,b.batch_id),[]))
                feasibility.append(dict(configuration=cfg,network=net,batch_id=b.batch_id,origin=b.origin,
                    destination=b.destination,quantity_TEU=b.quantity,ET=b.ET,LT=b.LT,valid_paths=n))
                if not n:
                    csv_write(root/'BATCH_PATH_FEASIBILITY.csv',feasibility)
                    raise ValueError(f'No timetable-valid path: {cfg}/{net}/batch {b.batch_id}; no search started')
            option_ids={str(b.batch_id):[x.path.path_id for x in options[b.origin,b.destination,b.batch_id]] for b in batch_list}
            freeze(folder/'OPTION_PATH_IDS.json',option_ids)
            for od, ps in paths.items():
                path_rows.append(dict(configuration=cfg,network=net,origin=od[0],destination=od[1],paths=len(ps),
                    path_seed=path_seed,path_file_sha256=digest(path_file)))
            entry=dict(schema_version=1,instance_id=pair+'-'+net,pair_id=pair,configuration=cfg,network=net,
                K=count,deadline_window_alpha=1.0,split='topology_transfer',development_role='screen',
                path_seed=path_seed,ccp100_seed=scenario_seed,algorithm_seed=SEEDS[0],
                source_data_sha256=digest(books[net]),batches=[asdict(b) for b in batch_list],
                workbook=str(books[net]),input_folder=str(folder),path_library_sha256=digest(path_file),
                scenarios_sha256=digest(folder/'SCENARIOS.npz'),scenario_metadata_sha256=digest(folder/'SCENARIOS.json'),
                batch_signature=canonical([asdict(b) for b in batch_list]))
            entry['instance_digest']=canonical(entry);freeze(root/'instances'/(entry['instance_id']+'.json'),entry)
            entries.append(entry)
        for od, ps in paths_by_net['small'].items():
            small_keys={path_key(x) for x in ps};big_keys={path_key(x) for x in paths_by_net['large'][od]}
            overlap_rows.append(dict(configuration=cfg,origin=od[0],destination=od[1],small_paths=len(small_keys),
                large_paths=len(big_keys),shared_paths=len(small_keys&big_keys),
                retained_small_fraction=len(small_keys&big_keys)/len(small_keys)))
        print(f'[NETWORK] {cfg}: common realizations exact; paths frozen; all {count} batches passed in both networks',flush=True)
    csv_write(root/'PATH_LIBRARY.csv',path_rows);csv_write(root/'PATH_OVERLAP.csv',overlap_rows)
    csv_write(root/'BATCH_PATH_FEASIBILITY.csv',feasibility);csv_write(root/'COMMON_SCENARIOS.csv',scenario_checks)
    audit=dict(completed=True,checks=dict(aligned_network=True,common_realizations_exact=True,
        path_libraries_frozen=True,all_batches_have_timetable_valid_paths=True),
        network_sizes={'small':{'nodes':38,'arcs':708},'large':{'nodes':54,'arcs':978}},
        background_flow=manifest['background_flow_TEUday'],instances=3,network_instance_inputs=6,
        searched_network_methods=['small/BASE','small/'+SHARED,'large/'+SHARED],
        searches=27,training_updates=0,preparation_seconds=time.monotonic()-started,
        feasibility_scope='Structural/timetable-valid paths per batch. This does not guarantee jointly capacity-feasible allocations.',
        path_scope='Unchanged randomized DFS / top-k rules; counts describe retained libraries, not all graph paths.',
        cached_preparation_time_reported_separately=True)
    write(root/'PREFLIGHT.json',audit)
    return entries
