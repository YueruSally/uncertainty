from addon_common import *
kernel = load('standard09_kernel', EXP/'standard_nsga2_alpha08_v1/kernel.py')
make_solver, dominates = kernel.make_solver, kernel.dominates
import platform
import os
import socket
import time


def search(root, plan, task):
    # All monkey-patches are confined to this dedicated subprocess and restored.
    sys.path.insert(0,str(EXP/'ppo_joint_b5000_pilot_v2'))
    p = load('standard09_pilot_engine',EXP/'ppo_joint_b5000_pilot_v2/run.py')
    p.torch.set_num_threads(1)
    e, out = task['entry'], Path(task['folder'])
    require(task in plan['tasks'] and root in out.parents, 'Unscheduled task')
    require(read(root/'instances'/(e['instance_id']+'.json')) == e, 'Instance changed')
    require(p.base.CROSSOVER_RATE==.9 and p.base.MUTATION_RATE==.15, 'Shared probability defaults differ')
    stats, holders = {}, []
    prior = {k:getattr(p.base,k) for k in ('run_nsga2','dominates')}
    def factory(*args, **kw):
        ctl = p.BudgetController(*args, settings=plan['settings'], arm='BASE', learned=None, **kw)
        holders.append(ctl); return ctl
    # Logging-only normalizer is inert in rank/crowding search; final reported
    # HV is recomputed with new union bounds after all independent evaluations.
    normpath = EXP/'formal_icm_s0_v1/assets/reward_normalization.json'
    normalizer = p.legacy_learning.normalizer(read(normpath))
    p.base.run_nsga2 = make_solver(p.base, plan['protocol'], stats)
    p.base.dominates = dominates
    begin = time.monotonic()
    try:
        r = p.run_episode(Path(task['workbook']),
            root/'instances'/(e['instance_id']+'.json'), out, None, normalizer,
            algorithm_seed=task['seed'], evaluation_budget=5000, pop_size=250, generations=1000000,
            evaluation_mode='ccp', component='BASE', scenario_count=100, ccp_confidence=.9,
            uncertainty_profile='transport_border_saa_v1', inference_controller_factories={'BASE':factory},
            method_override=METHOD)
    finally:
        for k,v in prior.items(): setattr(p.base,k,v)
    cfg = read(out/'configuration.json'); front = read(out/'final_feasible_nondominated.json')
    require(r['completed'] and r['real_objective_evaluation_count']==5000, 'Incomplete search')
    require(r['hidden_candidate_objective_evaluations']==0, 'Hidden evaluations')
    require(all(r['selection'][k]['calls']==0 for k in 'ICM'), 'ICM unexpectedly active')
    require(cfg['alpha']==.9 and cfg['training_size']==100 and
            cfg['seeds']['training']==e['ccp100_seed'], 'Incorrect alpha/scenarios')
    require(cfg['scenario_digest']==task['expected_scenario_digest'] and
            cfg['uncertainty']==task['source_uncertainty'], 'Optimization scenarios or sampler settings differ')
    cfg['standard_nsga2_protocol'] = plan['protocol']
    cfg['operator_probabilities_note'] = 'Legacy evaluator metadata; actual operators are specified in standard_nsga2_protocol and standard_kernel.'
    write(out/'configuration.json',cfg)
    inv = load('standard_front_validation',EXP/'final_front_oos_hv_v1/inventory.py')
    for row in front: inv.validate_decision(row,e)
    points = [inv.vector(row) for row in front]
    r.update(instance=e['instance_id'],configuration=e['configuration'],instance_digest=e['instance_digest'],
        seed=task['seed'],arm=METHOD,alpha=.9,phase='evaluation',stochastic=False,
        enabled_components=[],model_loaded=False,scenario_digest=cfg['scenario_digest'],
        final_front_raw=points,final_solutions_sha256=digest(out/'final_feasible_nondominated.json'),
        budget_fronts=[dict(evaluation=5000,actual_evaluations=5000,front_raw=points)],
        standard_kernel=stats,commit=plan['commit'],parent_commit=plan['parent_commit'],
        task_sha256=canonical(task),plan_sha256=plan['sha256'],
        hostname=socket.gethostname(),python=platform.python_version(),numpy=p.np.__version__,
        environment=dict(cpu_count=os.cpu_count(),load_average=list(os.getloadavg()),
            nice=os.getpriority(os.PRIO_PROCESS,0),
            threads={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')}),
        worker_wall_seconds=time.monotonic()-begin,empty_feasible_front=(not front),
        diagnostic_normalization_sha256=digest(normpath),
        diagnostic_hv_not_reported=True)
    write(out/'RUN_COMPLETE.json',r)
    seal_job(task)
