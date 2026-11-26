"""Deb et al. (2002) NSGA-II framework with explicit transport encoding.

The paper does not define operators for split multimodal transport plans.
Here a gene is one batch's complete path/share allocation: uniform block
crossover and random path/share mutation. No objective-guided operators.
"""
from copy import deepcopy
import math
import random
import time


def dominates(a, b):
    """Deb constrained domination; equal infeasible violation does not dominate."""
    if bool(a.feasible) != bool(b.feasible):
        return bool(a.feasible)
    if not a.feasible:
        return a.normalized_violation < b.normalized_violation
    return (all(x <= y for x, y in zip(a.objectives, b.objectives))
            and any(x < y for x, y in zip(a.objectives, b.objectives)))


def tournament(pop):
    a, b = random.sample(pop, 2)
    ka, kb = (a.rank, -a.crowding_distance), (b.rank, -b.crowding_distance)
    return a if ka < kb else b if kb < ka else random.choice((a, b))


def allocation(base, paths, max_paths):
    count = random.randint(1, min(max_paths, len(paths)))
    chosen = random.sample(paths, count)
    # Positive random shares; normalisation is representation validity only.
    weights = [max(random.random(), 1e-15) for _ in chosen]
    total = sum(weights)
    return [base.PathAllocation(p, w / total) for p, w in zip(chosen, weights)]


def check_genes(ind, batches, pools):
    expected = {(b.origin, b.destination, b.batch_id) for b in batches}
    if set(ind.od_allocations) != expected:
        raise ValueError('Incomplete chromosome')
    for key, genes in ind.od_allocations.items():
        if not genes or not math.isclose(sum(g.share for g in genes), 1., abs_tol=1e-12):
            raise ValueError('Shares do not sum to one')
        if any(g.share <= 0 or g.path not in pools[key[:2]] for g in genes):
            raise ValueError('Invalid path/share')


def mutate(base, ind, batch, pool):
    """Elementary, non-evaluating mutations; no cap on the number of paths.

    The maximum of three paths is an INITIALIZATION setting, not a feasible-set
    restriction. Add/delete/replace/share perturbation each has probability 1/4.
    Unavailable operations are no-ops, with no performance-based resampling.
    """
    key = (batch.origin, batch.destination, batch.batch_id)
    genes = ind.od_allocations[key]
    op = random.randrange(4)
    if op == 0:
        unused = [p for p in pool if all(p != g.path for g in genes)]
        if unused: genes.append(base.PathAllocation(random.choice(unused), max(random.random(),1e-15)))
    elif op == 1:
        if len(genes)>1: genes.pop(random.randrange(len(genes)))
    elif op == 2:
        i = random.randrange(len(genes))
        genes[i] = base.PathAllocation(random.choice(pool),genes[i].share)
    else:
        random.choice(genes).share *= random.uniform(.5,1.5)
    ind.od_allocations[key] = base.merge_and_normalize(genes)
    return op


def make_solver(base, protocol, stats):
    """Install in one worker process only; no source files/defaults are changed."""
    def solve(node_names, node_region, node_hold_cost, node_proc_cost, node_trans_cost,
              arcs, timetables, batches, waiting_cost_per_teu_h, wait_emis_g_per_teu_h,
              carbon_tax_map, emission_factor_map, mode_speeds_map, trans_map,
              border_delay_map, theta_rm, path_lib, reliable_options,
              pop_size=250, generations=200, archive_size=None):
        ctl = base.ACTIVE_MUTATION_LOGGER
        if ctl is None or any(getattr(ctl, 'controls_' + c) for c in
                              ('initialization', 'crossover', 'mutation')):
            raise ValueError('Standard comparator requires passive budget controller')
        if (pop_size, ctl.budget) != (protocol['population'], protocol['budget']):
            raise ValueError('Search settings differ')
        if any(not path_lib.get((b.origin, b.destination)) for b in batches):
            raise ValueError('Batch has no path')
        kwargs = dict(node_hold_cost=node_hold_cost, node_proc_cost=node_proc_cost,
                      carbon_tax_map=carbon_tax_map, trans_map=trans_map,
                      border_delay_map=border_delay_map, theta_rm=theta_rm,
                      node_trans_cost=node_trans_cost)
        tt = base.build_timetable_dict(timetables)
        started = time.perf_counter()
        stats.update(initial_random=0, offspring_evaluated=0, crossover_pairs=0,
                     mutations=0, feasibility_boosts=0, heuristic_calls=0,
                     generations=0, feasible_by_generation=[],mutation_operations=[0,0,0,0])

        def evaluate(ind):
            check_genes(ind, batches, path_lib)
            # Count every generated individual once, including copied offspring.
            # Cached inherited objective values never bypass a plan evaluation.
            if hasattr(ind, '_learning_eval_key'):
                del ind._learning_eval_key
            before = ctl.evaluations
            base.evaluate_individual(ind, batches, arcs, tt, waiting_cost_per_teu_h,
                                     wait_emis_g_per_teu_h, **kwargs)
            if ctl.evaluations != before + 1:
                raise ValueError('Each generated individual must consume one evaluation')

        pop = []
        for _ in range(pop_size):
            ind = base.Individual()
            for b in batches:
                key = (b.origin, b.destination, b.batch_id)
                ind.od_allocations[key] = allocation(base, path_lib[key[:2]], protocol['max_initial_paths'])
            evaluate(ind); pop.append(ind); stats['initial_random'] += 1
        for front in base.fast_non_dominated_sort(pop):
            base.crowding_distance_assignment(pop, front)
        ctl.log_initial_population(pop)
        for gen in range(generations):
            if not ctl.has_budget(1):
                break
            ctl.generation, ctl.phase = gen, 'offspring'
            offspring = []
            while len(offspring) < pop_size and ctl.has_budget(1):
                p1, p2 = tournament(pop), tournament(pop)
                if random.random() < protocol['crossover_probability']:
                    children = [base.Individual(), base.Individual()]
                    for b in batches:
                        key = (b.origin, b.destination, b.batch_id)
                        x, y = (p1, p2) if random.random() < .5 else (p2, p1)
                        children[0].od_allocations[key] = deepcopy(x.od_allocations[key])
                        children[1].od_allocations[key] = deepcopy(y.od_allocations[key])
                    stats['crossover_pairs'] += 1
                else:
                    children = [deepcopy(p1), deepcopy(p2)]
                for child in children:
                    if len(offspring) == pop_size or not ctl.has_budget(1):
                        break
                    if random.random() < protocol['mutation_probability']:
                        b = random.choice(batches)
                        key = (b.origin, b.destination, b.batch_id)
                        op = mutate(base, child, b, path_lib[key[:2]])
                        stats['mutation_operations'][op] += 1
                        stats['mutations'] += 1
                    evaluate(child); offspring.append(child)
                    stats['offspring_evaluated'] += 1
            pop = base.nsga2_environmental_selection(pop + offspring, pop_size)
            ctl.flush_events(pop); ctl.log_generation(pop, 0)
            stats['generations'] += 1
            stats['feasible_by_generation'].append(sum(x.feasible for x in pop))
        for front in base.fast_non_dominated_sort(pop):
            base.crowding_distance_assignment(pop, front)
        final = [x for x in pop if x.rank == 0 and x.feasible]
        stats['actual_evaluations'] = ctl.evaluations
        stats['final_feasible_solutions'] = len(final)
        if ctl.evaluations != protocol['budget']:
            raise ValueError('Budget was not exhausted')
        return pop, final, [], [], [], {}, [], [], [], [], time.perf_counter() - started
    return solve
