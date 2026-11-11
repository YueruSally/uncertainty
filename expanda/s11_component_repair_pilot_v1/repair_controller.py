"""Capacity-coordinate I candidates using existing nominal resource tables only."""
from bootstrap import *
from collections import defaultdict
from copy import deepcopy
import random
from controller import ScreenController, timed_features
from initialization import individual_batch_tokens, candidate_proxy_features, deterministic_proxy_rows
from m_candidate_information import private_seed
import baseline_uncertainty as base


def coordinate_capacity(ind, batches, options, arc_caps, rng, restarts, iterations):
    """Preserve allocation shares; coordinate path choices with bounded min-conflicts.

    Never call an objective or scenario/path simulator. Resource tables come from
    the common Base preprocessing. Merging repeated paths is the standard repair.
    """
    variables = []
    for batch in batches:
        key = (batch.origin, batch.destination, batch.batch_id)
        choices = options.get(key, [])
        old = ind.od_allocations.get(key, [])
        if not choices or not old:
            return deepcopy(ind), None
        for alloc in old:
            share = float(alloc.share)
            resources = [{r: flow * share for r, flow in option.resources.items()} for option in choices]
            preferred = next((i for i, option in enumerate(choices) if option.path == alloc.path), None)
            variables.append((key, share, choices, resources, preferred))
    limits = {r: base._resource_available_capacity(r, arc_caps)
              for _, _, _, resources, _ in variables for table in resources for r in table}

    def excess(r, load):
        return max(0., load - limits[r])

    best = None; best_score = float('inf')
    for restart in range(restarts):
        selected = [v[4] if restart == 0 and v[4] is not None else rng.randrange(len(v[2])) for v in variables]
        loads = defaultdict(float)
        for v, index in zip(variables, selected):
            for r, flow in v[3][index].items():
                loads[r] += flow
        score = sum(excess(r, load) for r, load in loads.items())
        for step in range(iterations + 1):
            if score < best_score:
                best_score = score; best = list(selected)
            if score <= 1e-9 or step == iterations:
                break
            conflicts = {r for r, load in loads.items() if excess(r, load) > 1e-9}
            contributors = [j for j, v in enumerate(variables) if conflicts.intersection(v[3][selected[j]])]
            if not contributors:
                break
            j = rng.choice(contributors); old = variables[j][3][selected[j]]
            moves = []
            for index, proposed in enumerate(variables[j][3]):
                if index == selected[j]:
                    continue
                touched = set(old) | set(proposed)
                trial = score + sum(excess(r, loads[r] - old.get(r, 0.) + proposed.get(r, 0.))
                                    - excess(r, loads[r]) for r in touched)
                moves.append((trial, rng.random(), index))
            if not moves:
                break
            trial, _, index = min(moves)
            if trial > score + 1e-9 and rng.random() >= .04:
                break
            proposed = variables[j][3][index]
            for r in set(old) | set(proposed):
                loads[r] += proposed.get(r, 0.) - old.get(r, 0.)
            selected[j] = index
            score = sum(excess(r, load) for r, load in loads.items())
        if best_score <= 1e-9:
            break
    child = base.Individual()
    for (key, share, choices, _, _), index in zip(variables, best):
        child.od_allocations.setdefault(key, []).append(base.PathAllocation(choices[index].path, share))
    child.od_allocations = {key: base.merge_and_normalize(values) for key, values in child.od_allocations.items()}
    return child, best_score


class RepairController(ScreenController):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.capacity_construction = {'candidates': 0, 'zero_nominal_excess': 0, 'missing_options': 0}

    @timed_features
    def initial_individual_from_base(self, ind, batches, path_lib, max_paths=3, slot_index=None):
        self.slot = int(slot_index)
        if self.slot < self.i_start_slot:
            self.rule_counts['fixed_initial_slots_preserved'] += 1
            return ind
        self.ih.pool_events = self.slot; self.ih._active = None
        placeholder = self.ih.initial_individual_from_base(ind, batches, path_lib, max_paths, slot_index)
        candidates = self.ih._active['candidates']; self.ih._active = None
        caps = {(a.from_node, a.to_node, a.mode): a.capacity for a in self.arcs}
        # Candidate 0 stays the exact Base draw; both rule and learning use the
        # same seven repaired alternatives. Candidate 1 is repaired Directed.
        for index in range(1, 8):
            rng = random.Random(private_seed('s11-I-capacity-v1', self.run_id, self.slot, index))
            candidates[index], excess = coordinate_capacity(
                candidates[index], batches, self.reliable_options, caps, rng,
                self.settings['i_capacity_restarts'], self.settings['i_capacity_iterations'])
            self.capacity_construction['candidates'] += 1
            self.capacity_construction['zero_nominal_excess'] += int(excess is not None and excess <= 1e-9)
            self.capacity_construction['missing_options'] += int(excess is None)
        proxy = deterministic_proxy_rows(candidates, batches, self.arcs)
        pairs, row = self._state('I', [[x] for x in candidates], candidate_proxy_features(proxy),
                                [individual_batch_tokens(x, batches) for x in candidates], 1)
        self.pending_i = (id(placeholder), pairs, row)
        return placeholder

    def summary(self):
        return {**super().summary(), 'capacity_construction': self.capacity_construction,
                'variant': self.settings['protocol']}
