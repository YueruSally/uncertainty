"""Outcome-blind replacement-rate control, uniform over legal alternatives."""
import hashlib
import numpy as np

TARGETS = dict(I=.981, C=.433, M=.329)


def policy_seed(instance, seed):
    return int.from_bytes(hashlib.sha256(
        f'shared-sl-ablation-v2/{instance}/{seed}'.encode()).digest()[:8], 'big')


class RateControlledRandom:
    """Track the all-decision target without spending requests on illegal positions.

    On eligible call t, q_t=clip(r*t-S_(t-1),0,1). This is an adaptive
    Bernoulli controller, not independent fixed-probability replacement. If a
    call has no legal alternative, q_t=0 and the deficit is carried forward.
    Outcomes, features, labels and HV are never consulted. Finite terminal
    deficits may remain, especially when the last decisions are ineligible.
    """
    def __init__(self, targets, seed):
        if set(targets) != set('ICM') or any(not 0 <= v <= 1 for v in targets.values()):
            raise ValueError('Invalid target rates')
        self.targets = dict(targets)
        streams = np.random.SeedSequence(seed).spawn(6)
        self.coins = {c: np.random.default_rng(streams[i]) for i,c in enumerate('ICM')}
        self.choices = {c: np.random.default_rng(streams[i+3]) for i,c in enumerate('ICM')}
        self.range_checks = {c: dict(values=0, clipped=0) for c in 'ICM'}
        self.counts = {c: dict(calls=0, eligible=0, no_legal_alternative=0,
                              switches=0, fallbacks=0, conditional_probability_sum=0.,
                              saturated_probability_calls=0, max_unserved_deficit=0.) for c in 'ICM'}

    def select(self, row):
        c = row['component']
        mask = np.asarray(row['legal_mask'], dtype=bool)
        if mask.shape != (8,) or not mask[0]:
            raise ValueError('Expected eight candidate positions and legal rule at index zero')
        candidates = np.flatnonzero(mask[1:])+1
        v = self.counts[c]; v['calls'] += 1
        deficit = self.targets[c]*v['calls']-v['switches']
        action = 0
        if len(candidates):
            v['eligible'] += 1
            probability = float(np.clip(deficit, 0., 1.))
            v['conditional_probability_sum'] += probability
            v['saturated_probability_calls'] += int(deficit >= 1.)
            if self.coins[c].random() < probability:
                action = int(self.choices[c].choice(candidates))
                v['switches'] += 1
        else:
            v['no_legal_alternative'] += 1
        v['fallbacks'] += int(action == 0)
        v['max_unserved_deficit'] = max(v['max_unserved_deficit'],
                                       self.targets[c]*v['calls']-v['switches'])
        return action
