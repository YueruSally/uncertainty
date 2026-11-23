"""Decision-only ablations; no objective calls, fitting, or global RNG changes."""
import hashlib
import numpy as np


def policy_seed(instance, algorithm_seed):
    value = f'shared-sl-ablation-v1/{instance}/{algorithm_seed}'.encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:8], 'big')


class RandomReplacement:
    """Bernoulli replacement at frozen aggregate rates, uniform among legal alternatives."""
    def __init__(self, probabilities, seed):
        self.probabilities = dict(probabilities)
        if set(probabilities) != set('ICM') or any(not 0 <= v <= 1 for v in probabilities.values()):
            raise ValueError('Invalid replacement probabilities')
        # Separate streams by component, independent of the NSGA-II random stream.
        children = np.random.SeedSequence(seed).spawn(3)
        self.rng = {c: np.random.default_rng(v) for c, v in zip('ICM', children)}
        self.range_checks = {c: dict(values=0, clipped=0) for c in 'ICM'}
        self.counts = {c: dict(calls=0, requested=0, no_legal_alternative=0,
                              blocked_request=0, switches=0) for c in 'ICM'}

    def select(self, row):
        c = row['component']
        mask = np.asarray(row['legal_mask'], dtype=bool)
        if mask.shape != (8,) or not mask[0]:
            raise ValueError('Expected eight positions with legal rule at index zero')
        alternatives = np.flatnonzero(mask & (np.arange(8) != 0))
        counts = self.counts[c]
        counts['calls'] += 1
        requested = self.rng[c].random() < self.probabilities[c]
        counts['requested'] += int(requested)
        counts['no_legal_alternative'] += int(not len(alternatives))
        if not len(alternatives):
            counts['blocked_request'] += int(requested)
            return 0
        if not requested:
            return 0
        counts['switches'] += 1
        return int(self.rng[c].choice(alternatives))


def zero_threshold_policy(factory, folder):
    """Change thresholds in memory only; retain legal masks and the rule option."""
    policy = factory(folder)
    if len(policy.models) != 1:
        raise ValueError('Only the frozen single member is allowed')
    for c in 'ICM':
        gate = policy.bundle['selected']['gates'][c]
        if gate['disabled']:
            raise ValueError('This study requires the original three enabled gates')
        gate['threshold_scaled'] = 0.
    return policy
