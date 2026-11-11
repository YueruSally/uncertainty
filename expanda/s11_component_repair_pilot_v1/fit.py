"""New I/M supervision, unchanged signed labels, immutable C checkpoint reuse."""
from bootstrap import *
import numpy as np
import torch

ORIGINAL_LOSS = learning.loss


def informative_weighted_loss(pred, target, mask, regression_weight=.5):
    legal = mask.clone(); legal[:, 0] = False
    informative = ((target.abs() > 1e-6) & legal).any(dim=1)
    if informative.any() and (~informative).any():
        # Negative-only examples also belong to the informative group. All
        # states are retained; validation remains unweighted and instance-equal.
        return (.75 * ORIGINAL_LOSS(pred[informative], target[informative], mask[informative], regression_weight)
                + .25 * ORIGINAL_LOSS(pred[~informative], target[~informative], mask[~informative], regression_weight))
    return ORIGINAL_LOSS(pred, target, mask, regression_weight)


def collections(root, cat, component):
    return [(root/'collection'/r['instance_id']/('COLLECT_'+component), r['development_role'])
            for r in cat['instances'] if r['development_role'] in ('training', 'validation')]


def signal_check(root, settings, cat):
    folders = collections(root, cat, 'I')
    norm = learning.fit_normalization([p for p, role in folders if role == 'training'])
    records = []
    for folder, role in folders:
        data = read_json(folder/'DATA.json'); rows = data['rows']
        targets = np.stack([learning.labels(row, norm) for row in rows])
        masks = np.asarray([row['legal_mask'] for row in rows], bool)
        children = [x for row in rows for legal, pair in zip(row['legal_mask'], row['labels']) if legal for x in pair]
        records.append({'instance':data['instance'], 'role':role, 'states':len(rows),
                        'legal_candidate_children':len(children), 'feasible_children':sum(x['feasible'] for x in children),
                        'positive_opportunity_states':int(np.sum(np.max(np.where(masks, targets, -np.inf), axis=1) > 1e-12))})
    positive = sum(r['positive_opportunity_states'] > 0 for r in records if r['role'] == 'training')
    report = {'records':records, 'training_positive_instances':positive,
              'required_training_positive_instances':settings['i_min_positive_training_instances'],
              'pass':positive >= settings['i_min_positive_training_instances'],
              'qualification':'Training-only preliminary normalization. Validation is descriptive, not a pass condition; no outcome-driven resampling.'}
    freeze(root/'I_SIGNAL_CHECK.json', report)
    print('[S11-REPAIR] I signal check ' + str(report), flush=True)
    return report['pass']


def train(root, settings, cat, sha):
    if (root/'models/FROZEN.json').exists():
        legacy_run.verify_frozen(root); return
    data = []
    for c in 'IM':
        for folder, role in collections(root, cat, c):
            run = legacy_run.completed(folder, sha)
            if not run:
                raise ValueError('Missing collection: '+str(folder))
            d = read_json(folder/'DATA.json')
            if d['role'] != role or not all(d['phase_counts'][c]):
                raise ValueError('Missing sampling phase or wrong role')
            if d['extra_evaluations'][c] > settings['label_caps'][c]:
                raise ValueError('Label cap exceeded')
            data.append((folder, d))
    norm = learning.fit_normalization([p for p,d in data if d['role']=='training'])
    freeze(root/'models/HV_NORMALIZATION.json', norm)
    for c in 'IM':
        tr = [r for p,d in data if d['role']=='training' for r in d['rows'] if r['component']==c]
        va = [r for p,d in data if d['role']=='validation' for r in d['rows'] if r['component']==c]
        s = dict(settings)
        if c == 'M':
            s['loss'] = 'per-minibatch 0.75 informative / 0.25 tied states; each term weighted_pairwise_softplus + 0.5 smooth_l1'
        prior = learning.loss
        try:
            learning.loss = informative_weighted_loss if c=='M' else ORIGINAL_LOSS
            learning.train_component(c, tr, va, norm, s, root/'models'/c)
        finally:
            learning.loss = prior
    assets = [root/'RULE_SELECTION.json', root/'instances/catalog.json', root/'I_SIGNAL_CHECK.json',
              root/'SOURCE.json', root/'models/HV_NORMALIZATION.json'] + [p/'DATA.json' for p,d in data]
    assets += [p for p in (root/'models').rglob('*') if p.is_file() and p.name!='FROZEN.json']
    freeze(root/'models/FROZEN.json', {'plan_sha256':sha, 'files':{str(p.relative_to(root)):digest(p) for p in assets},
                                    'no_ppo':True, 'screen_data_used_for_training':False, 'C_retrained':False})
