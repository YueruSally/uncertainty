"""Diagnostic invariants and a real checkpoint/data replay without search."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import audit as A
import numpy as np
import torch


class DiagnosticChecks(unittest.TestCase):
    def test_signed_labels_masked_oracle_and_disabled_gate(self):
        rows = [dict(configuration='S11', instance='a', legal_mask=[True]*7+[False])]
        y = np.array([[0., -.1, .2, 0., 0., 0., 0., 999.]])
        stats = A.label_summary(rows, y, 1.)
        self.assertEqual(stats['positive_alternatives'], 1)
        self.assertEqual(stats['negative_alternatives'], 1)
        self.assertEqual(stats['oracle_mean_residual_hv'], .2)
        pred = np.stack([y]*3)
        gate = dict(disabled=True, **{'lambda': 0., 'threshold_scaled': 0.})
        disabled = A.decision_summary(rows, y, pred, 1., gate)
        self.assertEqual(disabled['switches'], 0)
        self.assertEqual(disabled['mean_residual_hv'], 0.)
        gate['disabled'] = False
        enabled = A.decision_summary(rows, y, pred, 1., gate)
        self.assertEqual(enabled['mean_residual_hv'], .2)

    def test_aggregation_equal_instances_not_state_weighted(self):
        rows = [dict(configuration='S11', instance=x) for x in ('a', 'a', 'a', 'b')]
        self.assertEqual(A.equal_mean(rows, [1, 1, 1, -1]), 0.)

    def test_reference_coverage_counts_both_c_children(self):
        norm = dict(minimum=[0.]*3, maximum=[1.]*3, reference=[1.1]*3)
        row = dict(divisor=[1.]*3, legal_mask=[True, False], labels=[
            [dict(feasible=True, objectives=[2., .2, .2]), dict(feasible=False, objectives=None)],
            [dict(feasible=True, objectives=[.2]*3)]])
        self.assertEqual(A.objective_coverage([row], norm), dict(
            legal_candidate_children=2, feasible_children=1, feasible_children_outside_hv_reference=1))

    def test_full_frozen_replay_and_no_source_writes(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory(prefix='s11-audit-check-') as tmp:
            source = Path(tmp) / 'source'; source.mkdir()
            settings = dict(checkpoint_epochs=[1, 2], training_seeds=[11, 12, 13])
            code = ['s11_targeted_v1/learning.py', 's11_targeted_v1/common.py',
                    'formal_icm_s0_v1/src/network.py', 'formal_icm_s0_v1/src/support/transformer_ppo/archive.py']
            plan = dict(sha256='fixture', protocol=settings,
                        files={p: A.L.digest(A.HERE.parent / p) for p in code})
            A.L.write_json(source / 'PLAN.json', plan)
            catalog = dict(instances=[dict(instance_id=name, development_role=role) for name, role in
                           [('t1', 'training'), ('t2', 'training'), ('v1', 'validation')]])
            A.L.write_json(source / 'instances/catalog.json', catalog)
            norm = dict(minimum=[0.]*3, maximum=[1.]*3, reference=[1.1]*3)
            for component in 'ICM':
                rows_by_role = dict(training=[], validation=[])
                for item in catalog['instances']:
                    rows = []
                    for phase in range(3):
                        rng = np.random.default_rng(phase + 100)
                        row = dict(component=component, instance=item['instance_id'], configuration='S11',
                            phase=phase, divisor=[1.]*3, population_raw=[[.9, .9, .9]], progress=.1*phase,
                            initialization_progress=.1*phase, population_fraction=1., legal_mask=[True]*8,
                            candidate_batch_tokens=rng.normal(size=(8, 2, 3)).tolist(),
                            candidate_features=rng.normal(size=(8, 2)).tolist(),
                            labels=[[dict(feasible=True, objectives=[.7-.03*k, .6, .6])]
                                    for k in range(8)])
                        if component == 'C':
                            row['labels'] = [pair + [dict(feasible=False, objectives=None)] for pair in row['labels']]
                        rows.append(row)
                    folder = source / 'collection' / item['instance_id'] / ('COLLECT_' + component)
                    A.L.write_json(folder / 'DATA.json', dict(role=item['development_role'],
                                   instance=item['instance_id'], rows=rows))
                    rows_by_role[item['development_role']].extend(rows)
                train = rows_by_role['training']; valid = rows_by_role['validation']
                scaler = A.L.scaler(train, norm)
                target = np.stack([A.L.labels(r, norm) for r in train])
                valid_target = np.stack([A.L.labels(r, norm) for r in valid])
                scale = max(float(target.std()), 1e-9)
                folder = source / 'models' / component; folder.mkdir(parents=True)
                choices = []; history = []
                for epoch in settings['checkpoint_epochs']:
                    models = []
                    for seed in settings['training_seeds']:
                        torch.manual_seed(seed + epoch)
                        model = A.L.CandidateGainTransformer(3, 2, 15, width=8)
                        torch.save(dict(model_state=model.state_dict(), dimensions=(3, 2, 15), width=8),
                                   folder / f'seed_{seed}_epoch_{epoch}.pt')
                        models.append(model)
                        with torch.inference_mode():
                            value = A.L.loss(model(*A.L.pack(train, scaler, norm)),
                                            torch.tensor(target/scale, dtype=torch.float32),
                                            torch.tensor([r['legal_mask'] for r in train]))
                        history.append(dict(seed=seed, epoch=epoch, training_loss=float(value)))
                    pred = A.L.predict(models, valid, scaler, norm, scale)
                    for lam in (0., 1.):
                        score = A.L.validation_score(valid, pred, norm, lam, .02*scale, targets=valid_target)
                        choices.append(dict(epoch=epoch, disabled=False, threshold_scaled=.02, **{'lambda': lam}, **score))
                choices.append(dict(epoch=1, disabled=True, threshold_scaled=0., mean_gain=0.,
                                    switch_fraction=0., per_configuration={'S11': 0.}, **{'lambda': 0.}))
                gate = min(choices, key=lambda x: (-x['mean_gain'], x['switch_fraction'], x['epoch'], x['lambda'], x['threshold_scaled']))
                A.L.write_json(folder / 'VALIDATION.json', dict(choices=choices, history=history, selected=gate))
                A.L.write_json(folder / 'BUNDLE.json', dict(gate=gate, normalization=norm, scaler=scaler, scale=scale,
                    loss='weighted_pairwise_softplus + 0.5 * masked_smooth_l1_scaled_signed_residual',
                    training_instances=['t1', 't2'], validation_instances=['v1'], training_states=6, validation_states=3))
            manifest = dict(plan_sha256='fixture', files={str(p.relative_to(source)): A.L.digest(p)
                            for p in source.rglob('*') if p.is_file()})
            A.L.write_json(source / 'models/FROZEN.json', manifest)
            before = {str(p): A.L.digest(p) for p in source.rglob('*') if p.is_file()}
            with contextlib.redirect_stdout(io.StringIO()), patch.object(A.L, 'train_component', side_effect=AssertionError('must not train')):
                result = A.execute(source, Path(tmp) / 'audit')
            after = {str(p): A.L.digest(p) for p in source.rglob('*') if p.is_file()}
            self.assertEqual(before, after)
            self.assertEqual(result['new_objective_evaluations'], 0)
            self.assertEqual(result['new_training_steps'], 0)
            for value in result['components'].values():
                self.assertEqual(value['labels']['training']['independent_instances'], 2)
                for epoch in value['epochs']:
                    self.assertLess(epoch['original_validation_gain_max_abs_replay_error'], 1e-12)
                    self.assertEqual(epoch['original_validation_switch_fraction_max_abs_replay_error'], 0.)
            (source / 'models/I/BUNDLE.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'Frozen input changed'):
                A.verify_source(source)


if __name__ == '__main__':
    unittest.main()
