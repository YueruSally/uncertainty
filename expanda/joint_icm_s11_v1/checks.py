"""Local S11 compatibility checks. Synthetic labels and B400 runs are not results."""
import contextlib
import copy
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from argparse import Namespace
from unittest.mock import patch
from pathlib import Path
import run as s11
from joint_common import load_checkpoint
from shared_model import SharedTransformer
from train_shared import train
from network import CandidateGainTransformer


class S11Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(); cls.root = Path(cls.temp.name)
        cls.out = cls.root / 's11'; cls.model = cls.root / 'model'
        sample = s11.read_json(s11.PACKAGE / 'assets/search_instances/prescreen-search-S0-001.json')
        od = sorted({(b['origin'], b['destination']) for b in sample['batches']})
        cat = {'source_data_sha256': sample['source_data_sha256'], 'train_od_pool': od,
               'test_od_pool': [['held-out-origin', 'held-out-destination']], 'instances': []}
        cat['catalog_digest'] = s11.canonical_digest(cat)
        source = cls.root / 'source.json'; s11.write_json(source, cat)
        # Only the test fixture is shortened. Production settings stay 10x30, pop250/B50000.
        cls.cfg = {**s11.settings(), 'expected_source_catalog_digest': cat['catalog_digest'],
                   'instance_count': 1, 'algorithm_seeds': [940001],
                   'population_size': 20, 'evaluation_budget': 400, 'maximum_generations': 1000}
        with (cls.root / 'prepare.log').open('w') as h, contextlib.redirect_stdout(h), patch.object(s11, 'settings', return_value=cls.cfg):
            cls.plan = s11.prepare(cls.out, source)
        cls.name = cls.plan['instances'][0]
        rng = s11.np.random.default_rng(45); rows = {}; cls.dims = {}
        for c in s11.COMPONENTS:
            b = s11.read_json(s11.PACKAGE / 'assets' / c / 'BUNDLE.json')
            scaler = (b['models']['200'] if c == 'M' else b)['scaler']
            d = [len(scaler[k]['mean']) for k in ('batch', 'candidate', 'context')]; cls.dims[c] = d
            rows[c] = []
            for i in range(6):
                rows[c].append({'instance': 'fixture-' + str(i), 'development_role': 'train' if i < 4 else 'validation',
                    'candidate_batch_tokens': rng.normal(size=(8, 3, d[0])).tolist(),
                    'candidate_features': rng.normal(size=(8, d[1])).tolist(), 'population_points': [[.1, .2, .3]],
                    'slot_fraction': .2, 'generation': 11, 'evaluation': 250,
                    'labels': [{'hv_contribution': j / 100.} for j in range(8)]})
        from joint_common import SETTINGS
        with (cls.root / 'train.log').open('w') as h, contextlib.redirect_stdout(h):
            bundle = train(rows, cls.model, {**SETTINGS, 'epochs': 1, 'checkpoint_epochs': [1], 'batch_size': 2})
        s11.write_json(cls.out / 'SHARED_MODEL.json', {'model': str(cls.model), 'bundle_sha256': s11.digest(cls.model / 'BUNDLE.json'),
            'checkpoint_sha256': {str(cls.model / p): sha for p, sha in bundle['checkpoint_sha256'].items()}})

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_fixed_grid_and_no_duplicate_smoke_jobs(self):
        cfg = s11.settings()
        self.assertEqual((cfg['configuration'], cfg['batch_count'], cfg['deadline_window_alpha']), ('S11', 50, 2.))
        self.assertEqual((cfg['population_size'], cfg['evaluation_budget'], cfg['maximum_generations']), (250, 50000, 200))
        plan = {'methods': list(s11.experiment.METHODS), 'instances': [str(i) for i in range(10)],
                'algorithm_seeds': cfg['algorithm_seeds']}
        original = s11.jobs_for(plan, 'original'); shared = s11.jobs_for(plan, 'shared')
        self.assertEqual(len(set(original)), 2400); self.assertEqual(len(set(shared)), 300)
        self.assertFalse(set(original) & set(shared))
        self.assertEqual(s11.jobs_for(plan, 'original', True), original[:8])

    def test_instance_definition_and_no_s11_supervision(self):
        info, batches = s11.read_instance(self.out / 'instances' / (self.name + '.json'))
        self.assertEqual((info['configuration'], len(batches), info['deadline_window_alpha']), ('S11', 50, 2.))
        self.assertEqual(info['development_role'], 'frozen_transfer_development')
        self.assertEqual(s11.canonical_digest(info['batches']), info['batch_signature'])
        with self.assertRaisesRegex(ValueError, '200/20'):
            s11.shared_manifest(self.out, self.model, self.plan)

    def test_all_old_weights_and_shared_backbone_accept_50_batch_tokens(self):
        s11.torch.set_num_threads(1)
        shared = SharedTransformer(self.dims, 16).eval()
        for c in s11.COMPONENTS:
            d = self.dims[c]
            x = (s11.torch.zeros(1, 8, 50, d[0]), s11.torch.zeros(1, 8, d[1]), s11.torch.zeros(1, d[2]))
            marker = s11.torch.zeros(1, 8, 1); marker[:, 0] = 1.
            with s11.torch.no_grad():
                score, _ = shared(c, *x)
                self.assertEqual(tuple(score.shape), (1, 8)); self.assertTrue(s11.torch.isfinite(score).all())
                for f in (s11.PACKAGE / 'assets' / c).glob('*.pt'):
                    ck = load_checkpoint(f)
                    old = CandidateGainTransformer(*d, width=ck['width']).eval()
                    old.load_state_dict(ck['model'])
                    score = old(*x, marker)
                    self.assertEqual(tuple(score.shape), (1, 8)); self.assertTrue(s11.torch.isfinite(score).all())

    def test_real_50_batch_original_shared_pair_and_resume_identity(self):
        for m in ('I_C_M_K8_TRANSFORMER', s11.METHOD):
            if not (self.out / 'runs' / self.name / 'seed_940001' / m / 'RUN_COMPLETE.json').exists():
                with (self.root / (m + '.log')).open('w') as h, contextlib.redirect_stdout(h):
                    s11.run_worker(self.out, self.name, 940001, m)
            row = s11.validate_completion(self.out, self.plan, (self.name, 940001, m))
            self.assertEqual(row['real_objective_evaluation_count'], 400)
            self.assertEqual(row['combined_hidden_candidate_objective_evaluations'], 0)
        row = s11.validate_completion(self.out, self.plan, (self.name, 940001, s11.METHOD))
        for c in s11.COMPONENTS:
            self.assertGreater(row['shared_policy_counts'][c]['calls'], 0)
        file = self.out / 'runs' / self.name / 'seed_940001' / s11.METHOD / 'RUN_COMPLETE.json'
        modified = copy.deepcopy(row); modified['batch_count'] = 20; s11.write_json(file, modified)
        try:
            with self.assertRaises(ValueError):
                s11.validate_completion(self.out, self.plan, (self.name, 940001, s11.METHOD))
        finally:
            s11.write_json(file, row)

    def test_parallel_stage_locks_and_subprocess_workers(self):
        def jobs(plan, stage, smoke=False):
            return [(self.name, 940001, 'BASE' if stage == 'original' else s11.METHOD)]
        options = dict(out=self.out, workers=1, smoke=True, shared_model=self.model,
                       diagnostics=self.root, wait_hours=1)
        with (self.root / 'parallel.log').open('w') as h, contextlib.redirect_stdout(h), \
                patch.object(s11, 'ensure_shared'), patch.object(s11, 'jobs_for', side_effect=jobs):
            with ThreadPoolExecutor(max_workers=2) as pool:
                tasks = [pool.submit(s11.search_stage, Namespace(stage=stage, **options), self.plan)
                         for stage in ('original', 'shared')]
                for task in tasks:
                    task.result(timeout=180)
            # Repeating the exact stages must reuse valid completed runs.
            for stage in ('original', 'shared'):
                s11.search_stage(Namespace(stage=stage, **options), self.plan)
        for stage in ('ORIGINAL', 'SHARED'):
            status = s11.read_json(self.out / (stage + '_STATUS.json'))
            self.assertTrue(status['completed']); self.assertEqual(status['finished'], 1)
            self.assertEqual(status['running'], 0)


if __name__ == '__main__':
    unittest.main()
