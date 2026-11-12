"""Synthetic training plus actual B400 rollout; no formal performance claims."""
from common import *
import contextlib
import copy
import io
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from shared_model import targets
from shared_policy import SharedPolicy
from train_shared import train
from terminal_ppo import ppo_update, records, snapshot
import run as workflow
from results import comparison_tables


class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(); cls.root = Path(cls.temp.name)
        cls.model = cls.root / 'supervised'; rng = np.random.default_rng(55); rows = {}
        for c in COMPONENTS:
            b = read_json(PACKAGE / 'assets' / c / 'BUNDLE.json')
            s = (b['models']['200'] if c == 'M' else b)['scaler']
            d = [len(s[k]['mean']) for k in ('batch', 'candidate', 'context')]
            rows[c] = [{'instance': ('train' if i < 4 else 'valid') + str(i),
                'development_role': 'train' if i < 4 else 'validation',
                'candidate_batch_tokens': rng.normal(size=(8, 3, d[0])).tolist(),
                'candidate_features': rng.normal(size=(8, d[1])).tolist(),
                'population_points': [[.1, .2, .3]], 'slot_fraction': .2, 'generation': 11,
                'evaluation': 220, 'labels': [{'hv_contribution': j / 100} for j in range(8)]} for i in range(6)]
        cls.rows = rows
        torch.set_num_threads(1)
        with contextlib.redirect_stdout(io.StringIO()):
            train(rows, cls.model, {**SETTINGS, 'epochs': 1, 'checkpoint_epochs': [1], 'batch_size': 2})

    @classmethod
    def tearDownClass(cls): cls.temp.cleanup()

    def test_fixed_budget_and_update_zero_can_win(self):
        cfg = SETTINGS_B400['ppo']
        self.assertEqual(cfg['population'], 20); self.assertEqual(cfg['budget'], 400)
        self.assertEqual((cfg['updates'] * cfg['episodes_per_update'] + 20 * 3 + 30 * 2) * 400, 60800)
        self.assertEqual(choose_checkpoint([{'update': 0, 'mean_difference_vs_initial': 0},
            {'update': 4, 'mean_difference_vs_initial': -.1}, {'update': 8, 'mean_difference_vs_initial': 0}])['update'], 0)

    def test_split_and_changed_normalization_rejected(self):
        b = read_json(self.model / 'BUNDLE.json')
        b['training_instances'] = {c: ['t' + str(i) for i in range(200)] for c in COMPONENTS}
        b['validation_instances'] = {c: ['v' + str(i) for i in range(20)] for c in COMPONENTS}
        check_split(b)
        broken = copy.deepcopy(b); broken['validation_instances']['C'][0] = 't0'
        with self.assertRaises(ValueError): check_split(broken)
        b['normalization_sha256'] = 'wrong'
        with self.assertRaises(ValueError): check_split(b)

    def test_real_worker_all_actions_ppo_and_resume(self):
        n = 'prescreen-search-S0-001'; folder = self.root / 'search'; seed = 930001
        p = {'sha256': 'fixture', 'runtime_files': {}, 'instances': [n], 'train_instances': [],
             'validation_instances': [], 'instances_root': ''}
        task = {'instance': n, 'seed': seed, 'method': PPO, 'model': str(self.model),
                'folder': str(folder), 'sample': False, 'role': 'screen', 'reference': None}
        taskfile = self.root / 'TASK.json'; write_json(taskfile, task)
        with (self.root / 'search.log').open('w') as log, contextlib.redirect_stdout(log):
            workflow.worker(SimpleNamespace(out=self.root, worker='search', task=taskfile), p)
        r = read_json(folder / 'RUN_COMPLETE.json')
        self.assertEqual(r['real_objective_evaluation_count'], 400)
        self.assertEqual(r['strategy'], PPO); self.assertEqual(r['combined_hidden_candidate_objective_evaluations'], 0)
        for c in COMPONENTS: self.assertGreater(r['shared_policy_counts'][c]['calls'], 0)
        before = digest(folder / 'RUN_COMPLETE.json')
        workflow.rollout(self.root, p, n, seed, self.model, folder, PPO)
        self.assertEqual(before, digest(folder / 'RUN_COMPLETE.json'))
        with self.assertRaises(ValueError): workflow.rollout(self.root, p, n, seed + 1, self.model, folder, PPO)
        # A real stochastic full episode, followed by one genuine terminal PPO update.
        stochastic = self.root / 'stochastic'
        import shared_search
        with (self.root / 'rollout.log').open('w') as log, contextlib.redirect_stdout(log):
            r = shared_search.run_one(PACKAGE / 'assets/search_instances' / (n + '.json'), seed,
                self.model, stochastic, 'fixture', 400, 20, 1000, True, True)
        events = records(stochastic, r['final_hypervolume'])
        self.assertEqual({e['component'] for e in events}, set(COMPONENTS))
        self.assertTrue(all(e['return'] == r['final_hypervolume'] for e in events))
        policy = SharedPolicy(self.model)
        stat = ppo_update(policy, [events], {**SETTINGS_B400['ppo'], 'epochs_per_update': 1})
        self.assertLess(stat['on_policy_log_probability_max_error'], 2e-5)
        self.assertGreater(stat['optimizer_steps'], 0)
        checkpoint = self.root / 'checkpoint'; snapshot(policy, checkpoint, stat); SharedPolicy(checkpoint)
        task['sample'] = True; write_json(taskfile, task)
        with self.assertRaises(ValueError): workflow.worker(SimpleNamespace(out=self.root, worker='search', task=taskfile), p)

    def test_report_pairing_reference_zero_and_pool_bounds(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location('hv_fixture', HERE.parent / 'hv_instance_union_v1/recompute.py')
        hv = importlib.util.module_from_spec(spec); spec.loader.exec_module(hv)
        rows = []
        methods = ['BASE', RULE, INDEPENDENT, SHARED, PPO]
        for n in ('one', 'two', 'three'):
            for seed in range(10):
                for m in methods:
                    rows.append({'instance': n, 'algorithm_seed': seed, 'method': m, 'hv': 0. if m == 'BASE' else .5,
                                 'seconds': 1., 'points': [(1., 2., 3.)]})
        comparisons, _, concise = comparison_tables(rows, methods, hv)
        self.assertIsNone(comparisons[SHARED]['BASE']['percent']['mean'])
        self.assertEqual(comparisons[PPO][SHARED]['absolute']['ties'], 30)
        self.assertEqual(len(concise), 5)
        # Include all per-run nondominated points; don't discard dominated union points before bounds.
        bounds = hv.bounds_of([[(1., 2., 3.)], [(2., 3., 4.)]])
        self.assertEqual(bounds['maximum'], [2., 3., 4.])

    def test_complete_terminal_schedule_role_isolation_and_resume(self):
        root = self.root / 'terminal_schedule'; root.mkdir()
        p = {'sha256': 'schedule', 'train_instances': ['t' + str(i) for i in range(200)],
             'validation_instances': ['v' + str(i) for i in range(20)], 'instances': ['screen-only']}
        seen = []
        def fake_rollout(run_root, plan, name, seed, model, folder, method, sample=False, role='screen', reference=None):
            self.assertNotEqual(role, 'screen')
            self.assertIn(name, p['train_instances'] if sample else p['validation_instances'])
            self.assertEqual(role, 'train' if sample else 'validation')
            rowfile = folder / 'RUN_COMPLETE.json'
            if rowfile.exists(): return read_json(rowfile)
            folder.mkdir(parents=True)
            if sample:
                policy = SharedPolicy(model, True, seed, True)
                for c in COMPONENTS: policy.choose(c, self.rows[c][0])
                policy.save_events(folder / 'TRAJECTORY.npz')
            # Fixed synthetic returns ensure validation rejects trained checkpoints.
            update = int(model.name[-3:]); reward = .5 if sample else .5 - update * .001
            row = {'final_hypervolume': reward}
            write_json(rowfile, row); seen.append((name, role))
            return row
        with patch.object(workflow, 'rollout', side_effect=fake_rollout), contextlib.redirect_stdout(io.StringIO()):
            chosen = workflow.terminal(root, p, self.model)
        self.assertEqual(chosen.name, 'update_000')
        self.assertEqual(sum(role == 'train' for _, role in seen), 32)
        self.assertEqual(sum(role == 'validation' for _, role in seen), 60)
        before = {str(f): digest(f) for f in (root / 'terminal/checkpoints').rglob('*.pt')}
        with patch.object(workflow, 'rollout', side_effect=fake_rollout), contextlib.redirect_stdout(io.StringIO()):
            workflow.terminal(root, p, self.model)
        self.assertEqual(len(seen), 92)
        self.assertEqual(before, {str(f): digest(f) for f in (root / 'terminal/checkpoints').rglob('*.pt')})

if __name__ == '__main__': unittest.main()
