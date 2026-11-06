from copy import deepcopy
import random
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from experiments import m_replace_profile_screen as screen
from transformer_ppo.controller import PPOController


def candidate(cost, emission, risk, batch, path, eligible=True):
    return {'eligible': eligible, 'target': SimpleNamespace(batch_id=batch), 'path_id': path,
        'batch_quantity': 1., 'replacement_delta_cost_per_teu': cost,
        'replacement_delta_emission_per_teu': emission, 'replacement_delta_nominal_time_h': risk,
        'replacement_max_lateness_h': risk, 'replacement_on_time_probability': 1-risk}


def grid():
    rows = []
    for name in screen.INSTANCES:
        for seed in screen.SEEDS:
            for p in screen.PROFILES:
                rows.append({'instance': name, 'algorithm_seed': seed, 'strategy': p, 'replace_profile': p,
                    'real_objective_evaluation_count': 400, 'evaluation_budget': 400,
                    'minimum_control_generation': 10, 'final_hypervolume': 1., 'optimisation_seconds': 1.,
                    'pairing': {'instance': name, 'seed': seed, 'ccp': 100},
                    'first_profile_state': {'generation': 10, 'parent': 'same'}})
    return rows


class ReplaceProfileTests(unittest.TestCase):
    def test_balanced_uses_exact_old_ranking_for_masks_and_ties(self):
        rows = [candidate(.3, .5, .1, 2, 'b'), candidate(.3, .5, .1, 1, 'a'),
                candidate(0., 0., 0., 0, 'illegal', False)]
        self.assertEqual(screen.profile_index(rows, 'BALANCED'), 1)
        self.assertEqual(screen.profile_index(rows, 'BALANCED'),
                         PPOController._joint_balanced_candidate_index(rows))
        with patch.object(PPOController, '_joint_balanced_candidate_index', return_value=1) as old:
            self.assertEqual(screen.profile_index(rows, 'BALANCED'), 1)
            old.assert_called_once_with(rows)

    def test_each_predeclared_preference_can_change_target(self):
        rows = [candidate(.4, .4, .4, 1, 'balanced'), candidate(0, .8, .6, 2, 'cost'),
                candidate(.8, 0, .6, 3, 'emission'), candidate(.7, .7, 0, 4, 'time'),
                candidate(1, 1, 1, 5, 'dominated'),
                candidate(-100, -100, -100, 0, 'illegal', False)]
        self.assertEqual([screen.profile_index(rows, p) for p in screen.PROFILES], [0, 1, 2, 3])

    def test_target_selection_does_not_mutate_rows_or_consume_rng(self):
        rows = [candidate(.4, .2, .1, 1, 'x'), candidate(.1, .3, .5, 2, 'y')]
        saved = deepcopy(rows)
        py, npstate = random.getstate(), np.random.get_state()
        for profile in screen.PROFILES:
            screen.profile_index(rows, profile)
        self.assertEqual(rows, saved)
        self.assertEqual(py, random.getstate())
        self.assertTrue(np.array_equal(npstate[1], np.random.get_state()[1]))

    def test_no_legal_target_and_constant_columns_are_supported(self):
        rows = [candidate(1., 1., 1., 2, 'b'), candidate(1., 1., 1., 1, 'a')]
        for profile in screen.PROFILES:
            self.assertEqual(screen.profile_index(rows, profile), 1)
            self.assertIsNone(screen.profile_index([], profile))
            self.assertIsNone(screen.profile_index([{**r, 'eligible': False} for r in rows], profile))
        with self.assertRaises(ValueError):
            screen.profile_index(rows, 'unfrozen')

    def test_identity_profile_calls_original_executor_and_rejects_early_use(self):
        controller = object.__new__(screen.ProfileController)
        controller.generation, controller.evaluations = 10, 220
        controller.profile = 'BALANCED'
        controller.profile_decisions, controller.first_profile_state = 0, None
        controller.current_population = []
        with patch.object(screen, 'fingerprint', return_value='parent'), patch.object(
                screen.CandidateController, '_balanced_replace_target', return_value=('b', 't', 'm')) as old:
            self.assertEqual(controller._balanced_replace_target('parent', [], []), ('b', 't', 'm'))
            old.assert_called_once_with('parent', [], [])
        controller.generation = 9
        with self.assertRaisesRegex(RuntimeError, 'pre-G10'):
            controller._balanced_replace_target('parent', [], [])

    def test_grid_checks_budget_pairing_and_profile_identity(self):
        rows = grid()
        self.assertEqual(screen.analyze(rows)['new_objective_evaluations'], 14400)
        for field, value in [('real_objective_evaluation_count', 401), ('pairing', {}),
                             ('replace_profile', 'wrong'), ('first_profile_state', None),
                             ('counterfactual_evaluation_count', 1), ('candidate_audit_evaluation_count', 1),
                             ('forced_operator_candidate_evaluations', 1)]:
            with self.subTest(field=field):
                bad = deepcopy(rows)
                bad[1][field] = value
                with self.assertRaises(ValueError):
                    screen.analyze(bad)
        with self.assertRaisesRegex(ValueError, 'grid'):
            screen.analyze(rows[:-1])
        with self.assertRaisesRegex(ValueError, 'grid'):
            screen.analyze(rows+[rows[0]])

    def test_cross_seed_selection_does_not_see_heldout_outcomes(self):
        rows = grid()
        for row in rows:
            if row['strategy'] == 'COST':
                row['final_hypervolume'] = 1.02
        first = screen.analyze(rows)
        bad = deepcopy(rows)
        for row in bad:
            if row['algorithm_seed'] == screen.SEEDS[0] and row['strategy'] == 'EMISSION':
                row['final_hypervolume'] = 2.
        second = screen.analyze(bad)
        for report in (first, second):
            choices = [r['selected_profile'] for r in report['cross_seed_selection']
                       if r['heldout_seed'] == screen.SEEDS[0]]
            self.assertEqual(choices, ['COST']*3)
        self.assertTrue(first['screen_pass'])
        self.assertFalse(first['automatic_training'])

    def test_always_balanced_or_a_single_lucky_instance_does_not_pass(self):
        rows = grid()
        report = screen.analyze(rows)
        self.assertFalse(report['screen_pass'])
        self.assertEqual(set(r['selected_profile'] for r in report['cross_seed_selection']), {'BALANCED'})
        for row in rows:
            if row['instance'] == screen.INSTANCES[0] and row['strategy'] == 'COST':
                row['final_hypervolume'] = 1.1
        self.assertFalse(screen.analyze(rows)['screen_pass'])


if __name__ == '__main__':
    unittest.main()
