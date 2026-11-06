"""The prescreen inventory must not count unlabeled or incompatible runs."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from experiments.audit_m_prescreen_data import collect


class PrescreenInventoryTests(unittest.TestCase):
    def test_ccp100_labels_require_exact_setup_and_full_encoding(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            config = {
                'evaluation_mode': 'ccp', 'training_size': 100,
                'alpha': .9, 'population': 20, 'evaluation_budget': 400,
                'uncertainty': {'name': 'transport_border_saa_v1',
                                'scenario_count': 100},
            }
            good = root/'good'
            good.mkdir()
            (good/'configuration.json').write_text(json.dumps(config))
            (good/'COMPLETE.json').write_text(json.dumps({'completed': True}))
            rows = [
                {'q90_cost_after': 1, 'q90_emission_after': 2,
                 'q90_makespan_after': 3, 'feasible_after': True,
                 'child_id': 'fingerprint'},
                {'q90_cost_after': 1, 'q90_emission_after': 2,
                 'q90_makespan_after': 3, 'feasible_after': True,
                 'child_chromosome': {'allocations': []}},
                {'q90_cost_after': None, 'q90_emission_after': 2,
                 'q90_makespan_after': 3, 'child_id': 'invalid'},
                {'operator': 'replace', 'parent_id': 'a', 'child_id': 'b'},
            ]
            (good/'mutation_outcomes.jsonl').write_text(
                ''.join(json.dumps(row)+'\n' for row in rows))
            bad = root/'wrong_scenarios'
            bad.mkdir()
            config['uncertainty']['scenario_count'] = 20
            (bad/'configuration.json').write_text(json.dumps(config))
            (bad/'COMPLETE.json').write_text(json.dumps({'completed': True}))
            (bad/'mutation_outcomes.jsonl').write_text(json.dumps(rows[1])+'\n')

            result = collect([root], None)
            self.assertEqual(result['selected_log_files'], 1)
            self.assertEqual(result['nonmatching_or_incomplete_files'], 1)
            self.assertEqual(result['selected']['finite_selected_q90'], 2)
            self.assertEqual(result['selected']['selected_with_full_encoding'], 1)
            self.assertEqual(result['selected']['fingerprint_only_labels'], 1)
            self.assertEqual(result['selected']['action_only_rows'], 1)


if __name__ == '__main__':
    unittest.main()
