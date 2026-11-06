import tempfile
from pathlib import Path
import unittest

from m_operator_offline import digest, write_json
from run_m_prescreen_k8_collect import validate_run


class K8RunnerTests(unittest.TestCase):
    def test_completed_run_requires_matching_budget_scenarios_and_feedback(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            write_json(path/'FEEDBACK.json', {'rows': [
                {'labels': [{}, {}]}],
                'additional_objective_evaluations': 1})
            result = {'protocol': 'm-prescreen-k8-collection-v1',
                'instance_id': 'train-S0-002', 'algorithm_seed': 910001,
                'plan_sha256': 'frozen', 'real_objective_evaluation_count': 400,
                'feedback_sha256': digest(path/'FEEDBACK.json'),
                'feedback_state_count': 1, 'training_feedback_evaluations': 1}
            config = {'instance_id': 'train-S0-002',
                'evaluation_mode': 'ccp', 'population': 20,
                'evaluation_budget': 400, 'training_size': 100, 'alpha': .9,
                'uncertainty': {'name': 'transport_border_saa_v1',
                                'scenario_count': 100},
                'scenario_digest': 'frozen-scenarios'}
            write_json(path/'RUN_COMPLETE.json', result)
            write_json(path/'configuration.json', config)
            validate_run(path, 'train-S0-002', 910001, 'frozen',
                         'm-prescreen-k8-collection-v1')
            config['uncertainty']['scenario_count'] = 20
            write_json(path/'configuration.json', config)
            with self.assertRaisesRegex(ValueError, 'invalid completed'):
                validate_run(path, 'train-S0-002', 910001, 'frozen',
                             'm-prescreen-k8-collection-v1')


if __name__ == '__main__':
    unittest.main()
