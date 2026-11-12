"""S0 B400 joint-learning extension; all historical code and outputs are read-only."""
from pathlib import Path
import importlib.util
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'joint_icm_v1'))
from joint_common import *
from terminal_ppo import CONFIG as TERMINAL_CONFIG

SHARED = 'ICM_SHARED_SUPERVISED'
PPO = 'ICM_SHARED_TERMINAL_PPO'
RULE = 'I_C_M_K8_HEURISTIC'
INDEPENDENT = 'I_C_M_K8_TRANSFORMER'
SETTINGS_B400 = {
    'protocol': 's0-b400-shared-and-terminal-v1',
    'population': 20, 'budget': 400, 'generations': 1000,
    'scenario_count': 100, 'confidence': .9, 'candidate_pool': 8,
    'workers': 1, 'worker_timeout_seconds': 1800,
    'ppo': {**TERMINAL_CONFIG, 'population': 20, 'budget': 400,
            'max_generations': 1000, 'automatic_execution': True},
    'scope': 'Existing three S0 development instances x ten paired seeds; not a new independent test',
    'maximum_new_searches': 152,
    'maximum_completed_search_evaluations': 60800,
    'timing': 'Historical runs used three workers. New runs use one worker and record hostname; times are descriptive, not a controlled speed comparison.',
    'reporting': 'Recompute all eleven methods using each instance union of final feasible fronts; never use reporting bounds for training or selection.'
}


def history(source):
    spec = importlib.util.spec_from_file_location(
        's0_history_report', HERE.parent / 'icm_heuristic_270_v1/report.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    design, rows, files = module.hv.load_runs(source)
    if design['protocol'] != module.PROTOCOL or len(rows) != 270:
        raise ValueError('Expected the original 240 + 30 experiment, not the 900-run extension')
    return design, rows, files, module.hv


def runtime_files():
    paths = [HERE / n for n in ('common.py', 'run.py', 'results.py')]
    paths += [HERE.parent / 'icm_heuristic_270_v1/report.py',
              HERE.parent / 'hv_instance_union_v1/recompute.py']
    return {**sources(), **{str(p): digest(p) for p in paths}}


def check_split(bundle):
    train = set(bundle['training_instances']['I'])
    valid = set(bundle['validation_instances']['I'])
    if len(train) != 200 or len(valid) != 20 or train & valid:
        raise ValueError('Expected original disjoint 200/20 S0 split')
    for c in COMPONENTS:
        if (set(bundle['training_instances'][c]) != train or
                set(bundle['validation_instances'][c]) != valid):
            raise ValueError('Component split identities differ')
    if bundle['normalization_sha256'] != digest(PACKAGE / 'assets/reward_normalization.json'):
        raise ValueError('Normalization differs from historical experiment')
    return sorted(train), sorted(valid)


def choose_checkpoint(validation):
    return max(validation, key=lambda r: (r['mean_difference_vs_initial'], -r['update']))
