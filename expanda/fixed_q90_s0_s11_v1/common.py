"""Isolated fixed-decision audit; no search or model-training imports."""
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['MPLBACKEND'] = 'Agg'
HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ENGINE = EXP / 'formal_icm_s0_v1/src'
DATA = EXP / 'formal_icm_s0_v1/assets/data/data_expanded.xlsx'
DIAG = EXP / 'transformer_ppo_runs/diagnostics'
CONFIGS = [f'S{i}' for i in range(12)]
ARMS = ('BASE', 'ICM_SHARED_SUPERVISED')
OBJECTIVES = ('cost', 'emission', 'makespan')
PROTOCOL = dict(version='fixed-q90-s0-s11-v1', configurations=CONFIGS,
    selection='first named instance 001; Base and frozen single Shared SL; all three existing seeds',
    population=250, search_budget=5000, original_realizations=100,
    reference_realizations=1000, replicate_realizations=100, replicates=20,
    confidence=0.9, new_seed_base=280000000, representative='smallest exact decision SHA256',
    workers=1, threads=1, new_searches=0, training_updates=0,
    reference_qualification='Independent finite-sample reference, not ground truth',
    original_check_rtol=1e-10, original_check_atol=1e-8)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.writing')
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    tmp.replace(path)


def freeze(path, value):
    if path.exists():
        if read(path) != value:
            raise ValueError('Frozen input differs; use a new output version: ' + str(path))
    else:
        write(path, value)


def csv_write(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.writing')
    with tmp.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    tmp.replace(path)


def verify(files):
    for name, expected in files.items():
        if digest(Path(name)) != expected:
            raise ValueError('Frozen source changed: ' + name)


def runtime_files():
    paths = list(HERE.glob('*.py')) + list(HERE.glob('*.sh')) + [DATA]
    paths += [p for p in ENGINE.rglob('*') if p.suffix in ('.py', '.csv')]
    return {str(p): digest(p) for p in sorted(paths)}


def checked_plan(path):
    plan = read(path)
    if canonical({k: v for k, v in plan.items() if k != 'sha256'}) != plan['sha256']:
        raise ValueError('Invalid source plan: ' + str(path))
    return plan
