#!/usr/bin/env python3
"""Read existing M logs and count objective labels usable for a K8 screen.

No training, objective evaluation, instance generation, or server job starts.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n',
                         encoding='utf-8')
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

ENCODING_KEYS = ('child_allocations', 'od_allocations', 'child_chromosome',
                 'chromosome', 'child_encoding')
OBJECTIVE_KEYS = ('q90_cost_after', 'q90_emission_after', 'q90_makespan_after')


def valid_objectives(values):
    try:
        return len(values) == 3 and all(
            value is not None and math.isfinite(float(value)) for value in values)
    except (TypeError, ValueError):
        return False


def scan_selected(path):
    result = {'events': 0, 'finite_selected_q90': 0, 'selected_feasible': 0,
              'selected_with_full_encoding': 0, 'evaluated_parent_available': 0,
              'fingerprint_only_labels': 0}
    with path.open(encoding='utf-8') as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            result['events'] += 1
            valid = valid_objectives([row.get(key) for key in OBJECTIVE_KEYS])
            result['finite_selected_q90'] += int(valid)
            result['selected_feasible'] += int(valid and row.get('feasible_after') is True)
            has_encoding = any(row.get(key) is not None for key in ENCODING_KEYS)
            result['selected_with_full_encoding'] += int(valid and has_encoding)
            result['evaluated_parent_available'] += int(row.get('parent_objectives_available') is True)
            result['fingerprint_only_labels'] += int(valid and not has_encoding and row.get('child_id') is not None)
    return result


def scan_audit(path):
    result = {'evaluated_children': 0, 'finite_objective_labels': 0,
              'with_full_child_encoding': 0, 'operator_names': set()}
    with path.open(encoding='utf-8') as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            result['evaluated_children'] += 1
            result['finite_objective_labels'] += int(valid_objectives(row.get('objectives')))
            result['with_full_child_encoding'] += int(any(row.get(key) is not None for key in ENCODING_KEYS))
            if row.get('operator'):
                result['operator_names'].add(str(row['operator']))
    result['operator_names'] = sorted(result['operator_names'])
    return result


def valid_ccp100(path):
    config_path = path.parent/'configuration.json'
    if not config_path.exists():
        return False
    config = read_json(config_path)
    return (config.get('evaluation_mode') == 'ccp'
            and config.get('training_size') == 100
            and float(config.get('alpha', -1)) == .9
            and config.get('population') == 20
            and config.get('evaluation_budget') == 400
            and config.get('uncertainty', {}).get('name') == 'transport_border_saa_v1'
            and config.get('uncertainty', {}).get('scenario_count') == 100)


def collect(search_roots, collection_root):
    counts = {'selected': {'events': 0, 'finite_selected_q90': 0,
              'selected_feasible': 0, 'selected_with_full_encoding': 0,
              'evaluated_parent_available': 0, 'fingerprint_only_labels': 0},
              'audit': {'evaluated_children': 0, 'finite_objective_labels': 0,
              'with_full_child_encoding': 0, 'operator_names': []}}
    selected_files, audit_files, skipped_files = 0, 0, 0
    seen = set()
    for root in search_roots:
        if not root.is_dir():
            raise FileNotFoundError(root)
        for path in root.rglob('mutation_outcomes.jsonl'):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            if not valid_ccp100(path):
                skipped_files += 1
                continue
            selected_files += 1
            for key, value in scan_selected(path).items():
                counts['selected'][key] += value
    if collection_root is not None:
        quality = read_json(collection_root/'batch_2/QUALITY.json')
        if quality.get('data_quality_pass') is not True or quality.get('state_count') != 384:
            raise ValueError('requires complete 384-state collection')
        names = set()
        for path in collection_root.rglob('mutation_candidate_audit_outcomes.jsonl'):
            if not valid_ccp100(path):
                skipped_files += 1
                continue
            audit_files += 1
            row = scan_audit(path)
            names.update(row.pop('operator_names'))
            for key, value in row.items():
                counts['audit'][key] += value
        counts['audit']['operator_names'] = sorted(names)
    return {'selected_log_files': selected_files, 'audit_log_files': audit_files,
            'nonmatching_or_incomplete_files': skipped_files, **counts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--search-root', type=Path, action='append', required=True)
    parser.add_argument('--collection-root', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    protocol = SOURCE/'experiments/m_prescreen_protocol_v1.json'
    design = read_json(protocol)
    if design['protocol'] != 'm-prescreen-replace-v1' or design['candidate_limit'] != 8:
        raise ValueError('frozen candidate plan changed')
    roots = [p.resolve() for p in args.search_root]
    collection = args.collection_root.resolve() if args.collection_root else None
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    inputs = {'protocol_sha256': digest(protocol), 'search_roots': [str(p) for p in roots],
              'collection_root': str(collection) if collection else None}
    if (out/'PLAN.json').exists() and read_json(out/'PLAN.json') != inputs:
        raise ValueError('different source or protocol; choose a new output directory')
    write_json(out/'PLAN.json', inputs)
    print('[M-PRESCREEN-AUDIT] reading existing CCP100 logs', flush=True)
    result = collect(roots, collection)
    result.update({'completed': True, 'new_objective_evaluations': 0,
        'new_training_steps': 0, 'new_searches': 0,
        'historical_384_state_panel_is_K8_match': False,
        'directly_reusable_K8_labels': 0,
        'why': 'old selected logs lack full child encodings; the audited five operators are a different action pool',
        'next_action': 'construct and label a bounded exact-K8 panel using the current search executor; preserve every training-only counterfactual evaluation'})
    write_json(out/'QUICK_RESULT.json', result)
    (out/'AUDIT_COMPLETE').write_text('read-only inventory\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
