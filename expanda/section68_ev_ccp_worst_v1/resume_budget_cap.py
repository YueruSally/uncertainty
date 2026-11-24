#!/usr/bin/env python3
"""Explicit continuation after accepting the frozen Base's 4999/5000 stop rule.

Do not edit launch.py: existing plans hash that runner and the evaluator. This
entry point records the authorised protocol amendment separately and reuses the
unchanged worker, inputs, completed attempts and reporting functions.
"""
import argparse
import fcntl
import os
from pathlib import Path
import sys
import traceback

sys.dont_write_bytecode = True
import launch as runner


def completed_budget_rows(root, plan):
    rows = []
    for index, task in enumerate(plan['tasks']):
        folder = runner.completed_task(root, task, plan)
        if folder is None:
            continue
        metadata = runner.read(folder/'COMPLETE.json')
        search = runner.read(folder/'search/RUN_COMPLETE.json')
        count = metadata['actual_evaluations']
        reason = search.get('stop_reason')
        expected_reason = {4999: 'atomic_pair_budget_remainder', 5000: 'evaluation_cap'}
        if (type(count) is not int or count not in expected_reason
                or search['real_objective_evaluation_count'] != count
                or reason != expected_reason[count]):
            raise ValueError('Completed search violates the accepted original budget stop: '+
                             str(dict(task=task, actual_evaluations=count, stop_reason=reason)))
        rows.append(dict(index=index, **task, actual_evaluations=count,
                         stop_reason=reason, complete_file=str(folder/'COMPLETE.json'),
                         complete_sha256=runner.sha(folder/'COMPLETE.json')))
    return rows


def main(args):
    if not args.accept_original_budget_stop:
        raise ValueError('Explicit --accept-original-budget-stop is required')
    if not 1 <= args.workers <= 2:
        raise ValueError('At most two searches in parallel')
    root = args.out.resolve()
    if not (root/'PLAN.json').is_file():
        raise ValueError('Existing prepared run directory required; this command does not prepare new experiments')
    lock = (root/'RUN.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.environ['MPLCONFIGDIR'] = str(root/'matplotlib_cache')
    try:
        plan = runner.read(root/'PLAN.json')
        if (plan['output'] != str(root)
                or runner.canonical({k: v for k, v in plan.items() if k != 'sha256'}) != plan['sha256']):
            raise ValueError('Frozen plan changed')
        if (plan['protocol'] != 'section68-ev-ccp-worst-v1'
                or plan['settings']['budget'] != 5000 or len(plan['tasks']) != 18
                or plan['pilot_task_indices'] != [0, 1, 2]):
            raise ValueError('Unexpected experiment protocol')
        runner.verify(plan['files'])
        env = runner.environment_review(Path(plan['preflight']),
                    plan.get('environment_policy', {}).get('accepted_memtotal_pair'))
        environment_file = root/('ENVIRONMENT_RECHECK_'+runner.stamp()+'.json')
        runner.save(environment_file, env)
        completed = completed_budget_rows(root, plan)
        if not {0, 1, 2}.issubset({r['index'] for r in completed}):
            raise ValueError('All three pilot searches must already be complete and verified')
        policy_file = root/('BUDGET_CAP_CONTINUATION_'+runner.stamp()+'.json')
        policy = dict(protocol='section68-budget-cap-continuation-v1',
            accepted_at_utc=runner.stamp(), explicitly_requested=True,
            acceptance_flag='--accept-original-budget-stop',
            plan_sha256=plan['sha256'], plan_file_sha256=runner.sha(root/'PLAN.json'),
            budget_upper_limit=5000, accepted_actual_counts=[4999, 5000],
            accepted_stop_reasons={'4999': 'atomic_pair_budget_remainder', '5000': 'evaluation_cap'},
            amendment='Accept the unchanged Base atomic-pair stop at 4999; report actual counts; continue after pilot',
            old_plan_and_results_unchanged=True, completed_before=completed,
            remaining_searches=18-len(completed), workers=args.workers,
            environment_file=str(environment_file), environment_sha256=runner.sha(environment_file),
            previous_status=runner.read(root/'STATUS.json') if (root/'STATUS.json').exists() else None,
            entrypoint=str(Path(__file__).resolve()), entrypoint_sha256=runner.sha(__file__),
            commit=runner.command(['git', 'rev-parse', 'HEAD']))
        runner.save(policy_file, policy)
        print('[SECTION68] Verified completed searches:', len(completed),
              '; remaining:', 18-len(completed), '; budget policy:', policy_file, flush=True)
        runner.status(root, completed=False, stage='formal', done=len(completed), total=18,
                      budget_policy=str(policy_file), actual_evaluation_counts=completed)
        runner.run_tasks(root, plan, list(range(18)), args.workers)
        completed = completed_budget_rows(root, plan)
        if len(completed) != 18:
            raise ValueError('Formal run did not complete all 18 searches')
        runner.verify(plan['files'])
        runner.verify({str(root/'PLAN.json'): policy['plan_file_sha256'],
                       str(Path(__file__).resolve()): policy['entrypoint_sha256']})
        output, counts = runner.export(root, plan, list(range(18)), 'RESULTS')
        # Separate append-only provenance: export's existing manifest stays intact.
        runner.save(output/'BUDGET_POLICY.json', dict(
            source=str(policy_file), sha256=runner.sha(policy_file),
            budget_upper_limit=5000, actual_evaluation_counts=completed,
            qualification='Budget is an upper limit. Actual counts are not rounded up or supplemented.'))
        runner.status(root, completed=True, stage='all', done=18, total=18,
                      results=str(output), budget_policy=str(policy_file),
                      actual_evaluation_counts=counts)
    except BaseException as exc:
        runner.save(root/('FAILURE_'+runner.stamp()+'.json'),
                    dict(error=repr(exc), traceback=traceback.format_exc()))
        runner.status(root, completed=False, stage='stopped', error=str(exc))
        raise
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--accept-original-budget-stop', action='store_true')
    main(parser.parse_args())
