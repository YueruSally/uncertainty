#!/usr/bin/env python3
"""S0 reproduction first, then S0-S11 fixed-plan Monte Carlo precision audit."""
import argparse
import fcntl
import os
from pathlib import Path
import signal
import socket
import sys
import time
from common import (CONFIGS, DIAG, PROTOCOL, canonical, csv_write, freeze, read,
                    runtime_files, verify, write)


def run(args):
    root = args.out.resolve()
    for source in (args.source.resolve(), args.large_source.resolve()):
        if root == source or source in root.parents or root in source.parents:
            raise SystemExit('Audit output must be separate from both source experiments.')
    if (root/'PLAN.json').exists():
        if read(root/'PLAN.json').get('protocol', {}).get('version') != PROTOCOL['version']:
            raise SystemExit('Existing output belongs to another experiment; nothing changed.')
    root.mkdir(parents=True, exist_ok=True)
    lock = (root/'.audit.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit('This isolated audit is already running; no second process started.')
    os.environ['MPLCONFIGDIR'] = str(root/'.matplotlib')
    # Changes priority only for this process. Never inspect/stop existing jobs.
    if hasattr(os, 'nice'):
        os.nice(max(0, 10-os.nice(0)))
    started = time.monotonic()
    latest = {}
    def heartbeat(**fields):
        latest.clear(); latest.update(fields)
        payload = dict(completed=False, pid=os.getpid(), hostname=socket.gethostname(),
            workers=1, threads=1, nice=os.nice(0), new_searches=0, training_updates=0,
            invocation_seconds=round(time.monotonic()-started, 2), **fields)
        write(root/'STATUS.json', payload)
        print('[Q90-FIXED]', payload, flush=True)
    def stop(signum, frame):
        raise KeyboardInterrupt(f'Received signal {signum}; completed packs are reusable')
    signal.signal(signal.SIGTERM, stop)
    try:
        from sources import Pending, prepare
        from evaluate import evaluate_packs
        from report import configuration_report, aggregate
        runtime = runtime_files()
        plan = dict(protocol=PROTOCOL, source=str(args.source.resolve()),
            large_source=str(args.large_source.resolve()), files=runtime)
        plan['sha256'] = canonical(plan)
        freeze(root/'PLAN.json', plan)
        known, unavailable = {}, {}
        for cfg in CONFIGS:
            try:
                known[cfg] = prepare(cfg, args.source.resolve(), args.large_source.resolve(), root/cfg, runtime)
            except Pending as e:
                unavailable[cfg] = str(e)
        write(root/'SOURCE_INVENTORY.json', dict(ready={c:v['unique_plans'] for c,v in known.items()},
            waiting=unavailable, selection='Names 001 fixed in advance; no outcome-dependent substitution'))
        heartbeat(stage='source_inventory', ready=list(known), waiting=list(unavailable),
                  ready_unique_plans=sum(v['unique_plans'] for v in known.values()))
        results = []
        for cfg in CONFIGS:
            if args.s0_only and cfg != 'S0':
                break
            while cfg not in known:
                try:
                    known[cfg] = prepare(cfg, args.source.resolve(), args.large_source.resolve(), root/cfg, runtime)
                    unavailable.pop(cfg, None)
                    write(root/'SOURCE_INVENTORY.json', dict(ready={c:v['unique_plans'] for c,v in known.items()},
                        waiting=unavailable, selection='Names 001 fixed in advance; no outcome-dependent substitution'))
                except Pending as e:
                    heartbeat(stage='waiting_for_source', configuration=cfg, missing=str(e),
                        completed_configurations=[r['configuration'] for r in results],
                        resume='Repeat the same command; completed packs reused')
                    if args.no_wait:
                        return
                    time.sleep(30)
            inputs = known[cfg]
            verify(runtime); verify(inputs['files'])
            metas = evaluate_packs(root/cfg, inputs, plan['sha256'], heartbeat)
            result = configuration_report(root/cfg, inputs, metas)
            verify(inputs['files'])
            results.append(result)
            totals = aggregate(root, results)
            write(root/'COSTS.json', totals)
            done = {r['configuration'] for r in results}
            # K-weighted observed evaluator rate. Unknown front sizes use the
            # mean completed size and are explicitly marked as estimates.
            measured_weight = sum(r['unique_plans']*known[r['configuration']]['instance']['K'] for r in results)
            rate = totals['evaluator_seconds']/measured_weight
            typical = totals['unique_plans']/len(results)
            remaining_weight = 0.
            pending_sizes = []
            from transformer_ppo.instance_catalog import SCENARIOS
            for c in CONFIGS:
                if c in done:
                    continue
                if c in known:
                    n, k = known[c]['unique_plans'], known[c]['instance']['K']
                else:
                    n, k = typical, SCENARIOS[c][0]; pending_sizes.append(c)
                remaining_weight += n*k
            eta = dict(remaining_compute_hours=rate*remaining_weight/3600,
                unknown_front_sizes=pending_sizes,
                qualification='K-weighted estimate from completed configurations; excludes source waiting, future contention and reporting; unknown sizes use completed mean.')
            write(root/'ETA.json', eta)
            heartbeat(stage='configuration_complete', configuration=cfg,
                completed_configurations=sorted(done, key=CONFIGS.index),
                report=str(root/cfg/'SUMMARY.csv'), remaining_compute_hours=round(eta['remaining_compute_hours'], 3))
            if cfg == 'S0':
                print('[Q90-FIXED] S0 CHECK PASSED; S0/SUMMARY.csv is ready. No search/model changed.', flush=True)
        verify(runtime)
        for cfg in (r['configuration'] for r in results):
            verify(known[cfg]['files'])
        finished = len(results) == 12
        write(root/'STATUS.json', dict(completed=finished, stage='all' if finished else 's0_only_complete',
            requested_stage_complete=True, completed_configurations=[r['configuration'] for r in results],
            workers=1, threads=1, new_searches=0, training_updates=0,
            invocation_seconds=time.monotonic()-started, **{k:v for k,v in totals.items() if k!='completed_configurations'}))
        print('[Q90-FIXED] COMPLETE' if finished else '[Q90-FIXED] S0 COMPLETE; repeat without --s0-only to continue', root, flush=True)
    except KeyboardInterrupt as e:
        write(root/'STATUS.json', dict(completed=False, paused=True, reason=str(e), last_status=latest,
            new_searches=0, training_updates=0, resume='Repeat the same command; completed packs reused'))
        print('[Q90-FIXED] PAUSED', e, flush=True)
    except Exception as e:
        write(root/'STATUS.json', dict(completed=False, failed=True, reason=repr(e), last_status=latest,
                                     new_searches=0, training_updates=0))
        raise
    finally:
        lock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=DIAG/'ccp_fixed_q90_s0_s11_v1')
    parser.add_argument('--source', type=Path, default=DIAG/'ccp_icm_single_member_second_instance_b5000_v1')
    parser.add_argument('--large-source', type=Path, default=DIAG/'ccp_single_s9_s11_b5000_v1')
    parser.add_argument('--s0-only', action='store_true', help='Stop after S0; same output resumes the full protocol')
    parser.add_argument('--no-wait', action='store_true', help='Exit resumably if a predetermined source front is not ready')
    run(parser.parse_args())
