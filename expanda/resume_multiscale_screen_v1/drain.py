"""Linux handoff: stop dispatch, finish existing workers, retire the supervisor.

Only the verified supervisor receives signals. Workers are never stopped here.
An interrupted handoff resumes the old supervisor unless retirement has begun.
"""
from pathlib import Path
import json
import os
import signal
import time


def process(pid):
    p = Path('/proc') / str(pid)
    try:
        uid = p.stat().st_uid
        if uid != os.getuid():
            return None
        stat = (p/'stat').read_text().rsplit(') ', 1)[1].split()
        args = (p/'cmdline').read_bytes().decode().rstrip('\0').split('\0')
        return dict(pid=int(pid), ppid=int(stat[1]), state=stat[0], start=stat[19],
                    uid=uid, args=args)
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None


def processes():
    return [r for p in Path('/proc').iterdir() if p.name.isdigit()
            for r in [process(p.name)] if r is not None and r['uid'] == os.getuid()]


def matches(r, script, root, worker):
    args = r['args']
    if str(script) not in args or ('--worker' in args) != worker:
        return False
    if '--out' not in args or args.index('--out') + 1 >= len(args):
        return False
    return Path(args[args.index('--out')+1]).resolve() == root.resolve()


def active(r):
    return r is not None and r['state'] not in ('Z', 'X')


def finish_workers(handle, parent, workers, write_state, deadline, interval=2,
                   inspect=process, now=time.monotonic, sleep=time.sleep):
    """The handle pins a verified PID; it cannot accidentally signal a reused PID."""
    stopped = False
    retired = False
    try:
        if parent['state'] in ('T', 't'):
            raise RuntimeError('Supervisor was already stopped externally; refusing takeover')
        handle.send(signal.SIGSTOP)
        stopped = True
        # Once stopped, the supervisor cannot dispatch any additional workers.
        started = now()
        while True:
            p = inspect(parent['pid'])
            if not active(p):
                raise RuntimeError('Supervisor exited during handoff')
            if p['state'] in ('T', 't'):
                break
            if now()-started > 10:
                raise RuntimeError('Supervisor did not stop dispatching')
            sleep(.1)
        owned = workers()
        heartbeat = -1e30
        while True:
            pending = []
            for p in owned:
                current = inspect(p['pid'])
                if active(current) and current['start'] == p['start']:
                    pending.append(p)
            if now()-heartbeat >= 30 or not pending:
                write_state({'stage':'draining_existing_workers', 'supervisor_pid':parent['pid'],
                             'supervisor_start':parent['start'], 'remaining_pids':[p['pid'] for p in pending],
                             'qualification':'Supervisor is temporarily stopped; workers continue normally.'})
                heartbeat = now()
            if not pending:
                break
            if now() > deadline:
                raise TimeoutError('Drain timeout: old supervisor will resume, no worker is killed')
            sleep(interval)
        # Children have finished (possibly zombies awaiting the parent's wait()).
        # Its normal SIGTERM handler records a resumable pause and releases its lock.
        handle.send(signal.SIGTERM)
        handle.send(signal.SIGCONT)
        retired = True
        stopped = False
        write_state({'stage':'old_supervisor_retiring', 'supervisor_pid':parent['pid'],
                     'supervisor_start':parent['start'], 'worker_pids':[p['pid'] for p in owned]})
    finally:
        if stopped and not retired:
            handle.send(signal.SIGCONT)


class PinnedProcess:
    def __init__(self, expected):
        if not hasattr(os, 'pidfd_open') or not hasattr(signal, 'pidfd_send_signal'):
            raise RuntimeError('Linux Python with pidfd support is required for safe handoff')
        self.fd = os.pidfd_open(expected['pid'])
        current = process(expected['pid'])
        if current is None or (current['start'], current['uid'], current['args']) != (
                expected['start'], expected['uid'], expected['args']):
            os.close(self.fd)
            raise RuntimeError('Supervisor identity changed')

    def send(self, signum):
        try:
            signal.pidfd_send_signal(self.fd, signum)
        except ProcessLookupError:
            if signum != signal.SIGCONT:
                raise

    def close(self):
        os.close(self.fd)


def handoff(root, script, write_state, timeout):
    if not Path('/proc/self/stat').exists():
        raise RuntimeError('Process handoff must run on the Linux experiment server')
    candidates = [p for p in processes() if active(p) and matches(p, script, root, False)]
    if len(candidates) != 1:
        raise RuntimeError(f'Expected exactly one matching supervisor, found {len(candidates)}')
    parent = candidates[0]
    handle = PinnedProcess(parent)
    write_state({'stage':'before_dispatch_pause', 'supervisor_pid':parent['pid'],
                 'supervisor_start':parent['start']})
    def workers():
        children = [p for p in processes() if p['ppid'] == parent['pid'] and active(p)]
        if any(not matches(p, script, root, True) for p in children):
            raise RuntimeError('Unexpected child under supervisor; refusing handoff')
        write_state({'stage':'handoff_identity_verified', 'supervisor':parent,
                     'worker_pids':[p['pid'] for p in children]})
        return children
    try:
        finish_workers(handle, parent, workers, write_state, time.monotonic()+timeout)
    finally:
        handle.close()


def recover(record):
    """For a killed handoff process only: resume exactly its recorded supervisor."""
    if 'supervisor_pid' not in record or 'supervisor_start' not in record:
        raise RuntimeError('No stopped-supervisor identity recorded')
    p = process(record['supervisor_pid'])
    if not active(p) or p['uid'] != os.getuid() or p['start'] != record['supervisor_start']:
        raise RuntimeError('Recorded process is no longer present; no signal sent')
    with_handle = PinnedProcess(p)
    try:
        with_handle.send(signal.SIGCONT)
    finally:
        with_handle.close()
