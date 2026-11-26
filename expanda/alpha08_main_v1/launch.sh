#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
workers="${1:-1}"
case "$workers" in 1|2) ;; *) printf 'workers must be 1 or 2\n' >&2; exit 2;; esac
out="${2:-$PWD/transformer_ppo_runs/diagnostics/ccp_alpha08_single_full_v1}"
# Preflight must succeed before starting any background computation.
python3 -B -u alpha08_main_v1/run.py --out "$out" --workers "$workers" --prepare-only
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
log="$PWD/transformer_ppo_runs/diagnostics/alpha08_launch_${stamp}_$$.log"
nohup python3 -B -u alpha08_main_v1/run.py --out "$out" --workers "$workers" > "$log" 2>&1 < /dev/null &
pid=$!
printf 'P80 process: %s\nOutput: %s\nLog: %s\n' "$pid" "$out" "$log"
printf 'Progress: cat "%s/STATUS.json"\n' "$out"
printf 'Log viewer: tail -f "%s"\n' "$log"
