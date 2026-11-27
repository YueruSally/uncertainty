#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
parent="${1:-$PWD/transformer_ppo_runs/diagnostics/ccp_final_front_oos_hv_full_v1_20261126T133631290126Z}"
out="${2:-$PWD/transformer_ppo_runs/diagnostics/ccp_standard_nsga2_alpha09_v1}"
python3 -B -u standard_nsga2_alpha09_v1/run.py --parent "$parent" --out "$out" --prepare-only
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
log="$PWD/transformer_ppo_runs/diagnostics/standard_nsga2_alpha09_${stamp}_$$.log"
nohup python3 -B -u standard_nsga2_alpha09_v1/run.py --parent "$parent" --out "$out" > "$log" 2>&1 < /dev/null &
printf 'Standard NSGA-II P90 PID: %s\nOutput: %s\nLog: %s\n' "$!" "$out" "$log"
printf 'Live progress: watch -n 10 '\''cat "%s/STATUS.json"'\''\n' "$out"
printf 'Live log: tail -F "%s"\n' "$log"
