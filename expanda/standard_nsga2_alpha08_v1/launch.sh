#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
parent="${1:-$PWD/transformer_ppo_runs/diagnostics/ccp_alpha08_single_full_v1}"
out="${2:-$PWD/transformer_ppo_runs/diagnostics/ccp_standard_nsga2_alpha08_v1}"
python3 -B -u standard_nsga2_alpha08_v1/run.py --parent "$parent" --out "$out" --prepare-only
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
log="$PWD/transformer_ppo_runs/diagnostics/standard_nsga2_alpha08_${stamp}_$$.log"
nohup python3 -B -u standard_nsga2_alpha08_v1/run.py --parent "$parent" --out "$out" > "$log" 2>&1 < /dev/null &
printf 'Standard NSGA-II PID: %s\nOutput: %s\nLog: %s\n' "$!" "$out" "$log"
printf 'Live progress: watch -n 10 '\''cat "%s/STATUS.json"'\''\n' "$out"
printf 'Live log: tail -F "%s"\n' "$log"
