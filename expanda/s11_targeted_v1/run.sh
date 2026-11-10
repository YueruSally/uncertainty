#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec python3 -u "$SCRIPT_DIR/run.py" --stage all --workers "${MAX_WORKERS:-2}" --out "${TASK_ROOT:-$PWD/transformer_ppo_runs/diagnostics/ccp_s11_targeted_supervised_v1}"
