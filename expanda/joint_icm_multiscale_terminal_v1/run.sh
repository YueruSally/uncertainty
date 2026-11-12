#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec python3 -u "$SCRIPT_DIR/run.py" \
  --out "${MULTI_PPO_ROOT:-$PWD/transformer_ppo_runs/diagnostics/ccp_icm_multiscale_joint_v1}" \
  --workers "${MAX_WORKERS:-2}" "$@"
