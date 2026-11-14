#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
exec python3 -u ppo_joint_b5000_pilot_v2/run.py \
  --source "${PPO_PILOT_SOURCE:-$PWD/transformer_ppo_runs/diagnostics/ccp_component_joint_b5000_v1}" \
  --out "${PPO_PILOT_ROOT:-$PWD/transformer_ppo_runs/diagnostics/ccp_joint_ppo_b5000_pilot_v2}" \
  --workers "${MAX_WORKERS:-2}"
