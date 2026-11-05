#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
PYTHON=${PYTHON:-python3}
COLLECTION_ROOT=${COLLECTION_ROOT:-$PWD/transformer_ppo_runs/diagnostics/ccp_m_operator_feedback_v2_6inst_2seed_b400}
OUT_ROOT=${OUT_ROOT:-$PWD/transformer_ppo_runs/diagnostics/ccp_m_operator_offline_v1_384state}
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
mkdir -p "$OUT_ROOT"
echo "===== TEST GROUPED OFFLINE LEARNING / REAL TORCH BACKWARD ====="
if ! "$PYTHON" -u -m unittest tests.test_m_operator_offline_v1 > "$OUT_ROOT/tests.log" 2>&1; then
  cat "$OUT_ROOT/tests.log"
  exit 1
fi
cat "$OUT_ROOT/tests.log"
echo "===== SIX INSTANCE FOLDS / RIDGE AND TINY TRANSFORMER / NO PPO ====="
"$PYTHON" -u evaluate_m_operator_offline_v1.py \
  --collection-root "$COLLECTION_ROOT" --out "$OUT_ROOT" --threads 2
echo "===== OFFLINE COMPARISON COMPLETE ====="
