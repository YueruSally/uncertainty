#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
: "${INSTANCES:?set INSTANCES to the existing catalog directory}"
: "${NORMALIZATION:?set NORMALIZATION}"
PATH_ROOT=${PATH_ROOT:-transformer_ppo_runs/diagnostics/ccp_m_operator_path_representation_v1_384state}
COLLECTION_ROOT=${COLLECTION_ROOT:-transformer_ppo_runs/diagnostics/ccp_m_operator_feedback_v2_6inst_2seed_b400}
OUT_ROOT=${OUT_ROOT:-transformer_ppo_runs/diagnostics/ccp_m_path_closed_loop_v1_2inst_3seed_b400}
PYTHON=${PYTHON:-python3}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
mkdir -p "$OUT_ROOT"
echo '===== TEST FROZEN POLICY INFERENCE ====='
if "$PYTHON" -m unittest tests.test_m_operator_closed_loop > "$OUT_ROOT/tests.log" 2>&1; then
  tail -n 5 "$OUT_ROOT/tests.log"
else
  echo '===== TEST FAILED: details from tests.log ====='
  tail -n 80 "$OUT_ROOT/tests.log"
  exit 1
fi
echo '===== CLOSED LOOP: 18 SEARCHES / CCP100 / POP20 / B400 / NO TRAINING ====='
"$PYTHON" -u run_m_operator_closed_loop_v1.py --instances "$INSTANCES" --normalization "$NORMALIZATION" \
  --path-root "$PATH_ROOT" --collection-root "$COLLECTION_ROOT" --out "$OUT_ROOT"
