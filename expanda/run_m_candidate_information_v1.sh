#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
: "${INSTANCES:?set INSTANCES}"
: "${NORMALIZATION:?set NORMALIZATION}"
OUT_ROOT=${OUT_ROOT:-transformer_ppo_runs/diagnostics/ccp_m_candidate_information_v1}
PYTHON=${PYTHON:-python3}
MAX_WORKERS=${MAX_WORKERS:-2}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
mkdir -p "$OUT_ROOT"
echo '===== TEST CONCRETE CANDIDATES / NO OUTCOME LEAKAGE ====='
if "$PYTHON" -m unittest tests.test_m_candidate_information > "$OUT_ROOT/tests.log" 2>&1; then
  tail -n 5 "$OUT_ROOT/tests.log"
else
  cat "$OUT_ROOT/tests.log"
  exit 1
fi
echo '===== BOUNDED CANDIDATE-INFORMATION EXPERIMENT: CCP100 / POP20 / B400 ====='
"$PYTHON" -u run_m_candidate_information_v1.py --instances "$INSTANCES" --normalization "$NORMALIZATION" \
  --out "$OUT_ROOT" --workers "$MAX_WORKERS"
