#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${INSTANCES:?set INSTANCES}"
: "${NORMALIZATION:?set NORMALIZATION}"
SOURCE_ROOT=${SOURCE_ROOT:-transformer_ppo_runs/diagnostics/ccp_m_single_intervention_ppo_v2}
OUT_ROOT=${OUT_ROOT:-transformer_ppo_runs/diagnostics/ccp_m_complete_terminal_feedback_v3}
PYTHON=${PYTHON:-python3}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
mkdir -p "$OUT_ROOT"
echo '===== TEST COMPLETE TERMINAL FEEDBACK / FROZEN CCP100 ====='
if "$PYTHON" -m unittest tests.test_m_terminal_feedback_v3 tests.test_m_single_intervention_ppo tests.test_m_terminal_hv_ppo > "$OUT_ROOT/tests.log" 2>&1; then
  tail -n 5 "$OUT_ROOT/tests.log"
else
  cat "$OUT_ROOT/tests.log"
  exit 1
fi
echo '===== COMPLETE FEEDBACK / PRETRAINING / PPO / PAIRED SEARCH ====='
"$PYTHON" -u experiments/m_terminal_feedback_v3.py --source "$SOURCE_ROOT" \
  --instances "$INSTANCES" --normalization "$NORMALIZATION" --out "$OUT_ROOT" --workers "${MAX_WORKERS:-2}"
