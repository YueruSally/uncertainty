#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
: "${INSTANCES:?set INSTANCES}"
: "${NORMALIZATION:?set NORMALIZATION}"
BUNDLE=${BUNDLE:-transformer_ppo_runs/diagnostics/ccp_m_candidate_information_v1/training/BUNDLE.json}
OUT_ROOT=${OUT_ROOT:-transformer_ppo_runs/diagnostics/ccp_m_single_intervention_ppo_v2}
PYTHON=${PYTHON:-python3}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
mkdir -p "$OUT_ROOT"
echo '===== TEST SINGLE INTERVENTION / IDENTICAL TRAIN AND TEST POLICY ====='
if "$PYTHON" -m unittest tests.test_m_single_intervention_ppo tests.test_m_terminal_hv_ppo tests.test_m_candidate_information > "$OUT_ROOT/tests.log" 2>&1; then
  tail -n 5 "$OUT_ROOT/tests.log"
else
  cat "$OUT_ROOT/tests.log"
  exit 1
fi
echo '===== SINGLE-INTERVENTION PPO: CCP100 / POP20 / B400 ====='
"$PYTHON" -u run_m_terminal_hv_ppo_v1.py --single-intervention --instances "$INSTANCES" \
  --normalization "$NORMALIZATION" --bundle "$BUNDLE" --out "$OUT_ROOT" --workers "${MAX_WORKERS:-2}"
