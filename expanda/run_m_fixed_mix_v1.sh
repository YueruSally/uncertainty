#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
: "${INSTANCES:?set INSTANCES}"
: "${NORMALIZATION:?set NORMALIZATION}"
OUT_ROOT=${OUT_ROOT:-transformer_ppo_runs/diagnostics/ccp_m_fixed_mix_v1_2inst_20seed_b400}
MAX_WORKERS=${MAX_WORKERS:-2}
PYTHON=${PYTHON:-python3}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
mkdir -p "$OUT_ROOT"
echo '===== TEST FIXED BALANCED80 / MODE20 MIXTURE ====='
if "$PYTHON" -m unittest tests.test_m_operator_fixed_mix > "$OUT_ROOT/tests.log" 2>&1; then
  tail -n 5 "$OUT_ROOT/tests.log"
else
  cat "$OUT_ROOT/tests.log"
  exit 1
fi
echo '===== 80 SEARCHES: CCP100 / POP20 / B400 / 20 PAIRED SEEDS ====='
"$PYTHON" -u run_m_fixed_mix_v1.py --instances "$INSTANCES" --normalization "$NORMALIZATION" \
  --out "$OUT_ROOT" --workers "$MAX_WORKERS"
