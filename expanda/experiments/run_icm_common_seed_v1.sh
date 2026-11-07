#!/usr/bin/env bash
set -euo pipefail

: "${SEARCH_INSTANCES:?set SEARCH_INSTANCES}"
: "${NORMALIZATION:?set NORMALIZATION}"
: "${I_BUNDLE:?set I_BUNDLE}"
: "${C_BUNDLE:?set C_BUNDLE}"
: "${M_BUNDLE:?set M_BUNDLE}"
: "${OUT_ROOT:?set OUT_ROOT}"

mkdir -p "$OUT_ROOT"

echo "===== TEST COMMON-SEED I / C / M COMPONENT SCREEN ====="
python3 -m unittest \
  tests.test_icm_common_seed \
  tests.test_i_k8_residual \
  tests.test_c_k8_residual \
  tests.test_c_k8_crossover

echo "===== 3 INSTANCES / 10 COMMON SEEDS / 8 METHODS / CCP100 / POP20 / B400 ====="
python3 -u experiments/run_icm_common_seed_v1.py \
  --search-instances "$SEARCH_INSTANCES" \
  --normalization "$NORMALIZATION" \
  --i-bundle "$I_BUNDLE" \
  --c-bundle "$C_BUNDLE" \
  --m-bundle "$M_BUNDLE" \
  --out "$OUT_ROOT" \
  --workers "${MAX_WORKERS:-3}"

echo "===== COMMON-SEED I / C / M SCREEN COMPLETE ====="
