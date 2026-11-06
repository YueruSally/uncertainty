#!/usr/bin/env bash
set -euo pipefail

: "${INSTANCES:?set INSTANCES to ccp_m_prescreen_scale_search_instances_v1}"
: "${NORMALIZATION:?set NORMALIZATION to reward_normalization.json}"
: "${OUT_ROOT:?set OUT_ROOT to a fresh I-K8 oracle result directory}"

mkdir -p "$OUT_ROOT"

echo "===== TEST I-K8 COMPLETE-INDIVIDUAL INITIALIZATION ====="
python3 -m unittest \
  tests.test_i_k8_initialization \
  tests.test_initialization_rule \
  > "$OUT_ROOT/tests.log" 2>&1
cat "$OUT_ROOT/tests.log"

echo "===== I-K8 FINAL-HV ORACLE: CCP100 / POP20 / B400 ====="
python3 -u experiments/run_i_k8_final_oracle_v1.py \
  --instances "$INSTANCES" \
  --normalization "$NORMALIZATION" \
  --out "$OUT_ROOT" \
  --workers "${MAX_WORKERS:-2}"

cat "$OUT_ROOT/QUICK_RESULT.json"
