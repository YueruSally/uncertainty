#!/usr/bin/env bash
set -euo pipefail

: "${INSTANCES:?set INSTANCES to the frozen source instance catalog}"
: "${SCALE_INSTANCES:?set SCALE_INSTANCES to the completed 200/20 generated instances}"
: "${COLLECTION_ROOT:?set COLLECTION_ROOT to the completed scale collection}"
: "${CURVE_ROOT:?set CURVE_ROOT to the completed scale learning curve}"
: "${NORMALIZATION:?set NORMALIZATION to reward_normalization.json}"
: "${SEARCH_INSTANCES:?set SEARCH_INSTANCES}"
: "${OUT_ROOT:?set OUT_ROOT}"

mkdir -p "$SEARCH_INSTANCES" "$OUT_ROOT"

echo "===== TEST FROZEN SIZE-200 K8 SEARCH ====="
python3 -m unittest \
  tests.test_m_prescreen_scale_search \
  tests.test_m_prescreen_scale_v1 \
  tests.test_m_prescreen_k8 \
  > "$OUT_ROOT/tests.log" 2>&1
cat "$OUT_ROOT/tests.log"

echo "===== STAGE 1: GENERATE THREE UNTOUCHED TRAINING-DOMAIN DEV INSTANCES ====="
python3 -u generate_m_prescreen_search_instances.py \
  --source-instances "$INSTANCES" \
  --scale-instances "$SCALE_INSTANCES" \
  --out "$SEARCH_INSTANCES"

echo "===== STAGE 2: BALANCED VS FROZEN SIZE-200 K8 / CCP100 / POP20 / B400 ====="
python3 -u experiments/run_m_prescreen_scale_search_v1.py \
  --search-instances "$SEARCH_INSTANCES" \
  --collection "$COLLECTION_ROOT" \
  --curve-root "$CURVE_ROOT" \
  --normalization "$NORMALIZATION" \
  --bundle "$CURVE_ROOT/BUNDLE.json" \
  --out "$OUT_ROOT" \
  --workers "${MAX_WORKERS:-3}"

echo "===== FROZEN SIZE-200 K8 SEARCH COMPLETE ====="
cat "$OUT_ROOT/QUICK_RESULT.json"
