#!/usr/bin/env bash
set -euo pipefail

: "${SEARCH_INSTANCES:?set SEARCH_INSTANCES}"
: "${NORMALIZATION:?set NORMALIZATION}"
: "${CURVE_ROOT:?set CURVE_ROOT}"
: "${SOURCE_SEARCH:?set SOURCE_SEARCH}"
: "${OUT_ROOT:?set OUT_ROOT}"

mkdir -p "$OUT_ROOT"

echo "===== TEST M-K8 REVIEW AUDIT ====="
python3 -m unittest \
  tests.test_m_prescreen_review_v1 \
  tests.test_m_prescreen_scale_search \
  tests.test_m_prescreen_final_oracle \
  > "$OUT_ROOT/tests.log" 2>&1
cat "$OUT_ROOT/tests.log"

echo "===== STAGE 1: SAME-INSTANCE K8 SIMPLE AND GREEDY ORACLE ====="
echo "===== STAGE 2: SHARED INDEPENDENT OOS1000 RE-EVALUATION ====="
python3 -u experiments/run_m_prescreen_review_v1.py \
  --search-instances "$SEARCH_INSTANCES" \
  --normalization "$NORMALIZATION" \
  --bundle "$CURVE_ROOT/BUNDLE.json" \
  --source-search "$SOURCE_SEARCH" \
  --out "$OUT_ROOT" \
  --workers "${MAX_WORKERS:-3}"

echo "===== M-K8 REVIEW COMPLETE ====="
cat "$OUT_ROOT/QUICK_RESULT.json"
