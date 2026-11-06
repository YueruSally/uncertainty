#!/usr/bin/env bash
set -euo pipefail

: "${INSTANCES:?set INSTANCES to the frozen source instance catalog}"
: "${NORMALIZATION:?set NORMALIZATION to reward_normalization.json}"
: "${GENERATED_ROOT:?set GENERATED_ROOT}"
: "${COLLECTION_ROOT:?set COLLECTION_ROOT}"
: "${OUT_ROOT:?set OUT_ROOT}"

mkdir -p "$GENERATED_ROOT" "$COLLECTION_ROOT" "$OUT_ROOT"

echo "===== TEST M-K8 SCALE PIPELINE ====="
python3 -m unittest \
  tests.test_m_prescreen_scale_v1 \
  tests.test_m_prescreen_final_oracle \
  tests.test_m_prescreen_k8 \
  > "$OUT_ROOT/tests.log" 2>&1
cat "$OUT_ROOT/tests.log"

echo "===== STAGE 1: GENERATE 200 TRAIN / 20 VALIDATION S0 INSTANCES ====="
python3 -u generate_m_prescreen_scale_instances.py \
  --source-instances "$INSTANCES" \
  --out "$GENERATED_ROOT"

echo "===== STAGE 2: COLLECT 12-30 TEMPORALLY BALANCED K8 STATES PER TRAJECTORY ====="
python3 -u collect_m_prescreen_scale_v1.py \
  --generated-instances "$GENERATED_ROOT" \
  --normalization "$NORMALIZATION" \
  --out "$COLLECTION_ROOT" \
  --workers "${MAX_WORKERS:-4}"

echo "===== STAGE 3: 25 / 50 / 100 / 200 INSTANCE LEARNING CURVE ====="
python3 -u train_m_prescreen_scale_curve_v1.py \
  --collection "$COLLECTION_ROOT" \
  --normalization "$NORMALIZATION" \
  --out "$OUT_ROOT"

echo "===== M-K8 SCALE PIPELINE COMPLETE ====="
cat "$OUT_ROOT/QUICK_RESULT.json"
