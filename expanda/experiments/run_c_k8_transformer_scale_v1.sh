#!/usr/bin/env bash
set -euo pipefail

: "${GENERATED_INSTANCES:?set GENERATED_INSTANCES to the existing 200-train/20-validation catalog}"
: "${SEARCH_INSTANCES:?set SEARCH_INSTANCES to ccp_m_prescreen_scale_search_instances_v1}"
: "${C_ORACLE_ROOT:?set C_ORACLE_ROOT to the completed C-K8 oracle screen}"
: "${NORMALIZATION:?set NORMALIZATION to reward_normalization.json}"
: "${COLLECTION_ROOT:?set COLLECTION_ROOT}"
: "${TRAINING_ROOT:?set TRAINING_ROOT}"
: "${OUT_ROOT:?set OUT_ROOT}"

mkdir -p "$COLLECTION_ROOT" "$TRAINING_ROOT" "$OUT_ROOT"

echo "===== TEST C-K8 TRANSFORMER SCALE PIPELINE ====="
python3 -m unittest \
  tests.test_c_k8_crossover \
  tests.test_c_k8_transformer_scale \
  > "$OUT_ROOT/tests.log" 2>&1
cat "$OUT_ROOT/tests.log"

echo "===== STAGE 1: COLLECT 200 TRAIN / 20 VALIDATION C-K8 TRAJECTORIES ====="
python3 -u collect_c_k8_scale_v1.py \
  --generated-instances "$GENERATED_INSTANCES" \
  --normalization "$NORMALIZATION" \
  --out "$COLLECTION_ROOT" \
  --workers "${MAX_WORKERS:-4}"

echo "===== STAGE 2: TRAIN 25 / 50 / 100 / 200 INSTANCE TRANSFORMERS ====="
python3 -u train_c_k8_transformer_scale_v1.py \
  --collection "$COLLECTION_ROOT" \
  --out "$TRAINING_ROOT"

echo "===== STAGE 3: FROZEN TRANSFORMER PAIRED B400 DEVELOPMENT SEARCH ====="
python3 -u experiments/run_c_k8_transformer_search_v1.py \
  --search-instances "$SEARCH_INSTANCES" \
  --training-root "$TRAINING_ROOT" \
  --normalization "$NORMALIZATION" \
  --bundle "$TRAINING_ROOT/BUNDLE.json" \
  --oracle-root "$C_ORACLE_ROOT" \
  --out "$OUT_ROOT" \
  --workers "${SEARCH_WORKERS:-3}"

echo "===== C-K8 TRANSFORMER SCALE PIPELINE COMPLETE ====="
cat "$OUT_ROOT/QUICK_RESULT.json"
