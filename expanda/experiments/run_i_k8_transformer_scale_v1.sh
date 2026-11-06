#!/usr/bin/env bash
set -euo pipefail

: "${GENERATED_INSTANCES:?set GENERATED_INSTANCES to the existing 200-train/20-validation scale catalog}"
: "${SEARCH_INSTANCES:?set SEARCH_INSTANCES to ccp_m_prescreen_scale_search_instances_v1}"
: "${I_ORACLE_ROOT:?set I_ORACLE_ROOT to the completed I-K8 oracle screen}"
: "${NORMALIZATION:?set NORMALIZATION to reward_normalization.json}"
: "${COLLECTION_ROOT:?set COLLECTION_ROOT}"
: "${TRAINING_ROOT:?set TRAINING_ROOT}"
: "${OUT_ROOT:?set OUT_ROOT}"

mkdir -p "$COLLECTION_ROOT" "$TRAINING_ROOT" "$OUT_ROOT"

echo "===== TEST I-K8 TRANSFORMER SCALE PIPELINE ====="
python3 -m unittest \
  tests.test_i_k8_initialization \
  tests.test_i_k8_transformer_scale \
  > "$OUT_ROOT/tests.log" 2>&1
cat "$OUT_ROOT/tests.log"

echo "===== STAGE 1: COLLECT 200 TRAIN / 20 VALIDATION I-K8 TRAJECTORIES ====="
python3 -u collect_i_k8_scale_v1.py \
  --generated-instances "$GENERATED_INSTANCES" \
  --normalization "$NORMALIZATION" \
  --out "$COLLECTION_ROOT" \
  --workers "${MAX_WORKERS:-4}"

echo "===== STAGE 2: TRAIN 25 / 50 / 100 / 200 INSTANCE TRANSFORMERS ====="
python3 -u train_i_k8_transformer_scale_v1.py \
  --collection "$COLLECTION_ROOT" \
  --out "$TRAINING_ROOT"

echo "===== STAGE 3: FROZEN TRANSFORMER PAIRED B400 DEVELOPMENT SEARCH ====="
python3 -u experiments/run_i_k8_transformer_search_v1.py \
  --search-instances "$SEARCH_INSTANCES" \
  --training-root "$TRAINING_ROOT" \
  --normalization "$NORMALIZATION" \
  --bundle "$TRAINING_ROOT/BUNDLE.json" \
  --oracle-root "$I_ORACLE_ROOT" \
  --out "$OUT_ROOT" \
  --workers "${SEARCH_WORKERS:-3}"

echo "===== I-K8 TRANSFORMER SCALE PIPELINE COMPLETE ====="
cat "$OUT_ROOT/QUICK_RESULT.json"
