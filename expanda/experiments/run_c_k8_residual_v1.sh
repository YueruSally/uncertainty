#!/usr/bin/env bash
set -euo pipefail

: "${C_COLLECTION_ROOT:?set C_COLLECTION_ROOT}"
: "${C_SOURCE_SEARCH_ROOT:?set C_SOURCE_SEARCH_ROOT}"
: "${SEARCH_INSTANCES:?set SEARCH_INSTANCES}"
: "${NORMALIZATION:?set NORMALIZATION}"
: "${TRAINING_ROOT:?set TRAINING_ROOT}"
: "${OUT_ROOT:?set OUT_ROOT}"

MAX_WORKERS="${MAX_WORKERS:-3}"
mkdir -p "$TRAINING_ROOT" "$OUT_ROOT"

echo "===== TEST C-K8 SIMPLE RESIDUAL GATE ====="
python3 -m unittest \
  tests.test_c_k8_residual \
  tests.test_c_k8_crossover

echo "===== STAGE 1: ZERO-EVALUATION C-K8 RESIDUAL TRAINING ====="
python3 -u train_c_k8_residual_v1.py \
  --collection "$C_COLLECTION_ROOT" \
  --out "$TRAINING_ROOT"

echo "===== STAGE 2: 3-INSTANCE / 10-SEED / B400 PAIRED SEARCH ====="
python3 -u experiments/run_c_k8_residual_search_v1.py \
  --search-instances "$SEARCH_INSTANCES" \
  --training-root "$TRAINING_ROOT" \
  --normalization "$NORMALIZATION" \
  --bundle "$TRAINING_ROOT/BUNDLE.json" \
  --source-search "$C_SOURCE_SEARCH_ROOT" \
  --out "$OUT_ROOT" \
  --workers "$MAX_WORKERS"

echo "===== C-K8 SIMPLE RESIDUAL EXPERIMENT COMPLETE ====="
