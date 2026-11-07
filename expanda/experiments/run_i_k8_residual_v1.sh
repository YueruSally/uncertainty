#!/usr/bin/env bash
set -euo pipefail

: "${I_COLLECTION_ROOT:?set I_COLLECTION_ROOT}"
: "${I_SOURCE_SEARCH_ROOT:?set I_SOURCE_SEARCH_ROOT}"
: "${SEARCH_INSTANCES:?set SEARCH_INSTANCES}"
: "${NORMALIZATION:?set NORMALIZATION}"
: "${TRAINING_ROOT:?set TRAINING_ROOT}"
: "${OUT_ROOT:?set OUT_ROOT}"

MAX_WORKERS="${MAX_WORKERS:-3}"
mkdir -p "$TRAINING_ROOT" "$OUT_ROOT"

echo "===== TEST I-K8 DIRECTED RESIDUAL GATE ====="
python3 -m unittest \
  tests.test_i_k8_residual \
  tests.test_i_k8_initialization

echo "===== STAGE 1: ZERO-EVALUATION I-K8 RESIDUAL TRAINING ====="
python3 -u train_i_k8_residual_v1.py \
  --collection "$I_COLLECTION_ROOT" \
  --out "$TRAINING_ROOT"

echo "===== STAGE 2: 3-INSTANCE / 10-SEED / B400 PAIRED SEARCH ====="
python3 -u experiments/run_i_k8_residual_search_v1.py \
  --search-instances "$SEARCH_INSTANCES" \
  --training-root "$TRAINING_ROOT" \
  --normalization "$NORMALIZATION" \
  --bundle "$TRAINING_ROOT/BUNDLE.json" \
  --source-search "$I_SOURCE_SEARCH_ROOT" \
  --out "$OUT_ROOT" \
  --workers "$MAX_WORKERS"

echo "===== I-K8 DIRECTED RESIDUAL EXPERIMENT COMPLETE ====="
