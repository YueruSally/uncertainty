#!/usr/bin/env bash
set -euo pipefail

: "${GENERATED_INSTANCES:?set GENERATED_INSTANCES to the frozen 200-train/20-validation catalog}"
: "${COLLECTION_ROOT:?set COLLECTION_ROOT to the completed I-K8 collection}"
: "${CURVE_ROOT:?set CURVE_ROOT to the completed I-K8 size-200 Transformer}"
: "${DEVELOPMENT_ROOT:?set DEVELOPMENT_ROOT to the passed I-K8 development search}"
: "${NORMALIZATION:?set NORMALIZATION to reward_normalization.json}"
: "${PPO_ROOT:?set PPO_ROOT}"
: "${OUT_ROOT:?set OUT_ROOT}"

mkdir -p "$PPO_ROOT" "$OUT_ROOT"

echo "===== TEST I-K8 TRANSFORMER-PPO TRAINING AND VALIDATION SELECTION ====="
python3 -m unittest \
  tests.test_i_k8_initialization \
  tests.test_i_k8_transformer_scale \
  tests.test_i_k8_ppo_v1 \
  > "$OUT_ROOT/tests.log" 2>&1
cat "$OUT_ROOT/tests.log"

echo "===== STAGE 1: I-K8 PPO ON 200 TRAINING INSTANCES ONLY ====="
python3 -u train_i_k8_ppo_v1.py \
  --collection "$COLLECTION_ROOT" \
  --curve-root "$CURVE_ROOT" \
  --development-root "$DEVELOPMENT_ROOT" \
  --out "$PPO_ROOT"

echo "===== STAGE 2: SELECT I-K8 PPO CHECKPOINT ON 20 VALIDATION INSTANCES ====="
python3 -u experiments/run_i_k8_ppo_selection_v1.py \
  --generated-instances "$GENERATED_INSTANCES" \
  --curve-root "$CURVE_ROOT" \
  --development-root "$DEVELOPMENT_ROOT" \
  --normalization "$NORMALIZATION" \
  --ppo-root "$PPO_ROOT" \
  --out "$OUT_ROOT" \
  --workers "${MAX_WORKERS:-3}"

echo "===== I-K8 TRANSFORMER-PPO FROZEN; FINAL TEST REMAINS LOCKED ====="
cat "$OUT_ROOT/QUICK_RESULT.json"
