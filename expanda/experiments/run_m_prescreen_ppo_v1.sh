#!/usr/bin/env bash
set -euo pipefail

: "${SCALE_INSTANCES:?set SCALE_INSTANCES to the completed 200/20 generated instances}"
: "${COLLECTION_ROOT:?set COLLECTION_ROOT to the completed scale collection}"
: "${CURVE_ROOT:?set CURVE_ROOT to the completed size-200 learning curve}"
: "${NORMALIZATION:?set NORMALIZATION to reward_normalization.json}"
: "${PPO_ROOT:?set PPO_ROOT}"
: "${OUT_ROOT:?set OUT_ROOT}"

mkdir -p "$PPO_ROOT" "$OUT_ROOT"

echo "===== TEST K8 TRANSFORMER-PPO TRAINING AND VALIDATION SELECTION ====="
python3 -m unittest \
  tests.test_m_prescreen_ppo_v1 \
  tests.test_m_prescreen_scale_search \
  tests.test_m_prescreen_scale_v1 \
  > "$OUT_ROOT/tests.log" 2>&1
cat "$OUT_ROOT/tests.log"

echo "===== STAGE 1: PPO ON 200 TRAINING INSTANCES ONLY ====="
python3 -u train_m_prescreen_ppo_v1.py \
  --collection "$COLLECTION_ROOT" \
  --curve-root "$CURVE_ROOT" \
  --normalization "$NORMALIZATION" \
  --out "$PPO_ROOT"

echo "===== STAGE 2: SELECT PPO CHECKPOINT ON 20 VALIDATION INSTANCES ====="
python3 -u experiments/run_m_prescreen_ppo_selection_v1.py \
  --scale-instances "$SCALE_INSTANCES" \
  --curve-root "$CURVE_ROOT" \
  --normalization "$NORMALIZATION" \
  --ppo-root "$PPO_ROOT" \
  --out "$OUT_ROOT" \
  --workers "${MAX_WORKERS:-3}"

echo "===== K8 TRANSFORMER-PPO FROZEN; FINAL TEST REMAINS LOCKED ====="
cat "$OUT_ROOT/QUICK_RESULT.json"
