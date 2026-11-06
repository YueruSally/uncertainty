#!/usr/bin/env bash
set -euo pipefail

: "${SOURCE_INSTANCES:?set SOURCE_INSTANCES to the original frozen instance catalog}"
: "${NORMALIZATION:?set NORMALIZATION to reward_normalization.json}"
: "${FREEZE_ROOT:?set FREEZE_ROOT to the completed 20-validation PPO selection}"
: "${FINAL_INSTANCES:?set FINAL_INSTANCES to a new locked final-instance directory}"
: "${OUT_ROOT:?set OUT_ROOT to a new formal result directory}"

mkdir -p "$OUT_ROOT"

echo "===== TEST LOCKED FOUR-METHOD K8 FORMAL PIPELINE ====="
python3 -m unittest \
  tests.test_m_prescreen_formal_v1 \
  tests.test_m_prescreen_ppo_v1 \
  tests.test_s0_final_test_instances \
  tests.test_m_prescreen_scale_search \
  > "$OUT_ROOT/tests.log" 2>&1
cat "$OUT_ROOT/tests.log"

echo "===== STAGE 1: OPEN LOCKED 98M FINAL TEST INSTANCES ====="
if test ! -f "$FINAL_INSTANCES/catalog.json"; then
  python3 -u generate_s0_final_test_instances.py \
    --source-instances "$SOURCE_INSTANCES" \
    --out "$FINAL_INSTANCES" \
    --count 10 \
    --master-seed 98000000 \
    --ccp-seed-base 98100000 \
    --oos-seed-base 98200000
fi
printf 'final_instances=%s\nopened_before_optimization=true\n' \
  "$FINAL_INSTANCES" > "$OUT_ROOT/FINAL_TEST_OPENED"

echo "===== STAGE 2: BASE / BALANCED / SUPERVISED / PPO; CCP100 / POP20 / B400 ====="
echo "===== STAGE 3: INDEPENDENT OOS5000 RE-EVALUATION ====="
python3 -u experiments/run_m_prescreen_formal_v1.py \
  --final-instances "$FINAL_INSTANCES" \
  --normalization "$NORMALIZATION" \
  --freeze-root "$FREEZE_ROOT" \
  --out "$OUT_ROOT" \
  --workers "${MAX_WORKERS:-3}"

echo "===== LOCKED FOUR-METHOD K8 FORMAL EXPERIMENT COMPLETE ====="
cat "$OUT_ROOT/QUICK_RESULT.json"
