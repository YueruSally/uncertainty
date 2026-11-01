#!/usr/bin/env bash
set -euo pipefail

INSTANCES=${INSTANCES:-/home/lunet/cokt2/uncertainty-transformer-ppo/expanda/transformer_ppo_runs/instances}
NORMALIZATION=${NORMALIZATION:-/home/lunet/cokt2/uncertainty-transformer-ppo/expanda/transformer_ppo_runs/reward_normalization.json}
VALIDATION_ROOT=${VALIDATION_ROOT:-transformer_ppo_runs/diagnostics/ccp_i_directed_balanced_validation_v1_10inst_holdout7seed_b400}
SMOKE_ROOT=${SMOKE_ROOT:-transformer_ppo_runs/diagnostics/ccp_i_directed_balanced_oos_v1_smoke_1inst_1seed_s200}
OUT_ROOT=${OUT_ROOT:-transformer_ppo_runs/diagnostics/ccp_i_directed_balanced_oos_v1_10inst_holdout7seed_s1000}

for REQUIRED in \
  "$NORMALIZATION" \
  "$VALIDATION_ROOT/PIPELINE_COMPLETE" \
  "$VALIDATION_ROOT/COMPLETE.json"
do
  if ! test -f "$REQUIRED"; then
    echo "MISSING REQUIRED INPUT: $REQUIRED" >&2
    exit 1
  fi
done

mkdir -p "$SMOKE_ROOT" "$OUT_ROOT"

echo "===== TEST INDEPENDENT I OOS EVALUATOR ====="
python3 -u -m unittest \
  tests.test_m_directed_balanced_oos \
  > "$OUT_ROOT/tests.log" 2>&1

echo "===== I OOS SMOKE: 1 INSTANCE / 1 SEED / 200 NEW SCENARIOS ====="
python3 -u evaluate_m_directed_balanced_oos.py \
  --instances "$INSTANCES" \
  --normalization "$NORMALIZATION" \
  --validation-root "$VALIDATION_ROOT" \
  --out "$SMOKE_ROOT" \
  --instance-ids train-S0-002 \
  --algorithm-seeds 777004 \
  --holdout-seeds 777004 \
  --comparison-strategy I_DIRECTED_BALANCED \
  --scenario-count 200 \
  --oos-seed 880101 \
  --alpha 0.90 \
  --uncertainty-profile transport_border_saa_v1 \
  --max-workers 1 \
  | tee "$SMOKE_ROOT/evaluation.log"

if ! jq -e '
  .completed == true
  and .instance_count == 1
  and .paired_run_count == 1
  and .oos_scenario_count == 200
  and .comparison_strategy == "I_DIRECTED_BALANCED"
  and .oos_distinct_from_training == true
  and .one_common_oos_scenario_set == true
' "$SMOKE_ROOT/OOS_COMPLETE.json" >/dev/null; then
  echo "I DIRECTED BALANCED OOS SMOKE FAILED" >&2
  exit 1
fi

echo "===== I FORMAL OOS: 10 INSTANCES / 7 HELD-OUT SEEDS / 1000 NEW SCENARIOS ====="
python3 -u evaluate_m_directed_balanced_oos.py \
  --instances "$INSTANCES" \
  --normalization "$NORMALIZATION" \
  --validation-root "$VALIDATION_ROOT" \
  --out "$OUT_ROOT" \
  --instance-ids \
    train-S0-002 train-S0-005 train-S0-006 train-S0-007 train-S0-008 \
    train-S0-009 train-S0-010 train-S0-011 train-S0-012 train-S0-013 \
  --algorithm-seeds \
    777004 777005 777006 777007 777008 777009 777010 \
  --holdout-seeds \
    777004 777005 777006 777007 777008 777009 777010 \
  --comparison-strategy I_DIRECTED_BALANCED \
  --scenario-count 1000 \
  --oos-seed 880101 \
  --alpha 0.90 \
  --uncertainty-profile transport_border_saa_v1 \
  --max-workers 5 \
  | tee "$OUT_ROOT/evaluation.log"

if ! jq -e '
  .completed == true
  and .instance_count == 10
  and .algorithm_seed_count == 7
  and .paired_run_count == 70
  and .oos_scenario_count == 1000
  and .comparison_strategy == "I_DIRECTED_BALANCED"
  and .oos_distinct_from_training == true
  and .one_common_oos_scenario_set == true
  and .held_out_seeds.run_count == 70
' "$OUT_ROOT/OOS_COMPLETE.json" >/dev/null; then
  echo "I DIRECTED BALANCED OOS VALIDATION FAILED" >&2
  exit 1
fi

touch "$OUT_ROOT/PIPELINE_COMPLETE"

echo "===== I DIRECTED BALANCED INDEPENDENT OOS COMPLETE ====="
cat "$OUT_ROOT/QUICK_RESULT.json"
