#!/usr/bin/env bash
set -euo pipefail

INSTANCES=${INSTANCES:-/home/lunet/cokt2/uncertainty-transformer-ppo/expanda/transformer_ppo_runs/instances}
NORMALIZATION=${NORMALIZATION:-/home/lunet/cokt2/uncertainty-transformer-ppo/expanda/transformer_ppo_runs/reward_normalization.json}
REUSE_BASE_ROOT=${REUSE_BASE_ROOT:-transformer_ppo_runs/diagnostics/ccp_m_directed_balanced_validation_v1_10inst_10seed_b400}
OUT_ROOT=${OUT_ROOT:-transformer_ppo_runs/diagnostics/ccp_i_directed_balanced_hv_gate_v1_10inst_fresh7seed_b400}

if ! test -e "$NORMALIZATION"; then
  echo "MISSING REQUIRED INPUT: $NORMALIZATION" >&2
  exit 1
fi

mkdir -p "$OUT_ROOT"

echo "===== TEST FIXED-BUDGET I INITIAL HV GATE ====="
python3 -u -m unittest \
  tests.test_initialization_rule \
  > "$OUT_ROOT/tests.log" 2>&1

echo "===== B80 FIXED-BUDGET I HV-GATE SMOKE ====="
SMOKE_ROOT="$OUT_ROOT/smoke_S0_002_seed_779911_b80"
if ! test -f "$SMOKE_ROOT/COMPLETE.json"; then
  python3 -u run_learning_components_deterministic.py \
    --instance "$INSTANCES/train-S0-002.json" \
    --normalization "$NORMALIZATION" \
    --out "$SMOKE_ROOT" \
    --component RULE-I \
    --initialization-rule-profile joint_balanced_hv_gate \
    --algorithm-seed 779911 \
    --evaluation-mode ccp \
    --uncertainty-profile transport_border_saa_v1 \
    --scenario-count 200 \
    --ccp-confidence 0.90 \
    --evaluation-budget 80 \
    --pop 20 \
    --gens 1000 \
    --device cpu \
    > "$OUT_ROOT/smoke.log" 2>&1
fi

if ! jq -e '
  .completed == true
  and .real_objective_evaluation_count == 80
  and .initialization_attempts == 8
  and .counterfactual_initialization_evaluation_count == 8
  and (
    .initialization_hv_gate_selected == "base"
    or .initialization_hv_gate_selected == "directed"
  )
' "$SMOKE_ROOT/COMPLETE.json" >/dev/null; then
  echo "I HV-GATE SMOKE FAILED" >&2
  exit 1
fi

echo "===== FRESH 10-INSTANCE / 7-SEED I HV-GATE VALIDATION ====="
python3 -u evaluate_i_directed_initialization.py \
  --instances "$INSTANCES" \
  --normalization "$NORMALIZATION" \
  --reuse-base-root "$REUSE_BASE_ROOT" \
  --out "$OUT_ROOT" \
  --instance-ids \
    train-S0-002 train-S0-005 train-S0-006 train-S0-007 train-S0-008 \
    train-S0-009 train-S0-010 train-S0-011 train-S0-012 train-S0-013 \
  --algorithm-seeds \
    777011 777012 777013 777014 777015 777016 777017 \
  --strategies BASE I_DIRECTED_BALANCED_HV_GATE \
  --max-workers 12 \
  --evaluation-budget 400 \
  --scenario-count 200 \
  --ccp-confidence 0.90 \
  --pop 20 \
  --gens 1000 \
  --device cpu \
  | tee "$OUT_ROOT/evaluation.log"

if ! jq -e '
  .completed == true
  and .instance_count == 10
  and .algorithm_seed_count == 7
  and .run_count == 140
  and ([.status_counts[]] | add) == 140
  and .scenario_exact == true
  and (.strategies | length) == 2
  and .best_directed_strategy.strategy
      == "I_DIRECTED_BALANCED_HV_GATE"
  and .best_directed_strategy.mean_counterfactual_initialization_evaluations
      == 8
' "$OUT_ROOT/COMPLETE.json" >/dev/null; then
  echo "I DIRECTED BALANCED HV-GATE VALIDATION FAILED" >&2
  exit 1
fi

touch "$OUT_ROOT/PIPELINE_COMPLETE"

echo "===== I DIRECTED BALANCED HV-GATE VALIDATION COMPLETE ====="
cat "$OUT_ROOT/QUICK_RESULT.json"
