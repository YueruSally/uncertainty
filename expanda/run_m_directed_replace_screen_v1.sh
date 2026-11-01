#!/usr/bin/env bash
set -euo pipefail

INSTANCES=${INSTANCES:-/home/lunet/cokt2/uncertainty-transformer-ppo/expanda/transformer_ppo_runs/instances}
NORMALIZATION=${NORMALIZATION:-/home/lunet/cokt2/uncertainty-transformer-ppo/expanda/transformer_ppo_runs/reward_normalization.json}
CHECKPOINT=${CHECKPOINT:-transformer_ppo_runs/diagnostics/v5r_joint_replace_smoke_b160_p20_seed9501/M/final_policy.pt}
SOURCE_ROOT=${SOURCE_ROOT:-transformer_ppo_runs/diagnostics/ccp_m_late_replace_frequency_v1_3inst_3seed_b400}
DIAGNOSTIC_ROOT=${DIAGNOSTIC_ROOT:-transformer_ppo_runs/diagnostics/ccp_m_reversible_probe_diagnostic_v2_10inst_3seed_b400}
GRID_ROOT=${GRID_ROOT:-transformer_ppo_runs/diagnostics/ccp_m_replace_schedule_grid_v1_10inst_3seed_b400}
OUT_ROOT=${OUT_ROOT:-transformer_ppo_runs/diagnostics/ccp_m_directed_replace_screen_v1_10inst_3seed_b400}

for REQUIRED in \
  "$NORMALIZATION" \
  "$CHECKPOINT" \
  "$SOURCE_ROOT/COMPLETE.json" \
  "$DIAGNOSTIC_ROOT/COMPLETE.json" \
  "$GRID_ROOT/COMPLETE.json"
do
  if ! test -f "$REQUIRED"; then
    echo "MISSING REQUIRED INPUT: $REQUIRED" >&2
    exit 1
  fi
done

mkdir -p "$OUT_ROOT"

echo "===== TEST ZERO-EXTRA-EVALUATION DIRECTED REPLACE ====="
python3 -u -m unittest \
  tests.test_m_late_replace_frequency \
  tests.test_learning_components_v2_controls.V2ControlTests.test_joint_extreme_replace_selects_global_cost_reduction \
  > "$OUT_ROOT/tests.log" 2>&1

echo "===== SCREEN RANDOM / EXTREME / BALANCED / PARETO REPLACE ====="
python3 -u evaluate_m_late_replace_frequency.py \
  --instances "$INSTANCES" \
  --normalization "$NORMALIZATION" \
  --checkpoint "$CHECKPOINT" \
  --reuse-roots "$GRID_ROOT" "$DIAGNOSTIC_ROOT" "$SOURCE_ROOT" \
  --strategies \
    BASE REPLACE_G10_P100 \
    REPLACE_G10_JOINT_EXTREME \
    REPLACE_G10_JOINT_BALANCED \
    REPLACE_G10_JOINT_PARETO \
  --out "$OUT_ROOT" \
  --instance-ids \
    train-S0-002 train-S0-005 train-S0-006 train-S0-007 train-S0-008 \
    train-S0-009 train-S0-010 train-S0-011 train-S0-012 train-S0-013 \
  --algorithm-seeds 777001 777002 777003 \
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
  and .algorithm_seed_count == 3
  and .run_count == 150
  and ([.status_counts[]] | add) == 150
  and .scenario_exact == true
  and (.strategies | length) == 5
  and (.per_instance | length) == 10
  and (.directed_replace_diagnosis != null)
' "$OUT_ROOT/COMPLETE.json" >/dev/null; then
  echo "M DIRECTED REPLACE SCREEN FAILED COMPLETENESS CHECK" >&2
  exit 1
fi

touch "$OUT_ROOT/PIPELINE_COMPLETE"

echo "===== M DIRECTED REPLACE SCREEN COMPLETE ====="
cat "$OUT_ROOT/QUICK_RESULT.json"
