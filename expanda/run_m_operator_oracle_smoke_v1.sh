#!/usr/bin/env bash
set -euo pipefail

INSTANCES=${INSTANCES:-/home/lunet/cokt2/uncertainty-transformer-ppo/expanda/transformer_ppo_runs/instances}
NORMALIZATION=${NORMALIZATION:-/home/lunet/cokt2/uncertainty-transformer-ppo/expanda/transformer_ppo_runs/reward_normalization.json}
CHECKPOINT=${CHECKPOINT:-transformer_ppo_runs/diagnostics/v5r_joint_replace_smoke_b160_p20_seed9501/M/final_policy.pt}
OUT_ROOT=${OUT_ROOT:-transformer_ppo_runs/diagnostics/ccp_m_operator_oracle_v1_smoke_3inst_2state_5op_4seed_b400}

for required in "$NORMALIZATION" "$CHECKPOINT"; do
  if ! test -f "$required"; then
    echo "MISSING REQUIRED FILE: $required" >&2
    exit 1
  fi
done

mkdir -p "$OUT_ROOT"

echo "===== TEST OPERATOR-ONLY CONTROL ====="

python3 -u -m unittest \
  tests.test_m_operator_audit \
  tests.test_learning_components_v2_controls.V2ControlTests.test_forced_m_operator_keeps_base_batch_and_internal_target \
  tests.test_learning_components_v2_controls.V2ControlTests.test_m_continuation_seed_restarts_python_and_numpy_after_target \
  tests.test_learning_components_v2_controls.V2ControlTests.test_m_continuation_event_and_seed_are_atomic \
  > "$OUT_ROOT/tests.log" 2>&1

echo "===== 3-INSTANCE OPERATOR ORACLE SMOKE ====="

python3 -u audit_m_operator_interventions.py \
  --instances "$INSTANCES" \
  --normalization "$NORMALIZATION" \
  --checkpoint "$CHECKPOINT" \
  --out "$OUT_ROOT" \
  --instance-ids train-S0-005 train-S0-006 train-S0-012 \
  --algorithm-seed 777001 \
  --continuation-seeds 34001 34002 34003 34004 \
  --states-per-instance 2 \
  --minimum-generation 10 \
  --max-workers 8 \
  --evaluation-budget 400 \
  --evaluation-mode ccp \
  --uncertainty-profile transport_border_saa_v1 \
  --scenario-count 200 \
  --ccp-confidence 0.90 \
  --pop 20 \
  --gens 1000 \
  --device cpu \
  | tee "$OUT_ROOT/audit.log"

if ! jq -e '
  .completed == true
  and .instance_count == 3
  and .state_count == 6
  and .operator_action_count == 30
  and .continuation_seed_count == 4
  and .branch_record_count == 120
  and .total_run_count == 144
  and .scenario_exact == true
  and .pre_intervention_state_replay_exact == true
  and .forced_operator_exact == true
  and .shadow_base_operator_exact == true
' "$OUT_ROOT/AUDIT_COMPLETE.json" >/dev/null; then
  echo "OPERATOR AUDIT FAILED COMPLETENESS/EXACTNESS CHECK" >&2
  exit 1
fi

touch "$OUT_ROOT/PIPELINE_COMPLETE"

echo "===== M OPERATOR ORACLE SMOKE COMPLETE ====="
cat "$OUT_ROOT/QUICK_RESULT.json"
