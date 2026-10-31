#!/usr/bin/env bash
set -euo pipefail

INSTANCES=${INSTANCES:-/home/lunet/cokt2/uncertainty-transformer-ppo/expanda/transformer_ppo_runs/instances}
NORMALIZATION=${NORMALIZATION:-/home/lunet/cokt2/uncertainty-transformer-ppo/expanda/transformer_ppo_runs/reward_normalization.json}
CHECKPOINT=${CHECKPOINT:-transformer_ppo_runs/diagnostics/v5r_joint_replace_smoke_b160_p20_seed9501/M/final_policy.pt}
SOURCE_AUDIT_ROOT=${SOURCE_AUDIT_ROOT:-transformer_ppo_runs/diagnostics/ccp_m_downstream_unique_v1_10inst_3seed_b400}
SMOKE_ROOT=${SMOKE_ROOT:-transformer_ppo_runs/diagnostics/ccp_m_continuation_stability_v1_smoke}
OUT_ROOT=${OUT_ROOT:-transformer_ppo_runs/diagnostics/ccp_m_continuation_stability_v1_12state_4seed_3action_b400}

for required in \
  "$NORMALIZATION" \
  "$CHECKPOINT" \
  "$SOURCE_AUDIT_ROOT/AUDIT_COMPLETE.json"
do
  if ! test -f "$required"; then
    echo "MISSING REQUIRED FILE: $required" >&2
    exit 1
  fi
done

mkdir -p "$SMOKE_ROOT" "$OUT_ROOT"

echo "===== TESTS ====="

python3 -u -m unittest \
  tests.test_m_continuation_stability \
  tests.test_learning_components_v2_controls.V2ControlTests.test_m_continuation_seed_restarts_python_and_numpy_after_target \
  tests.test_learning_components_v2_controls.V2ControlTests.test_m_continuation_event_and_seed_are_atomic \
  > "$OUT_ROOT/tests.log" 2>&1

echo "===== SMOKE: 1 STATE x 2 ACTIONS x 2 CONTINUATIONS ====="

python3 -u collect_m_continuation_stability.py \
  --audit-root "$SOURCE_AUDIT_ROOT" \
  --instances "$INSTANCES" \
  --normalization "$NORMALIZATION" \
  --checkpoint "$CHECKPOINT" \
  --out "$SMOKE_ROOT" \
  --continuation-seeds 31991 31992 \
  --state-count 1 \
  --actions-per-state 2 \
  --minimum-oracle-percent 10 \
  --max-states-per-instance 1 \
  --max-workers 4 \
  --evaluation-budget 400 \
  --evaluation-mode ccp \
  --uncertainty-profile transport_border_saa_v1 \
  --scenario-count 200 \
  --ccp-confidence 0.90 \
  --pop 20 \
  --gens 1000 \
  --device cpu \
  > "$SMOKE_ROOT/collection.log" 2>&1

echo "===== FORMAL: 12 STATES x 3 ACTIONS x 4 CONTINUATIONS ====="

python3 -u collect_m_continuation_stability.py \
  --audit-root "$SOURCE_AUDIT_ROOT" \
  --instances "$INSTANCES" \
  --normalization "$NORMALIZATION" \
  --checkpoint "$CHECKPOINT" \
  --out "$OUT_ROOT" \
  --continuation-seeds 31001 31002 31003 31004 \
  --state-count 12 \
  --actions-per-state 3 \
  --minimum-oracle-percent 5 \
  --max-states-per-instance 4 \
  --max-workers 8 \
  --evaluation-budget 400 \
  --evaluation-mode ccp \
  --uncertainty-profile transport_border_saa_v1 \
  --scenario-count 200 \
  --ccp-confidence 0.90 \
  --pop 20 \
  --gens 1000 \
  --device cpu \
  > "$OUT_ROOT/collection.log" 2>&1

jq '{
  completed,
  selected_state_count,
  selected_action_count,
  continuation_seed_count,
  branch_record_count,
  top_action_stable_positive_states,
  top_action_stable_positive_state_fraction,
  mean_top_action_positive_continuation_fraction,
  mean_all_action_majority_sign_agreement,
  best_short_horizon,
  best_short_horizon_metrics,
  scenario_exact,
  forced_execution_exact,
  pre_intervention_state_replay_exact,
  admission_pass
}' "$OUT_ROOT/STABILITY_COMPLETE.json" \
  > "$OUT_ROOT/QUICK_RESULT.json"

touch "$OUT_ROOT/PIPELINE_COMPLETE"

echo "===== M CONTINUATION STABILITY V1 COMPLETE ====="
cat "$OUT_ROOT/QUICK_RESULT.json"
