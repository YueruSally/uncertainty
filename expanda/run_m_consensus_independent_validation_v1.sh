#!/usr/bin/env bash
set -euo pipefail

INSTANCES=${INSTANCES:-/home/lunet/cokt2/uncertainty-transformer-ppo/expanda/transformer_ppo_runs/instances}
NORMALIZATION=${NORMALIZATION:-/home/lunet/cokt2/uncertainty-transformer-ppo/expanda/transformer_ppo_runs/reward_normalization.json}
CHECKPOINT=${CHECKPOINT:-transformer_ppo_runs/diagnostics/v5r_joint_replace_smoke_b160_p20_seed9501/M/final_policy.pt}
AUDIT_ROOT=${AUDIT_ROOT:-transformer_ppo_runs/diagnostics/ccp_m_downstream_unique_v1_10inst_3seed_b400}
SOURCE_ROOT=${SOURCE_ROOT:-transformer_ppo_runs/diagnostics/ccp_m_continuation_mean_v1_60state_4action_4seed_b400}
CONSENSUS_ROOT=${CONSENSUS_ROOT:-transformer_ppo_runs/diagnostics/ccp_m_immediate_consensus_v1}
OUT_ROOT=${OUT_ROOT:-transformer_ppo_runs/diagnostics/ccp_m_consensus_independent_v1_5action_8seed_b400}

for required in \
  "$NORMALIZATION" \
  "$CHECKPOINT" \
  "$AUDIT_ROOT/AUDIT_COMPLETE.json" \
  "$SOURCE_ROOT/STABILITY_COMPLETE.json" \
  "$SOURCE_ROOT/selection_manifest.json" \
  "$CONSENSUS_ROOT/CONSENSUS_COMPLETE.json" \
  "$CONSENSUS_ROOT/ANALYSIS_COMPLETE"
do
  if ! test -f "$required"; then
    echo "MISSING REQUIRED FILE: $required" >&2
    exit 1
  fi
done

mkdir -p "$OUT_ROOT"

echo "===== TEST FROZEN CONSENSUS MANIFEST ====="

python3 -u -m unittest \
  tests.test_m_consensus_validation \
  tests.test_m_continuation_stability \
  tests.test_learning_components_v2_controls.V2ControlTests.test_m_continuation_seed_restarts_python_and_numpy_after_target \
  tests.test_learning_components_v2_controls.V2ControlTests.test_m_continuation_event_and_seed_are_atomic \
  > "$OUT_ROOT/tests.log" 2>&1

echo "===== VALIDATE 5 FROZEN M ACTIONS x 8 NEW CONTINUATIONS ====="

python3 -u validate_m_consensus_actions.py \
  --consensus-report "$CONSENSUS_ROOT/CONSENSUS_COMPLETE.json" \
  --source-stability-root "$SOURCE_ROOT" \
  --audit-root "$AUDIT_ROOT" \
  --instances "$INSTANCES" \
  --normalization "$NORMALIZATION" \
  --checkpoint "$CHECKPOINT" \
  --out "$OUT_ROOT" \
  --continuation-seeds \
    33001 33002 33003 33004 33005 33006 33007 33008 \
  --max-workers 8 \
  --evaluation-budget 400 \
  --evaluation-mode ccp \
  --uncertainty-profile transport_border_saa_v1 \
  --scenario-count 200 \
  --ccp-confidence 0.90 \
  --pop 20 \
  --gens 1000 \
  --device cpu \
  | tee "$OUT_ROOT/validation.log"

if ! jq -e '
  .completed == true
  and .selected_state_count == 5
  and .selected_action_count == 5
  and .independent_continuation_seed_count == 8
  and .paired_action_continuation_count == 40
  and .scenario_exact == true
  and .forced_execution_exact == true
  and .pre_intervention_state_replay_exact == true
' "$OUT_ROOT/VALIDATION_COMPLETE.json" >/dev/null; then
  echo "INDEPENDENT VALIDATION FAILED COMPLETENESS/EXACTNESS CHECK" >&2
  exit 1
fi

touch "$OUT_ROOT/PIPELINE_COMPLETE"

echo "===== M CONSENSUS INDEPENDENT VALIDATION COMPLETE ====="
cat "$OUT_ROOT/QUICK_RESULT.json"
