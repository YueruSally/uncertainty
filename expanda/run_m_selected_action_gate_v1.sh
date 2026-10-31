#!/usr/bin/env bash
set -euo pipefail

AUDIT_ROOT=${AUDIT_ROOT:-transformer_ppo_runs/diagnostics/ccp_m_downstream_unique_v1_10inst_3seed_b400}
TRANSFORMER_ROOT=${TRANSFORMER_ROOT:-transformer_ppo_runs/diagnostics/ccp_m_downstream_transformer_v2_regression}
OUT_ROOT=${OUT_ROOT:-transformer_ppo_runs/diagnostics/ccp_m_selected_action_gate_v1}

STATE_DATASET="$AUDIT_ROOT/offline_analysis/m_state_gate.jsonl"
CANDIDATE_DATASET="$AUDIT_ROOT/offline_analysis/m_candidate_ranker.jsonl"

for REQUIRED in \
  "$AUDIT_ROOT/AUDIT_COMPLETE.json" \
  "$STATE_DATASET" \
  "$CANDIDATE_DATASET" \
  "$TRANSFORMER_ROOT/FINAL_REPORT.json"
do
  if ! test -f "$REQUIRED"; then
    echo "MISSING REQUIRED INPUT: $REQUIRED" >&2
    exit 1
  fi
done

mkdir -p "$OUT_ROOT"

python3 -u probe_m_selected_action_gate.py \
  --state-dataset "$STATE_DATASET" \
  --candidate-dataset "$CANDIDATE_DATASET" \
  --transformer-root "$TRANSFORMER_ROOT" \
  --out "$OUT_ROOT/FINAL_REPORT.json" \
  --seed 10001 \
  > "$OUT_ROOT/analysis.log"

touch "$OUT_ROOT/ANALYSIS_COMPLETE"

jq '{
  completed,
  definition,
  state_count,
  positive_selected_actions,
  positive_selected_action_fraction,
  models:
    [.models[] | {
      model,
      loo_roc_auc,
      loo_average_precision,
      threshold_0_5:
        (.threshold_0_5 | del(
          .selected_state_keys,
          .selected_probabilities
        )),
      top_fraction:
        [.top_fraction[] | del(
          .selected_state_keys,
          .selected_probabilities
        )]
    }],
  admission_thresholds,
  admission_pass,
  admitted_models
}' "$OUT_ROOT/FINAL_REPORT.json"
