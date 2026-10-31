#!/usr/bin/env bash
set -euo pipefail

AUDIT_ROOT=${AUDIT_ROOT:-transformer_ppo_runs/diagnostics/ccp_m_downstream_unique_v1_10inst_3seed_b400}
OUT_ROOT=${OUT_ROOT:-transformer_ppo_runs/diagnostics/ccp_m_candidate_safe_selector_v1}
DATASET="$AUDIT_ROOT/offline_analysis/m_candidate_ranker.jsonl"

if ! test -f "$DATASET"; then
  echo "MISSING M CANDIDATE DATASET: $DATASET" >&2
  exit 1
fi

mkdir -p "$OUT_ROOT"

python3 -u probe_m_candidate_safe_selector.py \
  --dataset "$DATASET" \
  --out "$OUT_ROOT/FINAL_REPORT.json" \
  --seed 10021 \
  > "$OUT_ROOT/analysis.log"

touch "$OUT_ROOT/ANALYSIS_COMPLETE"

jq '{
  completed,
  definition,
  candidate_count,
  state_count,
  instance_count,
  feature_dimension,
  positive_candidate_count,
  models:
    [.models[] | {
      model,
      candidate_loo_roc_auc,
      candidate_loo_average_precision,
      ungated_selector,
      deployments:
        [.deployments[] | del(.selected_state_keys)]
    }],
  admission_thresholds,
  admission_pass,
  admitted_models
}' "$OUT_ROOT/FINAL_REPORT.json"
