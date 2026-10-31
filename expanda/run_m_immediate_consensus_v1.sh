#!/usr/bin/env bash
set -euo pipefail

SOURCE_ROOT=${SOURCE_ROOT:-transformer_ppo_runs/diagnostics/ccp_m_continuation_mean_v1_60state_4action_4seed_b400}
OUT_ROOT=${OUT_ROOT:-transformer_ppo_runs/diagnostics/ccp_m_immediate_consensus_v1}
SOURCE_DATASET="$SOURCE_ROOT/continuation_mean_candidates.jsonl"
LATE_DATASET="$OUT_ROOT/late_generation_10_plus.jsonl"
REPORT="$OUT_ROOT/CONSENSUS_COMPLETE.json"

for required in \
  "$SOURCE_ROOT/PIPELINE_COMPLETE" \
  "$SOURCE_DATASET"
do
  if ! test -f "$required"; then
    echo "MISSING REQUIRED FILE: $required" >&2
    exit 1
  fi
done

mkdir -p "$OUT_ROOT"
jq -c 'select(.generation >= 10)' "$SOURCE_DATASET" > "$LATE_DATASET"

if test "$(wc -l < "$LATE_DATASET")" -ne 120; then
  echo "EXPECTED 120 LATE-STAGE CANDIDATES" >&2
  exit 1
fi

python3 -u probe_m_immediate_consensus.py \
  --dataset "$LATE_DATASET" \
  --out "$REPORT" \
  --probability-threshold 0.5 \
  --seed 9997

jq '{
  completed,
  definition,
  probability_threshold,
  state_count,
  selected_state_count,
  selected_state_fraction,
  model_agreement_fraction,
  selected_immediate_precision,
  mean_selected_final_percent,
  median_selected_final_percent,
  minimum_selected_final_percent,
  maximum_selected_final_percent,
  selected_final_positive_fraction,
  selected_final_negative_fraction,
  selected_safe_positive_fraction,
  mean_deployed_final_percent,
  positive_deployed_folds,
  admission_thresholds,
  admission_pass
}' "$REPORT" > "$OUT_ROOT/QUICK_RESULT.json"

touch "$OUT_ROOT/ANALYSIS_COMPLETE"

echo "===== M IMMEDIATE CONSENSUS V1 COMPLETE ====="
cat "$OUT_ROOT/QUICK_RESULT.json"
