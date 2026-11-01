#!/usr/bin/env bash
set -euo pipefail

DEVELOPMENT_ROOT=${DEVELOPMENT_ROOT:-transformer_ppo_runs/diagnostics/ccp_i_balanced_strength_grid_v1_10inst_3seed_b400}
HOLDOUT_ROOT=${HOLDOUT_ROOT:-transformer_ppo_runs/diagnostics/ccp_i_directed_balanced_validation_v1_10inst_holdout7seed_b400}
OUT_ROOT=${OUT_ROOT:-transformer_ppo_runs/diagnostics/ccp_i_initial_quality_gate_v1_10inst_10seed_offline}

for REQUIRED in \
  "$DEVELOPMENT_ROOT/PIPELINE_COMPLETE" \
  "$DEVELOPMENT_ROOT/COMPLETE.json" \
  "$HOLDOUT_ROOT/PIPELINE_COMPLETE" \
  "$HOLDOUT_ROOT/COMPLETE.json"
do
  if ! test -e "$REQUIRED"; then
    echo "MISSING REQUIRED INPUT: $REQUIRED" >&2
    exit 1
  fi
done

mkdir -p "$OUT_ROOT"

echo "===== OFFLINE I INITIAL QUALITY GATE ====="
python3 -u analyze_i_initial_quality_gate.py \
  --development-complete "$DEVELOPMENT_ROOT/COMPLETE.json" \
  --holdout-complete "$HOLDOUT_ROOT/COMPLETE.json" \
  --out "$OUT_ROOT" \
  | tee "$OUT_ROOT/analysis.log"

if ! jq -e '
  .completed == true
  and .pair_count == 100
  and .splits.development.pairs == 30
  and .splits.holdout.pairs == 70
' "$OUT_ROOT/COMPLETE.json" >/dev/null; then
  echo "I INITIAL QUALITY GATE ANALYSIS FAILED" >&2
  exit 1
fi

touch "$OUT_ROOT/ANALYSIS_COMPLETE"

echo "===== I INITIAL QUALITY GATE COMPLETE ====="
cat "$OUT_ROOT/QUICK_RESULT.json"
