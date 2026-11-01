#!/usr/bin/env bash
set -euo pipefail

SOURCE_ROOT=${SOURCE_ROOT:-transformer_ppo_runs/diagnostics/ccp_m_late_replace_frequency_v1_3inst_3seed_b400}
DIAGNOSTIC_ROOT=${DIAGNOSTIC_ROOT:-transformer_ppo_runs/diagnostics/ccp_m_reversible_probe_diagnostic_v2_10inst_3seed_b400}
GRID_ROOT=${GRID_ROOT:-transformer_ppo_runs/diagnostics/ccp_m_replace_schedule_grid_v1_10inst_3seed_b400}
VALIDATION_ROOT=${VALIDATION_ROOT:-transformer_ppo_runs/diagnostics/ccp_m_replace_schedule_validation_v1_10inst_10seed_b400}
OUT_ROOT=${OUT_ROOT:-transformer_ppo_runs/diagnostics/ccp_m_schedule_three_way_selector_v1_10inst_10seed_b400}

for REQUIRED in \
  "$SOURCE_ROOT/COMPLETE.json" \
  "$DIAGNOSTIC_ROOT/COMPLETE.json" \
  "$GRID_ROOT/COMPLETE.json" \
  "$VALIDATION_ROOT/COMPLETE.json"
do
  if ! test -f "$REQUIRED"; then
    echo "MISSING REQUIRED INPUT: $REQUIRED" >&2
    exit 1
  fi
done

mkdir -p "$OUT_ROOT"

echo "===== TEST M THREE-WAY SELECTOR INPUTS ====="
python3 -u -m unittest tests.test_m_schedule_state_gate \
  > "$OUT_ROOT/tests.log" 2>&1

echo "===== ANALYZE BASE/G10/G12 SELECTOR ====="
python3 -u analyze_m_schedule_three_way_selector.py \
  --roots \
    "$VALIDATION_ROOT" "$GRID_ROOT" "$DIAGNOSTIC_ROOT" "$SOURCE_ROOT" \
  --out "$OUT_ROOT" \
  --instance-ids \
    train-S0-002 train-S0-005 train-S0-006 train-S0-007 train-S0-008 \
    train-S0-009 train-S0-010 train-S0-011 train-S0-012 train-S0-013 \
  --algorithm-seeds \
    777001 777002 777003 777004 777005 \
    777006 777007 777008 777009 777010 \
  --training-seeds 777001 777002 777003 \
  --evaluation-budget 400 \
  --seed 13001 \
  > "$OUT_ROOT/analysis.log"

if ! jq -e '
  .completed == true
  and .scenario_exact == true
  and .pre_intervention_replay_exact == true
  and .run_count == 100
  and (.seed_holdout_results | length) == 2
  and (.loo_results | length) == 4
' "$OUT_ROOT/QUICK_RESULT.json" >/dev/null; then
  echo "M THREE-WAY SELECTOR FAILED COMPLETENESS CHECK" >&2
  exit 1
fi

touch "$OUT_ROOT/ANALYSIS_COMPLETE"

echo "===== M THREE-WAY SELECTOR COMPLETE ====="
cat "$OUT_ROOT/QUICK_RESULT.json"
