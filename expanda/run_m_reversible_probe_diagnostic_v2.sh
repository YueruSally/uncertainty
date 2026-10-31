#!/usr/bin/env bash
set -euo pipefail

INSTANCES=${INSTANCES:-/home/lunet/cokt2/uncertainty-transformer-ppo/expanda/transformer_ppo_runs/instances}
NORMALIZATION=${NORMALIZATION:-/home/lunet/cokt2/uncertainty-transformer-ppo/expanda/transformer_ppo_runs/reward_normalization.json}
CHECKPOINT=${CHECKPOINT:-transformer_ppo_runs/diagnostics/v5r_joint_replace_smoke_b160_p20_seed9501/M/final_policy.pt}
SOURCE_ROOT=${SOURCE_ROOT:-transformer_ppo_runs/diagnostics/ccp_m_late_replace_frequency_v1_3inst_3seed_b400}
REVERSIBLE_ROOT=${REVERSIBLE_ROOT:-transformer_ppo_runs/diagnostics/ccp_m_late_replace_reversible_probe_v1_3inst_3seed_b400}
OUT_ROOT=${OUT_ROOT:-transformer_ppo_runs/diagnostics/ccp_m_reversible_probe_diagnostic_v2_10inst_3seed_b400}

for required in \
  "$NORMALIZATION" \
  "$CHECKPOINT" \
  "$SOURCE_ROOT/COMPLETE.json" \
  "$REVERSIBLE_ROOT/COMPLETE.json"
do
  if ! test -f "$required"; then
    echo "MISSING REQUIRED FILE: $required" >&2
    exit 1
  fi
done

mkdir -p "$OUT_ROOT"

echo "===== TEST M DIAGNOSTIC PIPELINE ====="

python3 -u -m unittest \
  tests.test_m_late_replace_frequency \
  tests.test_learning_components_v2_controls.V2ControlTests.test_forced_m_operator_generation_schedule \
  tests.test_learning_components_v2_controls.V2ControlTests.test_reversible_m_operator_probe_falls_back_to_base \
  > "$OUT_ROOT/tests.log" 2>&1

echo "===== 10-INSTANCE MATCHED CCP M DIAGNOSTIC ====="

python3 -u evaluate_m_late_replace_frequency.py \
  --instances "$INSTANCES" \
  --normalization "$NORMALIZATION" \
  --checkpoint "$CHECKPOINT" \
  --reuse-root "$SOURCE_ROOT" \
  --reversible-probe-root "$REVERSIBLE_ROOT" \
  --strategies \
    BASE \
    REPLACE_LATE_P100 \
    REPLACE_LATE_REVERSIBLE_PROBE \
  --out "$OUT_ROOT" \
  --instance-ids \
    train-S0-002 train-S0-005 train-S0-006 train-S0-007 train-S0-008 \
    train-S0-009 train-S0-010 train-S0-011 train-S0-012 train-S0-013 \
  --algorithm-seeds 777001 777002 777003 \
  --minimum-generation 15 \
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
  and .run_count == 90
  and ([.status_counts[]] | add) == 90
  and .scenario_exact == true
  and (.strategies | length) == 3
  and (.per_instance | length) == 10
  and (.diagnosis != null)
' "$OUT_ROOT/COMPLETE.json" >/dev/null; then
  echo "M DIAGNOSTIC FAILED COMPLETENESS CHECK" >&2
  exit 1
fi

touch "$OUT_ROOT/PIPELINE_COMPLETE"

echo "===== M REVERSIBLE-PROBE DIAGNOSTIC COMPLETE ====="
cat "$OUT_ROOT/QUICK_RESULT.json"
