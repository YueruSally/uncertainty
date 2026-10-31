#!/usr/bin/env bash
set -euo pipefail

INSTANCES=${INSTANCES:-/home/lunet/cokt2/uncertainty-transformer-ppo/expanda/transformer_ppo_runs/instances}
NORMALIZATION=${NORMALIZATION:-/home/lunet/cokt2/uncertainty-transformer-ppo/expanda/transformer_ppo_runs/reward_normalization.json}
CHECKPOINT=${CHECKPOINT:-transformer_ppo_runs/diagnostics/v5r_joint_replace_smoke_b160_p20_seed9501/M/final_policy.pt}
SOURCE_ROOT=${SOURCE_ROOT:-transformer_ppo_runs/diagnostics/ccp_m_late_replace_frequency_v1_3inst_3seed_b400}
PROBE_ROOT=${PROBE_ROOT:-transformer_ppo_runs/diagnostics/ccp_m_late_replace_positive_probe_v1_3inst_3seed_b400}
REVERSIBLE_ROOT=${REVERSIBLE_ROOT:-transformer_ppo_runs/diagnostics/ccp_m_late_replace_reversible_probe_v1_3inst_3seed_b400}
OUT_ROOT=${OUT_ROOT:-transformer_ppo_runs/diagnostics/ccp_m_late_replace_reversible_retry_v1_3inst_3seed_b400}

for required in \
  "$NORMALIZATION" \
  "$CHECKPOINT" \
  "$SOURCE_ROOT/COMPLETE.json" \
  "$PROBE_ROOT/COMPLETE.json" \
  "$REVERSIBLE_ROOT/COMPLETE.json"
do
  if ! test -f "$required"; then
    echo "MISSING REQUIRED FILE: $required" >&2
    exit 1
  fi
done

mkdir -p "$OUT_ROOT"

echo "===== TEST REVERSIBLE M PROBE RETRY ====="

python3 -u -m unittest \
  tests.test_learning_components_v2_controls.V2ControlTests.test_reversible_m_operator_probe_falls_back_to_base \
  tests.test_learning_components_v2_controls.V2ControlTests.test_reversible_m_operator_probe_retry_schedule \
  > "$OUT_ROOT/tests.log" 2>&1

echo "===== MATCHED CCP REVERSIBLE-PROBE RETRY SCREEN ====="

python3 -u evaluate_m_late_replace_frequency.py \
  --instances "$INSTANCES" \
  --normalization "$NORMALIZATION" \
  --checkpoint "$CHECKPOINT" \
  --reuse-root "$SOURCE_ROOT" \
  --positive-probe-root "$PROBE_ROOT" \
  --reversible-probe-root "$REVERSIBLE_ROOT" \
  --include-reversible-probe \
  --include-reversible-retry-probes \
  --out "$OUT_ROOT" \
  --instance-ids train-S0-005 train-S0-006 train-S0-012 \
  --algorithm-seeds 777001 777002 777003 \
  --minimum-generation 15 \
  --max-workers 9 \
  --evaluation-budget 400 \
  --scenario-count 200 \
  --ccp-confidence 0.90 \
  --pop 20 \
  --gens 1000 \
  --device cpu \
  | tee "$OUT_ROOT/evaluation.log"

if ! jq -e '
  .completed == true
  and .instance_count == 3
  and .algorithm_seed_count == 3
  and .run_count == 63
  and .status_counts.existing == 45
  and .status_counts.complete == 18
  and .scenario_exact == true
' "$OUT_ROOT/COMPLETE.json" >/dev/null; then
  echo "REVERSIBLE-PROBE RETRY SCREEN FAILED COMPLETENESS CHECK" >&2
  exit 1
fi

touch "$OUT_ROOT/PIPELINE_COMPLETE"

echo "===== M LATE REPLACE REVERSIBLE-PROBE RETRY COMPLETE ====="
cat "$OUT_ROOT/QUICK_RESULT.json"
