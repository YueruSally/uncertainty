#!/usr/bin/env bash
set -euo pipefail

INSTANCES=${INSTANCES:-/home/lunet/cokt2/uncertainty-transformer-ppo/expanda/transformer_ppo_runs/instances}
NORMALIZATION=${NORMALIZATION:-/home/lunet/cokt2/uncertainty-transformer-ppo/expanda/transformer_ppo_runs/reward_normalization.json}
CHECKPOINT=${CHECKPOINT:-transformer_ppo_runs/diagnostics/v5r_joint_replace_smoke_b160_p20_seed9501/M/final_policy.pt}
OUT_ROOT=${OUT_ROOT:-transformer_ppo_runs/diagnostics/ccp_m_late_replace_frequency_v1_3inst_3seed_b400}

for required in "$NORMALIZATION" "$CHECKPOINT"; do
  if ! test -f "$required"; then
    echo "MISSING REQUIRED FILE: $required" >&2
    exit 1
  fi
done

mkdir -p "$OUT_ROOT"

echo "===== TEST LATE REPLACE SCHEDULE ====="

python3 -u -m unittest \
  tests.test_learning_components_v2_controls.V2ControlTests.test_forced_m_operator_generation_schedule \
  tests.test_learning_components_v2_controls.V2ControlTests.test_forced_m_operator_keeps_base_batch_and_internal_target \
  > "$OUT_ROOT/tests.log" 2>&1

echo "===== MATCHED CCP LATE REPLACE FREQUENCY SCREEN ====="

python3 -u evaluate_m_late_replace_frequency.py \
  --instances "$INSTANCES" \
  --normalization "$NORMALIZATION" \
  --checkpoint "$CHECKPOINT" \
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
  and .run_count == 27
  and .scenario_exact == true
' "$OUT_ROOT/COMPLETE.json" >/dev/null; then
  echo "LATE REPLACE FREQUENCY SCREEN FAILED COMPLETENESS CHECK" >&2
  exit 1
fi

touch "$OUT_ROOT/PIPELINE_COMPLETE"

echo "===== M LATE REPLACE FREQUENCY SCREEN COMPLETE ====="
cat "$OUT_ROOT/QUICK_RESULT.json"
