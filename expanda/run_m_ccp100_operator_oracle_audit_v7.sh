#!/usr/bin/env bash
set -euo pipefail

INSTANCES=${INSTANCES:-/home/lunet/cokt2/uncertainty-transformer-ppo/expanda/transformer_ppo_runs/instances}
NORMALIZATION=${NORMALIZATION:-/home/lunet/cokt2/uncertainty-transformer-ppo/expanda/transformer_ppo_runs/reward_normalization.json}
V6_PPO_ROOT=${V6_PPO_ROOT:-transformer_ppo_runs/diagnostics/ccp_m_ccp100_operator_scheduler_v6_ppo_5inst_b400}
OUT_ROOT=${OUT_ROOT:-transformer_ppo_runs/diagnostics/ccp_m_ccp100_operator_oracle_audit_v7_64state_k3}
MAX_ATTEMPTS=${MAX_ATTEMPTS:-3}

for REQUIRED in \
  "$INSTANCES/catalog.json" \
  "$NORMALIZATION" \
  "$V6_PPO_ROOT/final_policy.pt"; do
  if ! test -f "$REQUIRED"; then
    echo "MISSING REQUIRED INPUT: $REQUIRED" >&2
    exit 1
  fi
done
if ! [[ "$MAX_ATTEMPTS" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_ATTEMPTS must be positive: $MAX_ATTEMPTS" >&2
  exit 1
fi

mkdir -p "$OUT_ROOT/runs" "$OUT_ROOT/logs"
exec 9>"$OUT_ROOT/pipeline.lock"
if ! flock -n 9; then
  echo "ANOTHER M-V7 ORACLE AUDIT IS ALREADY RUNNING" >&2
  exit 75
fi

echo "===== TEST M-V7 REPEATED FIVE-OPERATOR ORACLE AUDIT ====="
python3 -u -m unittest \
  tests.test_m_operator_scheduler_v6 \
  tests.test_m_generation_reward_v3 \
  tests.test_m_operator_oracle_analysis \
  > "$OUT_ROOT/tests.log" 2>&1

run_one() {
  local instance_id=$1
  local seed=$2
  local run=$3
  local log=$4
  if test -f "$run/COMPLETE.json"; then
    echo "[M-V7-AUDIT] existing $instance_id seed=$seed"
    return 0
  fi
  local attempt=1
  while (( attempt <= MAX_ATTEMPTS )); do
    if test -d "$run"; then
      local stamp
      stamp=$(date -u +%Y%m%dT%H%M%SZ)
      mv "$run" "${run}_interrupted_${stamp}_${attempt}"
    fi
    echo "[M-V7-AUDIT] $instance_id seed=$seed attempt=$attempt/$MAX_ATTEMPTS"
    set +e
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 -u \
      run_learning_components_deterministic.py \
      --instance "$INSTANCES/${instance_id}.json" \
      --normalization "$NORMALIZATION" \
      --out "$run" \
      --component M \
      --checkpoint "$V6_PPO_ROOT/final_policy.pt" \
      --inference-action-mode argmax \
      --inference-policy-seed 68101 \
      --model-choice-probability 1 \
      --algorithm-seed "$seed" \
      --evaluation-mode ccp \
      --uncertainty-profile transport_border_saa_v1 \
      --scenario-count 100 \
      --ccp-confidence 0.90 \
      --evaluation-budget 400 \
      --pop 20 \
      --gens 1000 \
      --candidate-audit-states 32 \
      --candidate-audit-candidates 5 \
      --candidate-audit-replicates 3 \
      --device cpu \
      > "$log" 2>&1
    local status=$?
    set -e
    if (( status == 0 )) && test -f "$run/COMPLETE.json"; then
      echo "[M-V7-AUDIT] complete $instance_id seed=$seed"
      return 0
    fi
    if (( attempt == MAX_ATTEMPTS )); then
      echo "M-V7 AUDIT FAILED: $instance_id seed=$seed log=$log" >&2
      return "$status"
    fi
    attempt=$((attempt + 1))
  done
}

echo "===== STAGE 1: 64 EARLY/MIDDLE/LATE STATES / FIVE OPERATORS / K3 ====="
run_one train-S0-009 890001 \
  "$OUT_ROOT/runs/train-S0-009_seed_890001" \
  "$OUT_ROOT/logs/train-S0-009_seed_890001.log" &
PID_A=$!
run_one train-S0-011 890002 \
  "$OUT_ROOT/runs/train-S0-011_seed_890002" \
  "$OUT_ROOT/logs/train-S0-011_seed_890002.log" &
PID_B=$!
wait "$PID_A"
wait "$PID_B"

echo "===== STAGE 2: ONE-STEP ORACLE AND REPEAT-STABILITY ANALYSIS ====="
python3 -u analyze_m_operator_oracle_audit.py \
  --runs \
    "$OUT_ROOT/runs/train-S0-009_seed_890001" \
    "$OUT_ROOT/runs/train-S0-011_seed_890002" \
  --expected-state-count 64 \
  --expected-replicates 3 \
  --out "$OUT_ROOT" \
  2>&1 | tee "$OUT_ROOT/analysis.log"

if ! jq -e '
  .completed == true
  and .completeness_pass == true
  and .raw_state_count == 64
  and .state_count == 64
  and .excluded_state_count == 0
  and .operator_count == 5
  and .replicates_per_operator == 3
  and .counterfactual_evaluation_count == .expected_counterfactual_evaluation_count
' "$OUT_ROOT/OPERATOR_ORACLE_AUDIT.json" >/dev/null; then
  echo "M-V7 AUDIT COMPLETENESS CHECK FAILED" >&2
  exit 1
fi

touch "$OUT_ROOT/PIPELINE_COMPLETE"
echo "===== M-V7 OPERATOR ORACLE AUDIT COMPLETE ====="
