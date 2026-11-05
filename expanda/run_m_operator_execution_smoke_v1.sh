#!/usr/bin/env bash
# New six-state consistency sample; not a replay/certification of old labels.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export OUT_ROOT=${SMOKE_ROOT:-transformer_ppo_runs/diagnostics/ccp_m_shared_execution_smoke_v1_6states}
export AUDIT_STATES_PER_RUN=3
echo "===== M-EXECUTION CHECK: CCP100 / POP20 / B400 / SIX NEW STATES / NO TRAINING ====="
bash run_m_ccp100_operator_oracle_audit_v7.sh
python3 -u analyze_m_operator_repeat_check.py \
  --runs "$OUT_ROOT/runs/train-S0-009_seed_890001" "$OUT_ROOT/runs/train-S0-011_seed_890002" \
  --expected-state-count 6 \
  --out "$OUT_ROOT/repeat_check"
jq '{state_count, new_training_steps, same_mutated_child_repair_comparison, training_data_approved}' \
  "$OUT_ROOT/repeat_check/REPEAT_CHECK.json"
echo "===== M-EXECUTION CHECK COMPLETE: $OUT_ROOT/repeat_check/REPEAT_CHECK.json ====="
