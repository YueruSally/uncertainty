#!/usr/bin/env bash
# Existing-data diagnostic only. No rollout collection or model training.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

AUDIT_ROOT=${AUDIT_ROOT:-transformer_ppo_runs/diagnostics/ccp_m_ccp100_operator_oracle_audit_v7_64state_k3}
REPORT_ROOT=${REPORT_ROOT:-$AUDIT_ROOT/repeat_check_v1}
PYTHON=${PYTHON:-python3}

for RUN_NAME in train-S0-009_seed_890001 train-S0-011_seed_890002; do
  for NAME in mutation_candidate_audit_states.jsonl mutation_candidate_audit_outcomes.jsonl; do
    if ! test -s "$AUDIT_ROOT/runs/$RUN_NAME/$NAME"; then
      echo "MISSING INPUT: $AUDIT_ROOT/runs/$RUN_NAME/$NAME" >&2
      exit 1
    fi
  done
done
mkdir -p "$REPORT_ROOT"
echo "===== M-CHECK: SHARED EXECUTION / FIXED CCP100 CONSISTENCY ====="
if ! "$PYTHON" -u -m unittest \
  tests.test_m_operator_scheduler_v6 \
  tests.test_m_generation_reward_v3 \
  tests.test_m_operator_oracle_analysis \
  tests.test_m_operator_execution_consistency \
  tests.test_m_operator_repeat_check > "$REPORT_ROOT/tests.log" 2>&1; then
  cat "$REPORT_ROOT/tests.log"
  exit 1
fi
cat "$REPORT_ROOT/tests.log"

echo "===== M-CHECK: HELD-OUT REPEATS / ZERO NEW OBJECTIVE EVALUATIONS ====="
"$PYTHON" -u analyze_m_operator_repeat_check.py \
  --runs \
    "$AUDIT_ROOT/runs/train-S0-009_seed_890001" \
    "$AUDIT_ROOT/runs/train-S0-011_seed_890002" \
  --expected-state-count 64 \
  --out "$REPORT_ROOT" \
  2>&1 | tee "$REPORT_ROOT/analysis.log"
echo "===== M-CHECK COMPLETE: $REPORT_ROOT/REPEAT_CHECK.json ====="
