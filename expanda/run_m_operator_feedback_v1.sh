#!/usr/bin/env bash
# Two predeclared collection batches; no optimisation of sampling by outcome.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

INSTANCES=${INSTANCES:-/home/lunet/cokt2/uncertainty-transformer-ppo/expanda/transformer_ppo_runs/instances}
NORMALIZATION=${NORMALIZATION:-/home/lunet/cokt2/uncertainty-transformer-ppo/expanda/transformer_ppo_runs/reward_normalization.json}
V6_PPO_ROOT=${V6_PPO_ROOT:-transformer_ppo_runs/diagnostics/ccp_m_ccp100_operator_scheduler_v6_ppo_5inst_b400}
COLLECTION_ROOT=${COLLECTION_ROOT:-transformer_ppo_runs/diagnostics/ccp_m_operator_feedback_v1_6inst_2seed_b400}
COLLECTION_BATCH=${COLLECTION_BATCH:-1}
MAX_WORKERS=${MAX_WORKERS:-2}
MAX_ATTEMPTS=${MAX_ATTEMPTS:-2}
PYTHON=${PYTHON:-python3}

mkdir -p "$COLLECTION_ROOT"
echo "===== TEST SHARED-EXECUTION OPERATOR DATA COLLECTION ====="
if ! "$PYTHON" -u -m unittest \
  tests.test_m_operator_scheduler_v6 \
  tests.test_m_generation_reward_v3 \
  tests.test_m_operator_oracle_analysis \
  tests.test_m_operator_execution_consistency \
  tests.test_m_operator_repeat_check \
  tests.test_m_operator_feedback_collection \
  > "$COLLECTION_ROOT/collection_tests.log" 2>&1; then
  cat "$COLLECTION_ROOT/collection_tests.log"
  exit 1
fi
cat "$COLLECTION_ROOT/collection_tests.log"
echo "===== COLLECT BATCH $COLLECTION_BATCH: CCP100 / POP20 / B400 / NO TRAINING ====="
"$PYTHON" -u collect_m_operator_feedback_v1.py \
  --instances "$INSTANCES" \
  --normalization "$NORMALIZATION" \
  --checkpoint "$V6_PPO_ROOT/final_policy.pt" \
  --out "$COLLECTION_ROOT" \
  --batch "$COLLECTION_BATCH" \
  --workers "$MAX_WORKERS" \
  --max-attempts "$MAX_ATTEMPTS"
