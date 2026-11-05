#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
INSTANCES=${INSTANCES:-/home/lunet/cokt2/uncertainty-transformer-ppo/expanda/transformer_ppo_runs/instances}
NORMALIZATION=${NORMALIZATION:-/home/lunet/cokt2/uncertainty-transformer-ppo/expanda/transformer_ppo_runs/reward_normalization.json}
V6_PPO_ROOT=${V6_PPO_ROOT:-transformer_ppo_runs/diagnostics/ccp_m_ccp100_operator_scheduler_v6_ppo_5inst_b400}
SOURCE_COLLECTION_ROOT=${SOURCE_COLLECTION_ROOT:-transformer_ppo_runs/diagnostics/ccp_m_operator_feedback_v1_6inst_2seed_b400}
COLLECTION_ROOT=${COLLECTION_ROOT:-transformer_ppo_runs/diagnostics/ccp_m_operator_feedback_v2_6inst_2seed_b400}
COLLECTION_BATCH=${COLLECTION_BATCH:-1}
PYTHON=${PYTHON:-python3}
mkdir -p "$COLLECTION_ROOT"
echo "===== TEST CAPACITY-AWARE SAMPLING AND EXACT REPLAY ====="
if ! "$PYTHON" -u -m unittest \
  tests.test_m_operator_scheduler_v6 tests.test_m_generation_reward_v3 \
  tests.test_m_operator_oracle_analysis tests.test_m_operator_execution_consistency \
  tests.test_m_operator_repeat_check tests.test_m_operator_feedback_collection \
  tests.test_m_operator_sampling_inspection tests.test_m_operator_feedback_v2 \
  tests.test_ccp_uncertainty_profile \
  > "$COLLECTION_ROOT/collection_tests.log" 2>&1; then
  cat "$COLLECTION_ROOT/collection_tests.log"
  exit 1
fi
cat "$COLLECTION_ROOT/collection_tests.log"
"$PYTHON" -u collect_m_operator_feedback_v2.py \
  --instances "$INSTANCES" --normalization "$NORMALIZATION" \
  --checkpoint "$V6_PPO_ROOT/final_policy.pt" \
  --source-root "$SOURCE_COLLECTION_ROOT" --out "$COLLECTION_ROOT" \
  --batch "$COLLECTION_BATCH" --workers "${MAX_WORKERS:-2}" \
  --max-attempts "${MAX_ATTEMPTS:-2}"
