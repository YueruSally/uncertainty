#!/usr/bin/env bash
set -euo pipefail

INSTANCES=${INSTANCES:-/home/lunet/cokt2/uncertainty-transformer-ppo/expanda/transformer_ppo_runs/instances}
NORMALIZATION=${NORMALIZATION:-/home/lunet/cokt2/uncertainty-transformer-ppo/expanda/transformer_ppo_runs/reward_normalization.json}
CHECKPOINT=${CHECKPOINT:-transformer_ppo_runs/diagnostics/ccp_nocustoms_3ep_b400_p20/I/final_policy.pt}
SMOKE_ROOT=${SMOKE_ROOT:-transformer_ppo_runs/diagnostics/ccp_i_downstream_unique_v1_smoke}
AUDIT_ROOT=${AUDIT_ROOT:-transformer_ppo_runs/diagnostics/ccp_i_downstream_unique_v1_10inst_3seed_b400}

for required in "$NORMALIZATION" "$CHECKPOINT"; do
  if ! test -f "$required"; then
    echo "MISSING REQUIRED FILE: $required" >&2
    exit 1
  fi
done

mkdir -p "$SMOKE_ROOT" "$AUDIT_ROOT"

echo "===== TESTS ====="

python3 -u -m unittest \
  tests.test_i_single_interventions \
  tests.test_initialization_learning.InitializationControllerTests.test_one_shot_path_diagnostic_replays_base_except_forced_choice \
  tests.test_initialization_learning.InitializationControllerTests.test_one_shot_path_diagnostic_without_target_is_exact_base \
  > "$AUDIT_ROOT/tests.log" 2>&1

echo "===== CHECKPOINT ====="

CHECKPOINT="$CHECKPOINT" python3 - > "$AUDIT_ROOT/checkpoint.log" 2>&1 <<'PY'
import os

from transformer_ppo.config import PPOConfig
from transformer_ppo.ppo import PPOTrainer

checkpoint = os.environ["CHECKPOINT"]
trainer = PPOTrainer.load(
    checkpoint, PPOConfig, lambda *_: None,
    device="cpu", training=False)
print({
    "component": trainer.component,
    "initialization_control": trainer.config.initialization_control,
    "candidate_representation": trainer.config.candidate_representation,
})
if trainer.component != "I":
    raise RuntimeError("checkpoint is not an I policy")
PY

echo "===== SMOKE ====="

python3 -u audit_i_single_interventions.py \
  --instances "$INSTANCES" \
  --normalization "$NORMALIZATION" \
  --checkpoint "$CHECKPOINT" \
  --out "$SMOKE_ROOT" \
  --instance-ids train-S0-005 \
  --algorithm-seeds 777001 \
  --states-per-instance 2 \
  --actions-per-state 4 \
  --max-workers 4 \
  --evaluation-budget 400 \
  --evaluation-mode ccp \
  --uncertainty-profile transport_border_saa_v1 \
  --scenario-count 200 \
  --ccp-confidence 0.90 \
  --pop 20 \
  --gens 1000 \
  --device cpu \
  > "$SMOKE_ROOT/audit.log" 2>&1

echo "===== FORMAL I AUDIT ====="

python3 -u audit_i_single_interventions.py \
  --instances "$INSTANCES" \
  --normalization "$NORMALIZATION" \
  --checkpoint "$CHECKPOINT" \
  --out "$AUDIT_ROOT" \
  --instance-ids \
    train-S0-002 train-S0-005 train-S0-006 train-S0-007 train-S0-008 \
    train-S0-009 train-S0-010 train-S0-011 train-S0-012 train-S0-013 \
  --algorithm-seeds 777001 777002 777003 \
  --states-per-instance 4 \
  --actions-per-state 10 \
  --max-workers 8 \
  --evaluation-budget 400 \
  --evaluation-mode ccp \
  --uncertainty-profile transport_border_saa_v1 \
  --scenario-count 200 \
  --ccp-confidence 0.90 \
  --pop 20 \
  --gens 1000 \
  --device cpu \
  > "$AUDIT_ROOT/audit.log" 2>&1

jq '{
  completed,
  instance_count,
  algorithm_seeds,
  state_count,
  branch_run_count,
  mean_audited_oracle_percent,
  median_audited_oracle_percent,
  maximum_audited_oracle_percent,
  states_with_positive_oracle,
  positive_state_fraction:
    (.states_with_positive_oracle / .state_count),
  states_with_oracle_above_5_percent,
  states_with_oracle_above_10_percent,
  above_10_state_fraction:
    (.states_with_oracle_above_10_percent / .state_count),
  positive_action_count,
  negative_action_count,
  zero_action_count,
  p000_exact_base,
  scenario_exact,
  state_replay_exact,
  forced_execution_exact,
  admission_pass:
    (
      .mean_audited_oracle_percent >= 5
      and (.states_with_positive_oracle / .state_count) >= 0.30
      and (.states_with_oracle_above_10_percent / .state_count) >= 0.20
    )
}' "$AUDIT_ROOT/AUDIT_COMPLETE.json" \
  > "$AUDIT_ROOT/QUICK_RESULT.json"

touch "$AUDIT_ROOT/PIPELINE_COMPLETE"

echo "===== I AUDIT COMPLETE ====="
cat "$AUDIT_ROOT/QUICK_RESULT.json"
