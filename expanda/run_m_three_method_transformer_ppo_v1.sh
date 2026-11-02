#!/usr/bin/env bash
set -euo pipefail

INSTANCES=${INSTANCES:-/home/lunet/cokt2/uncertainty-transformer-ppo/expanda/transformer_ppo_runs/instances}
NORMALIZATION=${NORMALIZATION:-/home/lunet/cokt2/uncertainty-transformer-ppo/expanda/transformer_ppo_runs/reward_normalization.json}
BOOTSTRAP_CHECKPOINT=${BOOTSTRAP_CHECKPOINT:-transformer_ppo_runs/diagnostics/v5r_joint_replace_smoke_b160_p20_seed9501/M/final_policy.pt}
DATA_ROOT=${DATA_ROOT:-transformer_ppo_runs/diagnostics/ccp_m_three_method_v1_teacher_data_7inst_20seed_b400}
BC_ROOT=${BC_ROOT:-transformer_ppo_runs/diagnostics/ccp_m_three_method_v2_soft_bc_stable}
PPO_ROOT=${PPO_ROOT:-transformer_ppo_runs/diagnostics/ccp_m_three_method_v2_ppo_7inst_b1200}
OUT_ROOT=${OUT_ROOT:-transformer_ppo_runs/diagnostics/ccp_m_three_method_v2_holdout_3inst_5seed_b400}

for REQUIRED in "$NORMALIZATION" "$BOOTSTRAP_CHECKPOINT"; do
  if ! test -f "$REQUIRED"; then
    echo "MISSING REQUIRED INPUT: $REQUIRED" >&2
    exit 1
  fi
done

mkdir -p "$DATA_ROOT" "$BC_ROOT" "$PPO_ROOT" "$OUT_ROOT"

echo "===== TEST M SOFT-BC + PPO THREE-METHOD PIPELINE ====="
python3 -u -m unittest \
  tests.test_m_balanced_teacher_transformer \
  tests.test_m_three_method_transformer_ppo \
  tests.test_learning_components_v2_controls.V2ControlTests.test_m_scheduled_operator_counts_base_before_start_generation \
  tests.test_learning_components_v2_controls.V2ControlTests.test_forced_replace_model_selector_executes_transformer_candidate \
  > "$OUT_ROOT/tests.log" 2>&1

echo "===== COLLECT >=3000 BALANCED M STATES ====="
python3 -u collect_m_teacher_rollouts.py \
  --instances "$INSTANCES" \
  --normalization "$NORMALIZATION" \
  --checkpoint "$BOOTSTRAP_CHECKPOINT" \
  --out "$DATA_ROOT" \
  --instance-ids \
    train-S0-002 train-S0-005 train-S0-006 train-S0-007 \
    train-S0-008 train-S0-009 train-S0-011 \
  --algorithm-seeds \
    881001 881002 881003 881004 881005 881006 881007 881008 \
    881009 881010 881011 881012 881013 881014 881015 881016 \
    881017 881018 881019 881020 \
  --minimum-generation 10 \
  --max-workers 12 \
  --evaluation-budget 400 \
  --scenario-count 200 \
  --ccp-confidence 0.90 \
  --pop 20 \
  --gens 1000 \
  --device cpu \
  | tee "$DATA_ROOT/collection.log"

if ! jq -e '
  .completed == true
  and .run_count == 140
  and .policy_state_count >= 3000
' "$DATA_ROOT/DATA_COMPLETE.json" >/dev/null; then
  echo "M TEACHER DATASET DID NOT REACH 3000 STATES" >&2
  exit 1
fi

if ! test -f "$BC_ROOT/TRAINING_COMPLETE.json"; then
  echo "===== SOFT-LABEL BEHAVIOUR PRETRAINING ====="
  python3 -u train_m_balanced_teacher_transformer.py \
    --data-root "$DATA_ROOT" \
    --checkpoint "$BOOTSTRAP_CHECKPOINT" \
    --out "$BC_ROOT" \
    --validation-instances S0_011 \
    --epochs 6 \
    --batch-size 8 \
    --early-stopping-patience 2 \
    --learning-rate 3e-5 \
    --weight-decay 1e-4 \
    --soft-label-temperature 0.25 \
    --seed 62011 \
    --device cpu \
    | tee "$BC_ROOT/training.log"
else
  echo "===== REUSE COMPLETE SOFT-BC CHECKPOINT ====="
fi

echo "===== ON-POLICY PPO WITH Q90-HV REWARD ====="
PPO_RESUME_ARGS=(--initial-checkpoint "$BC_ROOT/final_policy.pt")
if test -f "$PPO_ROOT/training_progress.json"; then
  PPO_RESUME_ARGS+=(--resume)
fi

if ! test -f "$PPO_ROOT/training_summary.json"; then
  python3 -u train_transformer_ppo_ccp100.py \
    --instances "$INSTANCES" \
    --normalization "$NORMALIZATION" \
    --out "$PPO_ROOT" \
    --architecture transformer \
    --component M \
    --policy-seed 64011 \
    --evaluation-budget 1200 \
    --evaluation-mode ccp \
    --uncertainty-profile transport_border_saa_v1 \
    --scenario-count 200 \
    --ccp-confidence 0.90 \
    --pop 20 \
    --gens 1000 \
    --instance-ids \
      train-S0-002 train-S0-005 train-S0-006 train-S0-007 \
      train-S0-008 train-S0-009 train-S0-011 \
    --candidate-representation path_v4 \
    --credit-assignment operation_terminal \
    --mutation-control replace_path \
    --m-forced-replace-start-generation 10 \
    --actor-policy direct \
    --rollout-steps 64 \
    --gae-horizon 8 \
    --actor-learning-rate 3e-5 \
    --critic-learning-rate 1e-5 \
    --entropy-coefficient 0.005 \
    --device cpu \
    "${PPO_RESUME_ARGS[@]}" \
    | tee "$PPO_ROOT/training.log"
else
  echo "===== REUSE COMPLETE PPO CHECKPOINT ====="
fi

if ! jq -e '
  .completed == true
  and .component == "M"
  and (.episodes | map(.ppo_updates) | add) > 0
  and .total_counterfactual_evaluations == 0
' "$PPO_ROOT/training_summary.json" >/dev/null; then
  echo "M PPO TRAINING COMPLETENESS CHECK FAILED" >&2
  exit 1
fi

echo "===== MATCHED HOLDOUT: BASE / HEURISTIC / TRANSFORMER-PPO ====="
python3 -u evaluate_m_balanced_teacher_transformer.py \
  --instances "$INSTANCES" \
  --normalization "$NORMALIZATION" \
  --bootstrap-checkpoint "$BOOTSTRAP_CHECKPOINT" \
  --transformer-checkpoint "$PPO_ROOT/final_policy.pt" \
  --training-report "$PPO_ROOT/training_summary.json" \
  --heuristic-result-name HEURISTIC_BALANCED \
  --transformer-result-name TRANSFORMER_PPO \
  --out "$OUT_ROOT" \
  --instance-ids train-S0-010 train-S0-012 train-S0-013 \
  --algorithm-seeds 883001 883002 883003 883004 883005 \
  --minimum-generation 10 \
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
  and .instance_count == 3
  and .algorithm_seed_count == 5
  and .run_count == 45
  and .scenario_exact == true
  and .ppo_training.ppo_updates > 0
  and ([.strategies[].strategy]
       == ["BASE", "HEURISTIC_BALANCED", "TRANSFORMER_PPO"])
' "$OUT_ROOT/QUICK_RESULT.json" >/dev/null; then
  echo "M THREE-METHOD RESULT COMPLETENESS CHECK FAILED" >&2
  exit 1
fi

touch "$OUT_ROOT/PIPELINE_COMPLETE"

echo "===== M THREE-METHOD TRANSFORMER-PPO EXPERIMENT COMPLETE ====="
cat "$OUT_ROOT/QUICK_RESULT.json"
