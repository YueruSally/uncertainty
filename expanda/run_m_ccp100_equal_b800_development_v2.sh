#!/usr/bin/env bash
set -euo pipefail

INSTANCES=${INSTANCES:-/home/lunet/cokt2/uncertainty-transformer-ppo/expanda/transformer_ppo_runs/instances}
NORMALIZATION=${NORMALIZATION:-/home/lunet/cokt2/uncertainty-transformer-ppo/expanda/transformer_ppo_runs/reward_normalization.json}
BOOTSTRAP_CHECKPOINT=${BOOTSTRAP_CHECKPOINT:-transformer_ppo_runs/diagnostics/v5r_joint_replace_smoke_b160_p20_seed9501/M/final_policy.pt}
DATA_ROOT=${DATA_ROOT:-transformer_ppo_runs/diagnostics/ccp_m_three_method_v1_teacher_data_7inst_20seed_b400}
BC_ROOT=${BC_ROOT:-transformer_ppo_runs/diagnostics/ccp_m_ccp100_equal_b800_v2_bc_2000train_300valid}
PPO_ROOT=${PPO_ROOT:-transformer_ppo_runs/diagnostics/ccp_m_ccp100_equal_b800_v3_r16_ppo_7inst}
OUT_ROOT=${OUT_ROOT:-transformer_ppo_runs/diagnostics/ccp_m_ccp100_equal_b800_v3_r16_holdout_3inst_3seed}
PPO_ROLLOUT_STEPS=${PPO_ROLLOUT_STEPS:-16}

if ! [[ "$PPO_ROLLOUT_STEPS" =~ ^[1-9][0-9]*$ ]] \
    || (( PPO_ROLLOUT_STEPS > 16 )); then
  echo "PPO_ROLLOUT_STEPS must be an integer in [1, 16]: $PPO_ROLLOUT_STEPS" >&2
  exit 1
fi

for REQUIRED in \
  "$NORMALIZATION" \
  "$BOOTSTRAP_CHECKPOINT" \
  "$DATA_ROOT/DATA_COMPLETE.json"; do
  if ! test -f "$REQUIRED"; then
    echo "MISSING REQUIRED INPUT: $REQUIRED" >&2
    exit 1
  fi
done

mkdir -p "$BC_ROOT" "$PPO_ROOT" "$OUT_ROOT"

echo "===== TEST EQUAL-BUDGET CCP100 M DEVELOPMENT ====="
python3 -u -m unittest \
  tests.test_m_balanced_teacher_transformer \
  tests.test_m_three_method_transformer_ppo \
  tests.test_learning_components_v2_controls.V2ControlTests.test_m_scheduled_operator_counts_base_before_start_generation \
  tests.test_learning_components_v2_controls.V2ControlTests.test_forced_replace_model_selector_executes_transformer_candidate \
  > "$OUT_ROOT/tests.log" 2>&1

echo "===== STAGE 1: THREE-EPOCH SOFT-BC / 2000 TRAIN / 300 VALID ====="
if ! test -f "$BC_ROOT/TRAINING_COMPLETE.json"; then
  python3 -u train_m_balanced_teacher_transformer.py \
    --data-root "$DATA_ROOT" \
    --checkpoint "$BOOTSTRAP_CHECKPOINT" \
    --out "$BC_ROOT" \
    --validation-instances S0_011 \
    --maximum-training-states 2000 \
    --maximum-validation-states 300 \
    --epochs 3 \
    --batch-size 8 \
    --early-stopping-patience 3 \
    --learning-rate 3e-5 \
    --weight-decay 1e-4 \
    --soft-label-temperature 0.25 \
    --seed 62201 \
    --device cpu \
    | tee "$BC_ROOT/training.log"
fi

jq '
  . as $r
  | ($r.baseline.validation.mean_soft_cross_entropy) as $before
  | ($r.validation.mean_soft_cross_entropy) as $after
  | {
      completed: $r.completed,
      training_states: $r.training.states,
      validation_states: $r.validation.states,
      selected_epoch: $r.selected_epoch,
      baseline_validation_soft_ce: $before,
      selected_validation_soft_ce: $after,
      relative_validation_soft_ce_improvement:
        (($before - $after) / $before),
      baseline_validation_top_1:
        $r.baseline.validation.top_1_accuracy,
      selected_validation_top_1:
        $r.validation.top_1_accuracy,
      pass:
        ($r.completed == true
         and $r.selected_epoch > 0
         and $after < $before)
    }
' "$BC_ROOT/TRAINING_COMPLETE.json" \
  | tee "$OUT_ROOT/BC_GATE.json"

if ! jq -e '.pass == true' "$OUT_ROOT/BC_GATE.json" >/dev/null; then
  touch "$OUT_ROOT/DEVELOPMENT_COMPLETE" "$OUT_ROOT/BC_REJECTED"
  echo "===== STOP: SOFT-BC DID NOT IMPROVE VALIDATION SOFT-CE ====="
  cat "$OUT_ROOT/BC_GATE.json"
  exit 0
fi

echo "===== STAGE 2: SEVEN-INSTANCE CCP100 PPO / B800 / R${PPO_ROLLOUT_STEPS} ====="
if ! test -f "$PPO_ROOT/training_summary.json"; then
  PPO_RESUME_ARGS=(--initial-checkpoint "$BC_ROOT/final_policy.pt")
  if test -f "$PPO_ROOT/training_progress.json"; then
    PPO_RESUME_ARGS+=(--resume)
  fi
  python3 -u train_transformer_ppo_ccp100.py \
    --instances "$INSTANCES" \
    --normalization "$NORMALIZATION" \
    --out "$PPO_ROOT" \
    --architecture transformer \
    --component M \
    --policy-seed 64201 \
    --evaluation-budget 800 \
    --evaluation-mode ccp \
    --uncertainty-profile transport_border_saa_v1 \
    --scenario-count 100 \
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
    --rollout-steps "$PPO_ROLLOUT_STEPS" \
    --gae-horizon 8 \
    --actor-learning-rate 3e-5 \
    --critic-learning-rate 1e-5 \
    --entropy-coefficient 0.005 \
    --device cpu \
    "${PPO_RESUME_ARGS[@]}" \
    | tee "$PPO_ROOT/training.log"
fi

if ! jq -e '
  .completed == true
  and .component == "M"
  and (.episodes | length) == 7
  and (.episodes | map(.ppo_updates) | add) > 0
  and (.episodes | map(.positive_reward_actions) | add) > 0
  and .total_counterfactual_evaluations == 0
' "$PPO_ROOT/training_summary.json" >/dev/null; then
  touch "$OUT_ROOT/DEVELOPMENT_COMPLETE" "$OUT_ROOT/PPO_REJECTED"
  echo "===== STOP: PPO UPDATE/REWARD GATE DID NOT PASS ====="
  cat "$PPO_ROOT/training_summary.json"
  exit 0
fi

echo "===== STAGE 3: EQUAL B800 CCP100 THREE-METHOD HOLDOUT ====="
if ! test -f "$OUT_ROOT/COMPLETE.json"; then
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
    --algorithm-seeds 885001 885002 885003 \
    --minimum-generation 10 \
    --max-workers 9 \
    --evaluation-budget 800 \
    --scenario-count 100 \
    --ccp-confidence 0.90 \
    --pop 20 \
    --gens 1000 \
    --device cpu \
    | tee "$OUT_ROOT/evaluation.log"
fi

jq '
  . as $r
  | ($r.strategies[] | select(.strategy == "BASE")) as $base
  | ($r.strategies[] | select(.strategy == "HEURISTIC_BALANCED")) as $heuristic
  | ($r.strategies[] | select(.strategy == "TRANSFORMER_PPO")) as $model
  | {
      completed: $r.completed,
      comparison_design: {
        instance_count: $r.instance_count,
        algorithm_seed_count: $r.algorithm_seed_count,
        run_count: $r.run_count,
        scenario_count: 100,
        ccp_confidence: 0.90,
        population_size: 20,
        evaluation_budget_per_run: 800,
        equal_test_parameters: true
      },
      scenario_exact: $r.scenario_exact,
      ppo_updates: $r.ppo_training.ppo_updates,
      positive_reward_actions: $r.ppo_training.positive_reward_actions,
      base: {
        mean_paired_percent: $base.mean_paired_percent,
        mean_optimisation_seconds: $base.mean_optimisation_seconds
      },
      heuristic: {
        mean_paired_percent: $heuristic.mean_paired_percent,
        mean_optimisation_seconds: $heuristic.mean_optimisation_seconds
      },
      transformer_ppo: {
        mean_paired_percent: $model.mean_paired_percent,
        mean_optimisation_seconds: $model.mean_optimisation_seconds,
        wins: $model.wins,
        runs: $model.runs
      },
      transformer_minus_heuristic_percentage_points:
        ($model.mean_paired_percent - $heuristic.mean_paired_percent),
      pass:
        ($r.completed == true
         and $r.scenario_exact == true
         and $r.instance_count == 3
         and $r.algorithm_seed_count == 3
         and $r.run_count == 27
         and $r.ppo_training.ppo_updates > 0
         and $r.ppo_training.positive_reward_actions > 0
         and $model.mean_paired_percent > $heuristic.mean_paired_percent
         and $model.wins >= 5)
    }
' "$OUT_ROOT/COMPLETE.json" \
  | tee "$OUT_ROOT/DEVELOPMENT_GATE.json"

touch "$OUT_ROOT/DEVELOPMENT_COMPLETE"
if jq -e '.pass == true' "$OUT_ROOT/DEVELOPMENT_GATE.json" >/dev/null; then
  touch "$OUT_ROOT/DEVELOPMENT_PASS"
  echo "===== EQUAL-BUDGET CCP100 M DEVELOPMENT PASSED ====="
else
  touch "$OUT_ROOT/DEVELOPMENT_REJECTED"
  echo "===== EQUAL-BUDGET CCP100 M DEVELOPMENT DID NOT PASS ====="
fi
cat "$OUT_ROOT/DEVELOPMENT_GATE.json"
