#!/usr/bin/env bash
set -euo pipefail

INSTANCES=${INSTANCES:-/home/lunet/cokt2/uncertainty-transformer-ppo/expanda/transformer_ppo_runs/instances}
NORMALIZATION=${NORMALIZATION:-/home/lunet/cokt2/uncertainty-transformer-ppo/expanda/transformer_ppo_runs/reward_normalization.json}
BOOTSTRAP_CHECKPOINT=${BOOTSTRAP_CHECKPOINT:-transformer_ppo_runs/diagnostics/v5r_joint_replace_smoke_b160_p20_seed9501/M/final_policy.pt}
PPO_ROOT=${PPO_ROOT:-transformer_ppo_runs/diagnostics/ccp_m_ccp100_paired_heuristic_v4_ppo_14inst_b800}
OUT_ROOT=${OUT_ROOT:-transformer_ppo_runs/diagnostics/ccp_m_ccp100_paired_heuristic_v4_dev_3inst_3seed_b800}
OOS_ROOT=${OOS_ROOT:-transformer_ppo_runs/diagnostics/ccp_m_ccp100_paired_heuristic_v4_dev_oos_3inst_3seed_s1000}
MAX_WORKERS=${MAX_WORKERS:-3}
MAX_ATTEMPTS=${MAX_ATTEMPTS:-3}

if ! [[ "$MAX_WORKERS" =~ ^[1-9][0-9]*$ ]] || (( MAX_WORKERS > 3 )); then
  echo "MAX_WORKERS must be an integer in [1, 3]: $MAX_WORKERS" >&2
  exit 1
fi
if ! [[ "$MAX_ATTEMPTS" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_ATTEMPTS must be a positive integer: $MAX_ATTEMPTS" >&2
  exit 1
fi
for REQUIRED in \
  "$INSTANCES/catalog.json" \
  "$NORMALIZATION" \
  "$BOOTSTRAP_CHECKPOINT"; do
  if ! test -f "$REQUIRED"; then
    echo "MISSING REQUIRED INPUT: $REQUIRED" >&2
    exit 1
  fi
done

mkdir -p "$PPO_ROOT" "$OUT_ROOT" "$OOS_ROOT"
exec 9>"$OUT_ROOT/pipeline.lock"
if ! flock -n 9; then
  echo "ANOTHER M-V4 PIPELINE IS ALREADY RUNNING" >&2
  exit 75
fi

echo "===== TEST M-V4 SAME-STATE PAIRED HEURISTIC REWARD ====="
python3 -u -m unittest \
  tests.test_m_paired_heuristic_v4 \
  tests.test_m_prior_residual_v2 \
  tests.test_m_three_method_transformer_ppo \
  tests.test_learning_components_v2_controls.V2ControlTests.test_forced_replace_model_selector_executes_transformer_candidate \
  > "$OUT_ROOT/tests.log" 2>&1

echo "===== STAGE 1: FOURTEEN CCP100 PPO EPISODES / B800 / PAIRED HEURISTIC ====="
if ! test -f "$PPO_ROOT/training_summary.json"; then
  ATTEMPT=1
  while (( ATTEMPT <= MAX_ATTEMPTS )); do
    PPO_RESUME_ARGS=()
    if test -f "$PPO_ROOT/training_progress.json"; then
      PPO_RESUME_ARGS+=(--resume)
    fi
    echo "[M-V4-TRAIN] attempt=$ATTEMPT/$MAX_ATTEMPTS resume=${#PPO_RESUME_ARGS[@]}"
    set +e
    python3 -u train_transformer_ppo_ccp100.py \
      --instances "$INSTANCES" \
      --normalization "$NORMALIZATION" \
      --out "$PPO_ROOT" \
      --architecture transformer \
      --component M \
      --policy-seed 65201 \
      --evaluation-budget 800 \
      --evaluation-mode ccp \
      --uncertainty-profile transport_border_saa_v1 \
      --scenario-count 100 \
      --ccp-confidence 0.90 \
      --pop 20 \
      --gens 1000 \
      --instance-ids \
        train-S0-002 train-S0-005 train-S0-006 train-S0-007 \
        train-S0-008 train-S0-009 train-S0-011 train-S0-014 \
        train-S0-015 train-S0-016 train-S0-017 train-S0-018 \
        train-S0-019 train-S0-020 \
      --candidate-representation path_v5 \
      --credit-assignment paired_heuristic_terminal \
      --mutation-control replace_path \
      --mutation-action-prior joint_balanced \
      --mutation-prior-temperature 0.25 \
      --m-forced-replace-start-generation 10 \
      --actor-policy base_residual \
      --residual-logit-limit 0.50 \
      --rollout-steps 16 \
      --gae-horizon 8 \
      --actor-learning-rate 1e-5 \
      --critic-learning-rate 1e-5 \
      --entropy-coefficient 0.002 \
      --device cpu \
      "${PPO_RESUME_ARGS[@]}" \
      2>&1 | tee -a "$PPO_ROOT/training.log"
    STATUS=${PIPESTATUS[0]}
    set -e
    if (( STATUS == 0 )) && test -f "$PPO_ROOT/training_summary.json"; then
      break
    fi
    if (( ATTEMPT == MAX_ATTEMPTS )); then
      echo "M-V4 PPO FAILED AFTER $MAX_ATTEMPTS ATTEMPTS" >&2
      exit "$STATUS"
    fi
    ATTEMPT=$((ATTEMPT + 1))
  done
fi

if ! jq -e '
  .completed == true
  and .component == "M"
  and (.episodes | length) == 14
  and (.episodes | map(.ppo_updates) | add) > 0
  and (.episodes | map(.paired_reward_events) | add) > 0
  and (.episodes | map(.positive_reward_actions) | add) > 0
  and (.episodes | map(.negative_reward_actions) | add) > 0
  and .total_counterfactual_evaluations > 0
' "$PPO_ROOT/training_summary.json" >/dev/null; then
  echo "M-V4 PPO TRAINING GATE FAILED" >&2
  exit 1
fi
if ! jq -e '
  .evaluation_mode == "ccp"
  and .scenario_count == 100
  and .ccp_confidence == 0.90
  and .evaluation_budget_per_episode == 800
  and .ppo.actor_policy == "base_residual"
  and .ppo.residual_logit_limit == 0.50
  and .ppo.candidate_representation == "path_v5"
  and .ppo.credit_assignment == "paired_heuristic_terminal"
  and .ppo.mutation_control == "replace_path"
  and .ppo.mutation_action_prior == "joint_balanced"
  and .ppo.mutation_prior_temperature == 0.25
' "$PPO_ROOT/training_configuration.json" >/dev/null; then
  echo "M-V4 FROZEN TRAINING CONFIGURATION CHECK FAILED" >&2
  exit 1
fi

echo "===== STAGE 2: EQUAL CCP100 / POP20 / B800 THREE-METHOD DEVELOPMENT ====="
if ! test -f "$OUT_ROOT/COMPLETE.json"; then
  ATTEMPT=1
  while (( ATTEMPT <= MAX_ATTEMPTS )); do
    echo "[M-V4-EVAL] attempt=$ATTEMPT/$MAX_ATTEMPTS"
    set +e
    python3 -u evaluate_m_balanced_teacher_transformer.py \
      --instances "$INSTANCES" \
      --normalization "$NORMALIZATION" \
      --bootstrap-checkpoint "$BOOTSTRAP_CHECKPOINT" \
      --transformer-checkpoint "$PPO_ROOT/final_policy.pt" \
      --training-report "$PPO_ROOT/training_summary.json" \
      --heuristic-result-name HEURISTIC_BALANCED \
      --transformer-result-name TRANSFORMER_PPO \
      --out "$OUT_ROOT" \
      --instance-ids train-S0-021 train-S0-022 train-S0-023 \
      --algorithm-seeds 887001 887002 887003 \
      --minimum-generation 10 \
      --max-workers "$MAX_WORKERS" \
      --evaluation-budget 800 \
      --scenario-count 100 \
      --ccp-confidence 0.90 \
      --pop 20 \
      --gens 1000 \
      --device cpu \
      2>&1 | tee -a "$OUT_ROOT/evaluation.log"
    STATUS=${PIPESTATUS[0]}
    set -e
    if (( STATUS == 0 )) && test -f "$OUT_ROOT/COMPLETE.json"; then
      break
    fi
    if (( ATTEMPT == MAX_ATTEMPTS )); then
      echo "M-V4 DEVELOPMENT FAILED AFTER $MAX_ATTEMPTS ATTEMPTS" >&2
      exit "$STATUS"
    fi
    ATTEMPT=$((ATTEMPT + 1))
  done
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
      paired_reward_events: $r.ppo_training.paired_reward_events,
      paired_reward_ties: $r.ppo_training.paired_reward_ties,
      positive_reward_actions: $r.ppo_training.positive_reward_actions,
      negative_reward_actions: $r.ppo_training.negative_reward_actions,
      training_counterfactual_evaluations:
        $r.ppo_training.counterfactual_evaluations,
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
         and $model.mean_paired_percent > $heuristic.mean_paired_percent
         and $model.wins >= 5)
    }
' "$OUT_ROOT/COMPLETE.json" | tee "$OUT_ROOT/DEVELOPMENT_GATE.json"

touch "$OUT_ROOT/DEVELOPMENT_COMPLETE"
if ! jq -e '.pass == true' "$OUT_ROOT/DEVELOPMENT_GATE.json" >/dev/null; then
  touch "$OUT_ROOT/DEVELOPMENT_REJECTED"
  echo "===== M-V4 EQUAL-PARAMETER DEVELOPMENT DID NOT PASS ====="
  cat "$OUT_ROOT/DEVELOPMENT_GATE.json"
  exit 0
fi
touch "$OUT_ROOT/DEVELOPMENT_PASS" "$OUT_ROOT/VALIDATION_PASS"
echo "===== M-V4 EQUAL-PARAMETER DEVELOPMENT PASSED ====="

echo "===== STAGE 3: SAME FRONTS / ONE NEW S1000 OOS SCENARIO SET ====="
if ! test -f "$OOS_ROOT/OOS_COMPLETE.json"; then
  python3 -u evaluate_m_three_method_oos.py \
    --instances "$INSTANCES" \
    --normalization "$NORMALIZATION" \
    --validation-root "$OUT_ROOT" \
    --out "$OOS_ROOT" \
    --instance-ids train-S0-021 train-S0-022 train-S0-023 \
    --algorithm-seeds 887001 887002 887003 \
    --scenario-count 1000 \
    --oos-seed 89000000 \
    --alpha 0.90 \
    --uncertainty-profile transport_border_saa_v1 \
    --max-workers 2 \
    2>&1 | tee -a "$OOS_ROOT/evaluation.log"
fi

touch "$OOS_ROOT/OOS_PIPELINE_COMPLETE"
if jq -e '.admission_pass == true' "$OOS_ROOT/OOS_COMPLETE.json" >/dev/null; then
  touch "$OOS_ROOT/OOS_PASS" "$OUT_ROOT/PIPELINE_COMPLETE"
  echo "===== M-V4 CCP100 DEVELOPMENT AND S1000 OOS PASSED ====="
else
  touch "$OOS_ROOT/OOS_REJECTED" "$OUT_ROOT/PIPELINE_COMPLETE"
  echo "===== M-V4 CCP100 DEVELOPMENT PASSED; S1000 OOS DID NOT PASS ====="
fi
cat "$OUT_ROOT/DEVELOPMENT_GATE.json"
cat "$OOS_ROOT/QUICK_RESULT.json"
