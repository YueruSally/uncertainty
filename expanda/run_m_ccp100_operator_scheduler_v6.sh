#!/usr/bin/env bash
set -euo pipefail

INSTANCES=${INSTANCES:-/home/lunet/cokt2/uncertainty-transformer-ppo/expanda/transformer_ppo_runs/instances}
NORMALIZATION=${NORMALIZATION:-/home/lunet/cokt2/uncertainty-transformer-ppo/expanda/transformer_ppo_runs/reward_normalization.json}
BOOTSTRAP_CHECKPOINT=${BOOTSTRAP_CHECKPOINT:-transformer_ppo_runs/diagnostics/v5r_joint_replace_smoke_b160_p20_seed9501/M/final_policy.pt}
PPO_ROOT=${PPO_ROOT:-transformer_ppo_runs/diagnostics/ccp_m_ccp100_operator_scheduler_v6_ppo_5inst_b400}
OUT_ROOT=${OUT_ROOT:-transformer_ppo_runs/diagnostics/ccp_m_ccp100_operator_scheduler_v6_quick_2inst_3seed_b400}
MAX_WORKERS=${MAX_WORKERS:-3}
MAX_ATTEMPTS=${MAX_ATTEMPTS:-3}

if ! [[ "$MAX_WORKERS" =~ ^[1-3]$ ]]; then
  echo "MAX_WORKERS must be 1, 2, or 3: $MAX_WORKERS" >&2
  exit 1
fi
if ! [[ "$MAX_ATTEMPTS" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_ATTEMPTS must be positive: $MAX_ATTEMPTS" >&2
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

mkdir -p "$PPO_ROOT" "$OUT_ROOT"
exec 9>"$OUT_ROOT/pipeline.lock"
if ! flock -n 9; then
  echo "ANOTHER M-V6 PIPELINE IS ALREADY RUNNING" >&2
  exit 75
fi

echo "===== TEST M-V6 FIVE-OPERATOR TRANSFORMER-PPO ====="
python3 -u -m unittest \
  tests.test_m_operator_scheduler_v6 \
  tests.test_m_generation_reward_v3 \
  tests.test_transformer_ppo_framework \
  > "$OUT_ROOT/tests.log" 2>&1

echo "===== STAGE 1: FIVE-INSTANCE CCP100 OPERATOR PPO / B400 ====="
if ! test -f "$PPO_ROOT/training_summary.json"; then
  ATTEMPT=1
  while (( ATTEMPT <= MAX_ATTEMPTS )); do
    RESUME_ARGS=()
    if test -f "$PPO_ROOT/training_progress.json"; then
      RESUME_ARGS+=(--resume)
    fi
    echo "[M-V6-TRAIN] attempt=$ATTEMPT/$MAX_ATTEMPTS resume=${#RESUME_ARGS[@]}"
    set +e
    python3 -u train_transformer_ppo_ccp100.py \
      --instances "$INSTANCES" \
      --normalization "$NORMALIZATION" \
      --out "$PPO_ROOT" \
      --architecture transformer \
      --component M \
      --policy-seed 67201 \
      --evaluation-budget 400 \
      --evaluation-mode ccp \
      --uncertainty-profile transport_border_saa_v1 \
      --scenario-count 100 \
      --ccp-confidence 0.90 \
      --pop 20 \
      --gens 1000 \
      --instance-ids \
        train-S0-002 train-S0-005 train-S0-006 train-S0-007 train-S0-008 \
      --candidate-representation path_v3 \
      --credit-assignment generation_terminal \
      --mutation-control operator \
      --actor-policy base_residual \
      --residual-logit-limit 2.0 \
      --rollout-steps 16 \
      --gae-horizon 8 \
      --actor-learning-rate 3e-5 \
      --critic-learning-rate 1e-5 \
      --entropy-coefficient 0.01 \
      --device cpu \
      "${RESUME_ARGS[@]}" \
      2>&1 | tee -a "$PPO_ROOT/training.log"
    STATUS=${PIPESTATUS[0]}
    set -e
    if (( STATUS == 0 )) && test -f "$PPO_ROOT/training_summary.json"; then
      break
    fi
    if (( ATTEMPT == MAX_ATTEMPTS )); then
      echo "M-V6 PPO FAILED AFTER $MAX_ATTEMPTS ATTEMPTS" >&2
      exit "$STATUS"
    fi
    ATTEMPT=$((ATTEMPT + 1))
  done
fi

if ! jq -e '
  .completed == true
  and .component == "M"
  and (.episodes | length) == 5
  and (.episodes | map(.ppo_updates) | add) > 0
  and (.episodes | map(.generation_reward_events) | add) > 0
  and .total_actual_evaluations == 2000
  and .total_counterfactual_evaluations == 0
' "$PPO_ROOT/training_summary.json" >/dev/null; then
  echo "M-V6 TRAINING GATE FAILED" >&2
  exit 1
fi
if ! jq -e '
  .evaluation_mode == "ccp"
  and .scenario_count == 100
  and .ccp_confidence == 0.90
  and .evaluation_budget_per_episode == 400
  and .ppo.actor_policy == "base_residual"
  and .ppo.candidate_representation == "path_v3"
  and .ppo.credit_assignment == "generation_terminal"
  and .ppo.mutation_control == "operator"
' "$PPO_ROOT/training_configuration.json" >/dev/null; then
  echo "M-V6 FROZEN CONFIGURATION CHECK FAILED" >&2
  exit 1
fi

echo "===== STAGE 2: CCP100 BASE / HEURISTIC / FIVE-OPERATOR PPO ====="
if ! test -f "$OUT_ROOT/COMPLETE.json"; then
  ATTEMPT=1
  while (( ATTEMPT <= MAX_ATTEMPTS )); do
    echo "[M-V6-EVAL] attempt=$ATTEMPT/$MAX_ATTEMPTS"
    set +e
    python3 -u evaluate_m_balanced_teacher_transformer.py \
      --instances "$INSTANCES" \
      --normalization "$NORMALIZATION" \
      --bootstrap-checkpoint "$BOOTSTRAP_CHECKPOINT" \
      --transformer-checkpoint "$PPO_ROOT/final_policy.pt" \
      --training-report "$PPO_ROOT/training_summary.json" \
      --heuristic-result-name HEURISTIC_BALANCED \
      --transformer-result-name TRANSFORMER_PPO \
      --transformer-action-scope operator \
      --out "$OUT_ROOT" \
      --instance-ids train-S0-021 train-S0-022 \
      --algorithm-seeds 889001 889002 889003 \
      --minimum-generation 10 \
      --max-workers "$MAX_WORKERS" \
      --evaluation-budget 400 \
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
      echo "M-V6 EVALUATION FAILED AFTER $MAX_ATTEMPTS ATTEMPTS" >&2
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
  | [
      $r.runs[]
      | select(.strategy == "TRANSFORMER_PPO")
      | . as $t
      | ($r.runs[] | select(
          .instance == $t.instance
          and .algorithm_seed == $t.algorithm_seed
          and .strategy == "HEURISTIC_BALANCED")) as $h
      | ($t.paired_percent - $h.paired_percent)
    ] as $paired
  | {
      completed: true,
      design: {
        instances: $r.instance_count,
        seeds: $r.algorithm_seed_count,
        scenario_count: 100,
        ccp_confidence: 0.90,
        population_size: 20,
        evaluation_budget: 400,
        equal_test_parameters: true,
        transformer_action_count: 5
      },
      ppo_updates: $r.ppo_training.ppo_updates,
      base_mean_percent: $base.mean_paired_percent,
      heuristic_mean_percent: $heuristic.mean_paired_percent,
      transformer_mean_percent: $model.mean_paired_percent,
      transformer_minus_heuristic_points: ($paired | add / length),
      transformer_vs_heuristic_wins: ($paired | map(select(. > 0)) | length),
      paired_runs: ($paired | length),
      pass: (
        $r.scenario_exact
        and $r.ppo_training.ppo_updates > 0
        and ($paired | add / length) > 0
        and ($paired | map(select(. > 0)) | length) >= 4
      )
    }
' "$OUT_ROOT/COMPLETE.json" > "$OUT_ROOT/OPERATOR_GATE.json"

cat "$OUT_ROOT/OPERATOR_GATE.json"
if jq -e '.pass == true' "$OUT_ROOT/OPERATOR_GATE.json" >/dev/null; then
  touch "$OUT_ROOT/PIPELINE_COMPLETE"
  echo "===== M-V6 FIVE-OPERATOR QUICK GATE PASSED ====="
else
  touch "$OUT_ROOT/DEVELOPMENT_REJECTED"
  echo "===== M-V6 FIVE-OPERATOR QUICK GATE DID NOT PASS ====="
fi
