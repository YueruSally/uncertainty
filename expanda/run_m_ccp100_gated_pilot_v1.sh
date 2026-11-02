#!/usr/bin/env bash
set -euo pipefail

INSTANCES=${INSTANCES:-/home/lunet/cokt2/uncertainty-transformer-ppo/expanda/transformer_ppo_runs/instances}
NORMALIZATION=${NORMALIZATION:-/home/lunet/cokt2/uncertainty-transformer-ppo/expanda/transformer_ppo_runs/reward_normalization.json}
BOOTSTRAP_CHECKPOINT=${BOOTSTRAP_CHECKPOINT:-transformer_ppo_runs/diagnostics/v5r_joint_replace_smoke_b160_p20_seed9501/M/final_policy.pt}
DATA_ROOT=${DATA_ROOT:-transformer_ppo_runs/diagnostics/ccp_m_three_method_v1_teacher_data_7inst_20seed_b400}
BC_ROOT=${BC_ROOT:-transformer_ppo_runs/diagnostics/ccp_m_ccp100_gate_v1_bc_800train_150valid}
PPO_ROOT=${PPO_ROOT:-transformer_ppo_runs/diagnostics/ccp_m_ccp100_gate_v1_ppo_3inst_b400}
OUT_ROOT=${OUT_ROOT:-transformer_ppo_runs/diagnostics/ccp_m_ccp100_gate_v1_holdout_2inst_3seed_b400}

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

echo "===== TEST CCP100 GATED M PILOT ====="
python3 -u -m unittest \
  tests.test_m_balanced_teacher_transformer \
  tests.test_m_three_method_transformer_ppo \
  tests.test_learning_components_v2_controls.V2ControlTests.test_m_scheduled_operator_counts_base_before_start_generation \
  tests.test_learning_components_v2_controls.V2ControlTests.test_forced_replace_model_selector_executes_transformer_candidate \
  > "$OUT_ROOT/tests.log" 2>&1

echo "===== GATE 1: TWO-EPOCH SOFT-BC DEVELOPMENT CHECK ====="
if ! test -f "$BC_ROOT/TRAINING_COMPLETE.json"; then
  python3 -u train_m_balanced_teacher_transformer.py \
    --data-root "$DATA_ROOT" \
    --checkpoint "$BOOTSTRAP_CHECKPOINT" \
    --out "$BC_ROOT" \
    --validation-instances S0_011 \
    --maximum-training-states 800 \
    --maximum-validation-states 150 \
    --epochs 2 \
    --batch-size 8 \
    --early-stopping-patience 2 \
    --learning-rate 3e-5 \
    --weight-decay 1e-4 \
    --soft-label-temperature 0.25 \
    --seed 62101 \
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
         and $after <= (0.995 * $before))
    }
' "$BC_ROOT/TRAINING_COMPLETE.json" \
  | tee "$OUT_ROOT/BC_GATE.json"

if ! jq -e '.pass == true' "$OUT_ROOT/BC_GATE.json" >/dev/null; then
  touch "$OUT_ROOT/PILOT_COMPLETE" "$OUT_ROOT/BC_REJECTED"
  echo "===== STOP: SOFT-BC GATE DID NOT PASS ====="
  cat "$OUT_ROOT/BC_GATE.json"
  exit 0
fi

echo "===== GATE 2: THREE-INSTANCE CCP100 PPO / B400 ====="
if ! test -f "$PPO_ROOT/training_summary.json"; then
  python3 -u train_transformer_ppo_ccp100.py \
    --instances "$INSTANCES" \
    --normalization "$NORMALIZATION" \
    --out "$PPO_ROOT" \
    --architecture transformer \
    --component M \
    --policy-seed 64101 \
    --evaluation-budget 400 \
    --evaluation-mode ccp \
    --uncertainty-profile transport_border_saa_v1 \
    --scenario-count 100 \
    --ccp-confidence 0.90 \
    --pop 20 \
    --gens 1000 \
    --instance-ids train-S0-002 train-S0-005 train-S0-006 \
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
    --initial-checkpoint "$BC_ROOT/final_policy.pt" \
    --device cpu \
    | tee "$PPO_ROOT/training.log"
fi

if ! jq -e '
  .completed == true
  and .component == "M"
  and (.episodes | map(.ppo_updates) | add) > 0
  and (.episodes | map(.positive_reward_actions) | add) > 0
  and .total_counterfactual_evaluations == 0
' "$PPO_ROOT/training_summary.json" >/dev/null; then
  touch "$OUT_ROOT/PILOT_COMPLETE" "$OUT_ROOT/PPO_REJECTED"
  echo "===== STOP: PPO UPDATE/REWARD GATE DID NOT PASS ====="
  cat "$PPO_ROOT/training_summary.json"
  exit 0
fi

echo "===== GATE 3: CCP100 BASE / HEURISTIC / TRANSFORMER-PPO ====="
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
    --instance-ids train-S0-012 train-S0-013 \
    --algorithm-seeds 884001 884002 884003 \
    --minimum-generation 10 \
    --max-workers 9 \
    --evaluation-budget 400 \
    --scenario-count 100 \
    --ccp-confidence 0.90 \
    --pop 20 \
    --gens 1000 \
    --device cpu \
    | tee "$OUT_ROOT/evaluation.log"
fi

jq '
  . as $r
  | ($r.strategies[] | select(.strategy == "HEURISTIC_BALANCED")
     | .mean_paired_percent) as $heuristic
  | ($r.strategies[] | select(.strategy == "TRANSFORMER_PPO")) as $model
  | {
      completed: $r.completed,
      scenario_exact: $r.scenario_exact,
      ppo_updates: $r.ppo_training.ppo_updates,
      positive_reward_actions: $r.ppo_training.positive_reward_actions,
      heuristic_mean_paired_percent: $heuristic,
      transformer_mean_paired_percent: $model.mean_paired_percent,
      transformer_wins: $model.wins,
      transformer_runs: $model.runs,
      transformer_minus_heuristic_percentage_points:
        ($model.mean_paired_percent - $heuristic),
      pass:
        ($r.completed == true
         and $r.scenario_exact == true
         and $r.ppo_training.ppo_updates > 0
         and $r.ppo_training.positive_reward_actions > 0
         and $model.mean_paired_percent > 0
         and $model.wins >= 3
         and $model.mean_paired_percent >= ($heuristic - 2.0))
    }
' "$OUT_ROOT/COMPLETE.json" \
  | tee "$OUT_ROOT/PILOT_GATE.json"

touch "$OUT_ROOT/PILOT_COMPLETE"
if jq -e '.pass == true' "$OUT_ROOT/PILOT_GATE.json" >/dev/null; then
  touch "$OUT_ROOT/PILOT_PASS"
  echo "===== CCP100 GATED M PILOT PASSED ====="
else
  touch "$OUT_ROOT/PILOT_REJECTED"
  echo "===== CCP100 GATED M PILOT COMPLETED BUT DID NOT PASS ====="
fi
cat "$OUT_ROOT/PILOT_GATE.json"
