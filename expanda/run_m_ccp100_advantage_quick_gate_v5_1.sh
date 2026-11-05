#!/usr/bin/env bash
set -euo pipefail

INSTANCES=${INSTANCES:-/home/lunet/cokt2/uncertainty-transformer-ppo/expanda/transformer_ppo_runs/instances}
NORMALIZATION=${NORMALIZATION:-/home/lunet/cokt2/uncertainty-transformer-ppo/expanda/transformer_ppo_runs/reward_normalization.json}
BOOTSTRAP_CHECKPOINT=${BOOTSTRAP_CHECKPOINT:-transformer_ppo_runs/diagnostics/v5r_joint_replace_smoke_b160_p20_seed9501/M/final_policy.pt}
V4_DATA_ROOT=${V4_DATA_ROOT:-transformer_ppo_runs/diagnostics/ccp_m_ccp100_paired_heuristic_v4_ppo_14inst_b800}
ADV_ROOT=${ADV_ROOT:-transformer_ppo_runs/diagnostics/ccp_m_ccp100_advantage_v5_1_quick_pretrain_128}
PPO_ROOT=${PPO_ROOT:-transformer_ppo_runs/diagnostics/ccp_m_ccp100_advantage_v5_1_quick_ppo_3inst_b400}
OUT_ROOT=${OUT_ROOT:-transformer_ppo_runs/diagnostics/ccp_m_ccp100_advantage_v5_1_quick_test_2inst_3seed_b400}
MAX_WORKERS=${MAX_WORKERS:-3}

for REQUIRED in \
  "$INSTANCES/catalog.json" \
  "$NORMALIZATION" \
  "$BOOTSTRAP_CHECKPOINT" \
  "$V4_DATA_ROOT/final_policy.pt" \
  "$V4_DATA_ROOT/training_summary.json"; do
  if ! test -f "$REQUIRED"; then
    echo "MISSING REQUIRED INPUT: $REQUIRED" >&2
    exit 1
  fi
done

mkdir -p "$ADV_ROOT" "$PPO_ROOT" "$OUT_ROOT"
exec 9>"$OUT_ROOT/pipeline.lock"
if ! flock -n 9; then
  echo "ANOTHER M-V5.1 QUICK GATE IS RUNNING" >&2
  exit 75
fi

echo "===== TEST M-V5.1 QUICK LEARNABILITY GATE ====="
python3 -u -m unittest \
  tests.test_m_paired_advantage_v5 \
  tests.test_m_paired_heuristic_v4 \
  > "$OUT_ROOT/tests.log" 2>&1

echo "===== QUICK STAGE 1: 128-STATE / TWO-EPOCH ADVANTAGE PRETRAIN ====="
if ! test -f "$ADV_ROOT/TRAINING_COMPLETE.json"; then
  python3 -u train_m_paired_advantage_transformer.py \
    --data-root "$V4_DATA_ROOT" \
    --checkpoint "$V4_DATA_ROOT/final_policy.pt" \
    --out "$ADV_ROOT" \
    --validation-instances S0_019 S0_020 \
    --maximum-training-states 128 \
    --maximum-validation-states 80 \
    --epochs 2 \
    --batch-size 1 \
    --early-stopping-patience 2 \
    --learning-rate 1e-5 \
    --weight-decay 1e-4 \
    --positive-override-weight 4.0 \
    --pairwise-weight 1.0 \
    --pairwise-margin 0.05 \
    --residual-logit-limit 2.0 \
    --minimum-validation-override-recall 0.10 \
    --minimum-validation-fallback-accuracy 0.80 \
    --seed 66401 \
    --device cpu \
    2>&1 | tee -a "$ADV_ROOT/training.log"
fi

if ! jq -e '
  .completed == true
  and .training_states == 128
  and .validation_states == 80
  and .selected_epoch > 0
  and .residual_logit_limit == 2.0
  and .validation.override_recall >= 0.10
  and .validation.fallback_accuracy >= 0.80
' "$ADV_ROOT/TRAINING_COMPLETE.json" >/dev/null; then
  touch "$OUT_ROOT/QUICK_PRETRAIN_REJECTED"
  echo "===== M-V5.1 QUICK PRETRAIN DID NOT PASS ====="
  cat "$ADV_ROOT/TRAINING_COMPLETE.json"
  exit 0
fi
echo "===== M-V5.1 QUICK PRETRAIN PASSED ====="

echo "===== QUICK STAGE 2: THREE CCP100 PPO EPISODES / B400 ====="
if ! test -f "$PPO_ROOT/training_summary.json"; then
  python3 -u train_transformer_ppo_ccp100.py \
    --instances "$INSTANCES" \
    --normalization "$NORMALIZATION" \
    --out "$PPO_ROOT" \
    --architecture transformer \
    --component M \
    --policy-seed 66402 \
    --evaluation-budget 400 \
    --evaluation-mode ccp \
    --uncertainty-profile transport_border_saa_v1 \
    --scenario-count 100 \
    --ccp-confidence 0.90 \
    --pop 20 \
    --gens 1000 \
    --instance-ids train-S0-002 train-S0-005 train-S0-006 \
    --candidate-representation path_v5 \
    --credit-assignment paired_heuristic_terminal \
    --mutation-control replace_path \
    --mutation-action-prior joint_balanced \
    --mutation-prior-temperature 0.25 \
    --m-forced-replace-start-generation 10 \
    --actor-policy base_residual \
    --residual-logit-limit 2.0 \
    --rollout-steps 16 \
    --gae-horizon 8 \
    --actor-learning-rate 5e-6 \
    --critic-learning-rate 1e-5 \
    --entropy-coefficient 0.001 \
    --initial-checkpoint "$ADV_ROOT/final_policy.pt" \
    --device cpu \
    2>&1 | tee -a "$PPO_ROOT/training.log"
fi

echo "===== QUICK STAGE 3: EQUAL CCP100 / POP20 / B400 THREE-METHOD TEST ====="
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
fi

jq '
  . as $r
  | ($r.strategies[] | select(.strategy == "BASE")) as $base
  | ($r.strategies[] | select(.strategy == "HEURISTIC_BALANCED")) as $heuristic
  | ($r.strategies[] | select(.strategy == "TRANSFORMER_PPO")) as $model
  | {
      completed: $r.completed,
      comparison_design: {
        instance_count: 2,
        algorithm_seed_count: 3,
        run_count: 18,
        scenario_count: 100,
        ccp_confidence: 0.90,
        population_size: 20,
        evaluation_budget_per_run: 400,
        equal_test_parameters: true
      },
      base_mean_paired_percent: $base.mean_paired_percent,
      heuristic_mean_paired_percent: $heuristic.mean_paired_percent,
      transformer_mean_paired_percent: $model.mean_paired_percent,
      transformer_minus_heuristic_percentage_points:
        ($model.mean_paired_percent - $heuristic.mean_paired_percent),
      transformer_wins: $model.wins,
      transformer_runs: $model.runs,
      pass: (
        $r.completed == true
        and $r.scenario_exact == true
        and $model.mean_paired_percent > $heuristic.mean_paired_percent
        and $model.wins >= 4)
    }
' "$OUT_ROOT/COMPLETE.json" | tee "$OUT_ROOT/QUICK_GATE.json"

touch "$OUT_ROOT/QUICK_GATE_COMPLETE"
if jq -e '.pass == true' "$OUT_ROOT/QUICK_GATE.json" >/dev/null; then
  touch "$OUT_ROOT/QUICK_GATE_PASS"
  echo "===== M-V5.1 QUICK GATE PASSED: FULL RUN IS JUSTIFIED ====="
else
  touch "$OUT_ROOT/QUICK_GATE_REJECTED"
  echo "===== M-V5.1 QUICK GATE DID NOT PASS: DO NOT START FULL RUN ====="
fi
cat "$OUT_ROOT/QUICK_GATE.json"
