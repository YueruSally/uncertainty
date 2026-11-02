#!/usr/bin/env bash
set -euo pipefail

SOURCE_INSTANCES=${SOURCE_INSTANCES:-/home/lunet/cokt2/uncertainty-transformer-ppo/expanda/transformer_ppo_runs/instances}
NORMALIZATION=${NORMALIZATION:-/home/lunet/cokt2/uncertainty-transformer-ppo/expanda/transformer_ppo_runs/reward_normalization.json}
BOOTSTRAP_CHECKPOINT=${BOOTSTRAP_CHECKPOINT:-transformer_ppo_runs/diagnostics/v5r_joint_replace_smoke_b160_p20_seed9501/M/final_policy.pt}
PPO_ROOT=${PPO_ROOT:-transformer_ppo_runs/diagnostics/ccp_m_ccp100_equal_b800_v3_r16_ppo_7inst}
VALIDATION_INSTANCES=${VALIDATION_INSTANCES:-transformer_ppo_runs/diagnostics/ccp_m_ccp100_independent_v1_instances_5}
OUT_ROOT=${OUT_ROOT:-transformer_ppo_runs/diagnostics/ccp_m_ccp100_independent_v1_5inst_3seed_b800}
MAX_WORKERS=${MAX_WORKERS:-3}
MAX_ATTEMPTS=${MAX_ATTEMPTS:-3}

if ! [[ "$MAX_WORKERS" =~ ^[1-9][0-9]*$ ]] \
    || (( MAX_WORKERS > 3 )); then
  echo "MAX_WORKERS must be an integer in [1, 3]: $MAX_WORKERS" >&2
  exit 1
fi
if ! [[ "$MAX_ATTEMPTS" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_ATTEMPTS must be a positive integer: $MAX_ATTEMPTS" >&2
  exit 1
fi

for REQUIRED in \
  "$SOURCE_INSTANCES/catalog.json" \
  "$NORMALIZATION" \
  "$BOOTSTRAP_CHECKPOINT" \
  "$PPO_ROOT/final_policy.pt" \
  "$PPO_ROOT/training_summary.json"; do
  if ! test -f "$REQUIRED"; then
    echo "MISSING REQUIRED INPUT: $REQUIRED" >&2
    exit 1
  fi
done

mkdir -p "$OUT_ROOT"

echo "===== TEST RESUMABLE INDEPENDENT CCP100 M VALIDATION ====="
python3 -u -m unittest \
  tests.test_m_three_method_transformer_ppo \
  > "$OUT_ROOT/tests.log" 2>&1

echo "===== GENERATE FIVE PREVIOUSLY UNSEEN S0 INSTANCES ====="
if ! test -f "$VALIDATION_INSTANCES/catalog.json"; then
  if test -d "$VALIDATION_INSTANCES" \
      && test -n "$(find "$VALIDATION_INSTANCES" -mindepth 1 -maxdepth 1 -print -quit)"; then
    STAMP=$(date -u +%Y%m%dT%H%M%SZ)
    INTERRUPTED="${VALIDATION_INSTANCES}_interrupted_${STAMP}"
    mv "$VALIDATION_INSTANCES" "$INTERRUPTED"
    echo "[FINAL-TEST] moved interrupted instance set to $INTERRUPTED"
  fi
  python3 -u generate_s0_final_test_instances.py \
    --source-instances "$SOURCE_INSTANCES" \
    --out "$VALIDATION_INSTANCES" \
    --count 5 \
    --master-seed 72000000 \
    --ccp-seed-base 75000000 \
    --oos-seed-base 76000000 \
    | tee "$OUT_ROOT/instance_generation.log"
fi

if ! jq -e '
  .instance_count == 5
  and .training_instances_used == false
  and .development_test_instances_used == false
  and .optimisation_outcomes_used == false
  and .model_selection_outcomes_used == false
' "$VALIDATION_INSTANCES/catalog.json" >/dev/null; then
  echo "INDEPENDENT INSTANCE CATALOG VALIDATION FAILED" >&2
  exit 1
fi

echo "===== FIVE-INSTANCE / THREE-SEED EQUAL-PARAMETER VALIDATION ====="
if ! test -f "$OUT_ROOT/COMPLETE.json"; then
  ATTEMPT=1
  while (( ATTEMPT <= MAX_ATTEMPTS )); do
    echo "[VALIDATION] attempt=$ATTEMPT/$MAX_ATTEMPTS"
    set +e
    python3 -u evaluate_m_balanced_teacher_transformer.py \
      --instances "$VALIDATION_INSTANCES" \
      --normalization "$NORMALIZATION" \
      --bootstrap-checkpoint "$BOOTSTRAP_CHECKPOINT" \
      --transformer-checkpoint "$PPO_ROOT/final_policy.pt" \
      --training-report "$PPO_ROOT/training_summary.json" \
      --heuristic-result-name HEURISTIC_BALANCED \
      --transformer-result-name TRANSFORMER_PPO \
      --out "$OUT_ROOT" \
      --instance-ids \
        final-test-S0-001 final-test-S0-002 final-test-S0-003 \
        final-test-S0-004 final-test-S0-005 \
      --algorithm-seeds 886001 886002 886003 \
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
      echo "VALIDATION FAILED AFTER $MAX_ATTEMPTS ATTEMPTS" >&2
      if (( STATUS == 0 )); then
        STATUS=1
      fi
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
      validation_role: "previously unseen instances; no model selection",
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
        mean_95ci_low: $model.mean_95ci_low,
        mean_95ci_high: $model.mean_95ci_high,
        mean_optimisation_seconds: $model.mean_optimisation_seconds,
        wins: $model.wins,
        runs: $model.runs
      },
      transformer_minus_heuristic_percentage_points:
        ($model.mean_paired_percent - $heuristic.mean_paired_percent),
      pass:
        ($r.completed == true
         and $r.scenario_exact == true
         and $r.instance_count == 5
         and $r.algorithm_seed_count == 3
         and $r.run_count == 45
         and $model.mean_paired_percent > $heuristic.mean_paired_percent
         and $model.mean_95ci_low > 0
         and $model.wins >= 8)
    }
' "$OUT_ROOT/COMPLETE.json" \
  | tee "$OUT_ROOT/INDEPENDENT_VALIDATION.json"

touch "$OUT_ROOT/VALIDATION_COMPLETE"
if jq -e '.pass == true' "$OUT_ROOT/INDEPENDENT_VALIDATION.json" >/dev/null; then
  touch "$OUT_ROOT/VALIDATION_PASS"
  echo "===== INDEPENDENT CCP100 M VALIDATION PASSED ====="
else
  touch "$OUT_ROOT/VALIDATION_REJECTED"
  echo "===== INDEPENDENT CCP100 M VALIDATION DID NOT PASS ====="
fi
cat "$OUT_ROOT/INDEPENDENT_VALIDATION.json"
