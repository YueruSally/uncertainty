#!/usr/bin/env bash
set -euo pipefail

INSTANCES=${INSTANCES:-/home/lunet/cokt2/uncertainty-transformer-ppo/expanda/transformer_ppo_runs/instances}
NORMALIZATION=${NORMALIZATION:-/home/lunet/cokt2/uncertainty-transformer-ppo/expanda/transformer_ppo_runs/reward_normalization.json}
CHECKPOINT=${CHECKPOINT:-transformer_ppo_runs/diagnostics/v5r_joint_replace_smoke_b160_p20_seed9501/M/final_policy.pt}
SOURCE_AUDIT_ROOT=${SOURCE_AUDIT_ROOT:-transformer_ppo_runs/diagnostics/ccp_m_downstream_unique_v1_10inst_3seed_b400}
OUT_ROOT=${OUT_ROOT:-transformer_ppo_runs/diagnostics/ccp_m_continuation_mean_v1_60state_4action_4seed_b400}

for required in \
  "$NORMALIZATION" \
  "$CHECKPOINT" \
  "$SOURCE_AUDIT_ROOT/AUDIT_COMPLETE.json"
do
  if ! test -f "$required"; then
    echo "MISSING REQUIRED FILE: $required" >&2
    exit 1
  fi
done

if ! jq -e '
  .completed == true
  and .p000_exact_base == true
  and .scenario_exact == true
  and .duplicate_replay_consistent == true
  and .forced_execution_exact == true
' "$SOURCE_AUDIT_ROOT/AUDIT_COMPLETE.json" >/dev/null; then
  echo "SOURCE M AUDIT FAILED EXACTNESS PREREQUISITES" >&2
  exit 1
fi

mkdir -p "$OUT_ROOT"

echo "===== TEST OUTCOME-INDEPENDENT SAMPLING AND MEAN LABELS ====="

python3 -u -m unittest \
  tests.test_m_continuation_stability \
  tests.test_m_continuation_mean_dataset \
  tests.test_learning_components_v2_controls.V2ControlTests.test_m_continuation_seed_restarts_python_and_numpy_after_target \
  tests.test_learning_components_v2_controls.V2ControlTests.test_m_continuation_event_and_seed_are_atomic \
  > "$OUT_ROOT/tests.log" 2>&1

echo "===== COLLECT: 60 STATES x 4 ACTIONS x 4 CONTINUATIONS ====="

python3 -u collect_m_continuation_stability.py \
  --audit-root "$SOURCE_AUDIT_ROOT" \
  --instances "$INSTANCES" \
  --normalization "$NORMALIZATION" \
  --checkpoint "$CHECKPOINT" \
  --out "$OUT_ROOT" \
  --continuation-seeds 32001 32002 32003 32004 \
  --state-count 60 \
  --actions-per-state 4 \
  --state-selection stratified \
  --action-selection index_spread \
  --max-workers 8 \
  --evaluation-budget 400 \
  --evaluation-mode ccp \
  --uncertainty-profile transport_border_saa_v1 \
  --scenario-count 200 \
  --ccp-confidence 0.90 \
  --pop 20 \
  --gens 1000 \
  --device cpu \
  | tee "$OUT_ROOT/collection.log"

if ! jq -e '
  .completed == true
  and .selected_state_count == 60
  and .selected_action_count == 240
  and .continuation_seed_count == 4
  and .branch_record_count == 960
  and .total_run_count == 1200
  and .scenario_exact == true
  and .forced_execution_exact == true
  and .pre_intervention_state_replay_exact == true
  and .selection.state_selection == "stratified"
  and .selection.action_selection == "index_spread"
' "$OUT_ROOT/STABILITY_COMPLETE.json" >/dev/null; then
  echo "CONTINUATION COLLECTION FAILED COMPLETENESS/EXACTNESS CHECK" >&2
  exit 1
fi

SOURCE_CANDIDATES="$OUT_ROOT/source_candidates.jsonl"
SOURCE_STATES="$OUT_ROOT/source_states.jsonl"
SOURCE_SUMMARY="$OUT_ROOT/source_dataset_summary.json"
CANDIDATE_DATASET="$OUT_ROOT/continuation_mean_candidates.jsonl"
STATE_DATASET="$OUT_ROOT/continuation_mean_states.jsonl"
DATASET_SUMMARY="$OUT_ROOT/continuation_mean_dataset_summary.json"
GATE_REPORT="$OUT_ROOT/gate_report.json"
LINEAR_REPORT="$OUT_ROOT/ranker_linear.json"
MLP_REPORT="$OUT_ROOT/ranker_mlp.json"
EXTRA_REPORT="$OUT_ROOT/ranker_extra_trees.json"
TARGET_FIELD=continuation_risk_adjusted_final_advantage
FINAL_FIELD=continuation_mean_final_advantage

echo "===== BUILD SOURCE FEATURES ====="

python3 -u build_m_downstream_gate_dataset.py \
  --root "$SOURCE_AUDIT_ROOT" \
  --candidate-out "$SOURCE_CANDIDATES" \
  --state-out "$SOURCE_STATES" \
  --summary "$SOURCE_SUMMARY" \
  --target-horizon final

echo "===== BUILD CONTINUATION-MEAN DATASET ====="

python3 -u build_m_continuation_mean_dataset.py \
  --stability-root "$OUT_ROOT" \
  --source-candidates "$SOURCE_CANDIDATES" \
  --source-states "$SOURCE_STATES" \
  --candidate-out "$CANDIDATE_DATASET" \
  --state-out "$STATE_DATASET" \
  --summary "$DATASET_SUMMARY" \
  --risk-lambda 1.0

echo "===== STATE GATE LOO ====="

python3 -u probe_c_downstream_gate.py \
  --dataset "$STATE_DATASET" \
  --out "$GATE_REPORT" \
  --labels positive_any \
  --seed 9981

for MODEL in pairwise_linear small_mlp extra_trees; do
  case "$MODEL" in
    pairwise_linear) REPORT="$LINEAR_REPORT" ;;
    small_mlp) REPORT="$MLP_REPORT" ;;
    extra_trees) REPORT="$EXTRA_REPORT" ;;
  esac

  echo "===== CANDIDATE RANKER $MODEL LOO ====="

  python3 -u probe_v6_downstream_dataset.py \
    --dataset "$CANDIDATE_DATASET" \
    --out "$REPORT" \
    --model "$MODEL" \
    --target-field "$TARGET_FIELD" \
    --final-field "$FINAL_FIELD" \
    --alpha 0.5 \
    --temperature 0.25 \
    --seed 9981
done

jq -n \
  --slurpfile dataset "$DATASET_SUMMARY" \
  --slurpfile gate "$GATE_REPORT" \
  --slurpfile linear "$LINEAR_REPORT" \
  --slurpfile mlp "$MLP_REPORT" \
  --slurpfile extra "$EXTRA_REPORT" '
  def ranker($report):
    $report[0].aggregate
    | {
        model: $report[0].model,
        pairwise: .weighted_pairwise_accuracy,
        target_top_1: .weighted_target_top_1_accuracy,
        final_top_1: .weighted_final_top_1_accuracy,
        mean_selected_safe_score: .weighted_mean_model_top_target,
        mean_selected_final_percent: .weighted_mean_model_top_final,
        selected_safe_fraction:
          .weighted_model_top_target_positive_fraction,
        selected_final_positive_fraction:
          .weighted_model_top_final_positive_fraction,
        positive_selected_safe_folds: .positive_model_top_target_folds,
        positive_selected_final_folds: .positive_model_top_final_folds
      };
  [ranker($linear), ranker($mlp), ranker($extra)] as $rankers
  | ($rankers | sort_by(.pairwise) | reverse | .[0]) as $best
  | ($gate[0].labels.positive_any.top_fraction[1]) as $gate20
  | {
      dataset: {
        instances: $dataset[0].instance_count,
        states: $dataset[0].state_count,
        candidates: $dataset[0].candidate_count,
        continuations_per_action:
          $dataset[0].continuation_count_per_action,
        positive_risk_states: $dataset[0].positive_risk_states,
        positive_risk_candidates: $dataset[0].positive_risk_candidates,
        mean_action_standard_deviation:
          $dataset[0].mean_action_standard_deviation,
        mean_action_positive_fraction:
          $dataset[0].mean_action_positive_fraction
      },
      gate_top_20: $gate20,
      rankers: $rankers,
      best_by_pairwise: $best,
      admission_thresholds: {
        gate_precision: 0.60,
        gate_mean_safe_oracle_percent: 1.0,
        pairwise: 0.60,
        target_top_1: 0.40,
        mean_selected_safe_score: 0.0,
        mean_selected_final_percent: 1.0,
        selected_final_positive_fraction: 0.60,
        positive_selected_final_folds: 7
      }
    }
  | . + {
      admission_pass: (
        .dataset.instances == 10
        and .dataset.states == 60
        and .dataset.candidates == 240
        and .dataset.continuations_per_action == 4
        and .gate_top_20.precision >=
          .admission_thresholds.gate_precision
        and .gate_top_20.mean_selected_oracle_percent >=
          .admission_thresholds.gate_mean_safe_oracle_percent
        and .best_by_pairwise.pairwise >=
          .admission_thresholds.pairwise
        and .best_by_pairwise.target_top_1 >=
          .admission_thresholds.target_top_1
        and .best_by_pairwise.mean_selected_safe_score >
          .admission_thresholds.mean_selected_safe_score
        and .best_by_pairwise.mean_selected_final_percent >=
          .admission_thresholds.mean_selected_final_percent
        and .best_by_pairwise.selected_final_positive_fraction >=
          .admission_thresholds.selected_final_positive_fraction
        and .best_by_pairwise.positive_selected_final_folds >=
          .admission_thresholds.positive_selected_final_folds
      )
    }
  ' > "$OUT_ROOT/QUICK_RESULT.json"

touch "$OUT_ROOT/PIPELINE_COMPLETE"

echo "===== M CONTINUATION-MEAN V1 COMPLETE ====="
cat "$OUT_ROOT/QUICK_RESULT.json"
