#!/usr/bin/env bash
set -euo pipefail

AUDIT_ROOT=${AUDIT_ROOT:-transformer_ppo_runs/diagnostics/ccp_m_downstream_unique_v1_10inst_3seed_b400}
OUT_ROOT=${OUT_ROOT:-transformer_ppo_runs/diagnostics/ccp_m_plus3_offline_v1}

if ! test -f "$AUDIT_ROOT/AUDIT_COMPLETE.json"; then
  echo "M AUDIT IS INCOMPLETE: $AUDIT_ROOT" >&2
  exit 1
fi

if ! jq -e '
  .completed == true
  and .p000_exact_base == true
  and .scenario_exact == true
  and .duplicate_replay_consistent == true
  and .forced_execution_exact == true
' "$AUDIT_ROOT/AUDIT_COMPLETE.json" >/dev/null; then
  echo "M AUDIT FAILED EXACTNESS PREREQUISITES: $AUDIT_ROOT" >&2
  exit 1
fi

mkdir -p "$OUT_ROOT"

CANDIDATE_DATASET="$OUT_ROOT/m_plus3_candidates.jsonl"
STATE_DATASET="$OUT_ROOT/m_plus3_states.jsonl"
DATASET_SUMMARY="$OUT_ROOT/dataset_summary.json"
LINEAR_REPORT="$OUT_ROOT/ranker_linear.json"
MLP_REPORT="$OUT_ROOT/ranker_mlp.json"
EXTRA_REPORT="$OUT_ROOT/ranker_extra_trees.json"
TARGET_FIELD=plus_3_generations_archive_advantage

echo "===== BUILD PAIRED +3-GENERATION DATASET ====="

python3 -u build_m_downstream_gate_dataset.py \
  --root "$AUDIT_ROOT" \
  --candidate-out "$CANDIDATE_DATASET" \
  --state-out "$STATE_DATASET" \
  --summary "$DATASET_SUMMARY" \
  --target-horizon plus_3_generations

echo "===== LINEAR LOO ====="

python3 -u probe_v6_downstream_dataset.py \
  --dataset "$CANDIDATE_DATASET" \
  --out "$LINEAR_REPORT" \
  --model pairwise_linear \
  --target-field "$TARGET_FIELD" \
  --final-field final_archive_advantage \
  --alpha 0.5 \
  --temperature 0.25 \
  --seed 9971

echo "===== MLP LOO ====="

python3 -u probe_v6_downstream_dataset.py \
  --dataset "$CANDIDATE_DATASET" \
  --out "$MLP_REPORT" \
  --model small_mlp \
  --target-field "$TARGET_FIELD" \
  --final-field final_archive_advantage \
  --alpha 0.5 \
  --temperature 0.25 \
  --seed 9971

echo "===== EXTRA TREES LOO ====="

python3 -u probe_v6_downstream_dataset.py \
  --dataset "$CANDIDATE_DATASET" \
  --out "$EXTRA_REPORT" \
  --model extra_trees \
  --target-field "$TARGET_FIELD" \
  --final-field final_archive_advantage \
  --alpha 0.5 \
  --temperature 0.25 \
  --seed 9971

jq -n \
  --slurpfile dataset "$DATASET_SUMMARY" \
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
        positive_target_folds: .positive_expected_target_folds,
        positive_final_folds: .positive_expected_final_folds,
        expected_plus3_gain: .weighted_model_minus_base_expected_target,
        expected_final_gain: .weighted_model_minus_base_expected_final,
        target_states_above_base:
          .weighted_states_model_above_base_target_fraction,
        final_states_above_base:
          .weighted_states_model_above_base_final_fraction
      };
  [ranker($linear), ranker($mlp), ranker($extra)] as $rankers
  | ($rankers | sort_by(.pairwise) | reverse | .[0]) as $best
  | {
      dataset: {
        source_states: $dataset[0].source_state_count,
        retained_states: $dataset[0].state_count,
        dropped_states: $dataset[0].dropped_state_count,
        candidates: $dataset[0].candidate_count,
        instances: $dataset[0].instance_count,
        target_final_sign_accuracy:
          $dataset[0].target_final_sign_accuracy,
        target_final_pearson: $dataset[0].target_final_pearson
      },
      rankers: $rankers,
      best_by_pairwise: $best,
      admission_thresholds: {
        pairwise: 0.60,
        target_top_1: 0.40,
        positive_target_folds: 7,
        positive_final_folds: 6,
        expected_plus3_gain: 0,
        expected_final_gain: 0
      }
    }
  | . + {
      admission_pass: (
        .dataset.instances == 10
        and .best_by_pairwise.pairwise >= .admission_thresholds.pairwise
        and .best_by_pairwise.target_top_1 >=
          .admission_thresholds.target_top_1
        and .best_by_pairwise.positive_target_folds >=
          .admission_thresholds.positive_target_folds
        and .best_by_pairwise.positive_final_folds >=
          .admission_thresholds.positive_final_folds
        and .best_by_pairwise.expected_plus3_gain >
          .admission_thresholds.expected_plus3_gain
        and .best_by_pairwise.expected_final_gain >
          .admission_thresholds.expected_final_gain
      )
    }
  ' > "$OUT_ROOT/QUICK_RESULT.json"

touch "$OUT_ROOT/ANALYSIS_COMPLETE"

echo "===== M +3 OFFLINE ANALYSIS COMPLETE ====="
cat "$OUT_ROOT/QUICK_RESULT.json"
