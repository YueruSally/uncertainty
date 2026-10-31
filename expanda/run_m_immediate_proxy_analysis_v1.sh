#!/usr/bin/env bash
set -euo pipefail

SOURCE_ROOT=${SOURCE_ROOT:-transformer_ppo_runs/diagnostics/ccp_m_continuation_mean_v1_60state_4action_4seed_b400}
OUT_ROOT=${OUT_ROOT:-transformer_ppo_runs/diagnostics/ccp_m_immediate_proxy_v1}
SOURCE_DATASET="$SOURCE_ROOT/continuation_mean_candidates.jsonl"

for required in \
  "$SOURCE_ROOT/PIPELINE_COMPLETE" \
  "$SOURCE_ROOT/QUICK_RESULT.json" \
  "$SOURCE_DATASET"
do
  if ! test -f "$required"; then
    echo "MISSING REQUIRED FILE: $required" >&2
    exit 1
  fi
done

if test "$(wc -l < "$SOURCE_DATASET")" -ne 240; then
  echo "EXPECTED 240 CONTINUATION-MEAN CANDIDATES" >&2
  exit 1
fi

mkdir -p "$OUT_ROOT"

ALL_DATASET="$OUT_ROOT/all_states.jsonl"
LATE_DATASET="$OUT_ROOT/late_generation_10_plus.jsonl"

cp "$SOURCE_DATASET" "$ALL_DATASET"
jq -c 'select(.generation >= 10)' "$SOURCE_DATASET" > "$LATE_DATASET"

if test "$(wc -l < "$LATE_DATASET")" -ne 120; then
  echo "EXPECTED 120 LATE-STAGE CANDIDATES" >&2
  exit 1
fi

TARGET_FIELD=immediate_archive_delta
FINAL_FIELD=continuation_mean_final_advantage

for SCOPE in all late; do
  if test "$SCOPE" = all; then
    DATASET="$ALL_DATASET"
  else
    DATASET="$LATE_DATASET"
  fi

  for MODEL in pairwise_linear small_mlp extra_trees; do
    REPORT="$OUT_ROOT/${SCOPE}_${MODEL}.json"

    echo "===== $SCOPE / $MODEL ====="

    python3 -u probe_v6_downstream_dataset.py \
      --dataset "$DATASET" \
      --out "$REPORT" \
      --model "$MODEL" \
      --target-field "$TARGET_FIELD" \
      --final-field "$FINAL_FIELD" \
      --alpha 0.5 \
      --temperature 0.25 \
      --seed 9991
  done
done

jq -n \
  --slurpfile all_linear "$OUT_ROOT/all_pairwise_linear.json" \
  --slurpfile all_mlp "$OUT_ROOT/all_small_mlp.json" \
  --slurpfile all_extra "$OUT_ROOT/all_extra_trees.json" \
  --slurpfile late_linear "$OUT_ROOT/late_pairwise_linear.json" \
  --slurpfile late_mlp "$OUT_ROOT/late_small_mlp.json" \
  --slurpfile late_extra "$OUT_ROOT/late_extra_trees.json" '
  def result($scope; $report):
    $report[0].aggregate
    | {
        scope: $scope,
        model: $report[0].model,
        informative_states: .informative_state_count,
        immediate_pairwise: .weighted_pairwise_accuracy,
        immediate_top_1: .weighted_target_top_1_accuracy,
        final_top_1: .weighted_final_top_1_accuracy,
        mean_selected_final_percent: .weighted_mean_model_top_final,
        selected_final_positive_fraction:
          .weighted_model_top_final_positive_fraction,
        positive_selected_final_folds: .positive_model_top_final_folds
      };
  [
    result("all"; $all_linear),
    result("all"; $all_mlp),
    result("all"; $all_extra),
    result("late_10_plus"; $late_linear),
    result("late_10_plus"; $late_mlp),
    result("late_10_plus"; $late_extra)
  ] as $results
  | ($results | sort_by(.immediate_pairwise) | reverse | .[0]) as $best
  | {
      definition: (
        "leave-one-instance-out prediction of immediate positive M actions; "
        "final metrics use four-continuation mean CCP HV percent"
      ),
      results: $results,
      best_by_immediate_pairwise: $best,
      admission_thresholds: {
        immediate_pairwise: 0.60,
        immediate_top_1: 0.40,
        mean_selected_final_percent: 1.0,
        selected_final_positive_fraction: 0.60,
        positive_selected_final_folds: 7
      }
    }
  | . + {
      admission_pass: (
        .best_by_immediate_pairwise.immediate_pairwise >=
          .admission_thresholds.immediate_pairwise
        and .best_by_immediate_pairwise.immediate_top_1 >=
          .admission_thresholds.immediate_top_1
        and .best_by_immediate_pairwise.mean_selected_final_percent >=
          .admission_thresholds.mean_selected_final_percent
        and .best_by_immediate_pairwise.selected_final_positive_fraction >=
          .admission_thresholds.selected_final_positive_fraction
        and .best_by_immediate_pairwise.positive_selected_final_folds >=
          .admission_thresholds.positive_selected_final_folds
      )
    }
  ' > "$OUT_ROOT/QUICK_RESULT.json"

touch "$OUT_ROOT/ANALYSIS_COMPLETE"

echo "===== M IMMEDIATE-PROXY V1 COMPLETE ====="
cat "$OUT_ROOT/QUICK_RESULT.json"
