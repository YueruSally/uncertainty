#!/usr/bin/env bash
set -euo pipefail

AUDIT_ROOT=${AUDIT_ROOT:-transformer_ppo_runs/diagnostics/ccp_i_downstream_unique_v1_10inst_3seed_b400}
ANALYSIS_ROOT=${ANALYSIS_ROOT:-$AUDIT_ROOT/offline_analysis}

if ! test -f "$AUDIT_ROOT/AUDIT_COMPLETE.json"; then
  echo "I AUDIT IS INCOMPLETE: $AUDIT_ROOT" >&2
  exit 1
fi

mkdir -p "$ANALYSIS_ROOT"

CANDIDATE_DATASET="$ANALYSIS_ROOT/i_candidate_ranker.jsonl"
STATE_DATASET="$ANALYSIS_ROOT/i_state_gate.jsonl"
DATASET_SUMMARY="$ANALYSIS_ROOT/dataset_summary.json"
GATE_REPORT="$ANALYSIS_ROOT/gate_report.json"
LINEAR_REPORT="$ANALYSIS_ROOT/ranker_linear.json"
MLP_REPORT="$ANALYSIS_ROOT/ranker_mlp.json"
EXTRA_REPORT="$ANALYSIS_ROOT/ranker_extra_trees.json"

python3 -u build_i_downstream_gate_dataset.py \
  --root "$AUDIT_ROOT" \
  --candidate-out "$CANDIDATE_DATASET" \
  --state-out "$STATE_DATASET" \
  --summary "$DATASET_SUMMARY"

python3 -u probe_c_downstream_gate.py \
  --dataset "$STATE_DATASET" \
  --out "$GATE_REPORT" \
  --seed 9971

python3 -u probe_v6_downstream_dataset.py \
  --dataset "$CANDIDATE_DATASET" \
  --out "$LINEAR_REPORT" \
  --model pairwise_linear \
  --alpha 0.5 \
  --temperature 0.25 \
  --seed 9971

python3 -u probe_v6_downstream_dataset.py \
  --dataset "$CANDIDATE_DATASET" \
  --out "$MLP_REPORT" \
  --model small_mlp \
  --alpha 0.5 \
  --temperature 0.25 \
  --seed 9971

python3 -u probe_v6_downstream_dataset.py \
  --dataset "$CANDIDATE_DATASET" \
  --out "$EXTRA_REPORT" \
  --model extra_trees \
  --alpha 0.5 \
  --temperature 0.25 \
  --seed 9971

jq -n \
  --slurpfile gate "$GATE_REPORT" \
  --slurpfile linear "$LINEAR_REPORT" \
  --slurpfile mlp "$MLP_REPORT" \
  --slurpfile extra "$EXTRA_REPORT" '
  def ranker($report):
    $report[0].aggregate
    | {
        model: $report[0].model,
        pairwise: .weighted_pairwise_accuracy,
        spearman: .weighted_mean_state_spearman,
        top_1: .weighted_final_top_1_accuracy,
        positive_folds: .positive_expected_final_folds,
        expected_gain: .weighted_model_minus_base_expected_final,
        states_above_base:
          .weighted_states_model_above_base_final_fraction,
        oracle_gap_captured: .weighted_oracle_gap_fraction_captured
      };
  {
    gate_positive_10: (
      $gate[0].labels.positive_10
      | {
          auc: .loo_roc_auc,
          average_precision: .loo_average_precision,
          top_20: .top_fraction[1]
        }
    ),
    rankers: [
      ranker($linear), ranker($mlp), ranker($extra)
    ]
  }
  | . + {
      admission_pass: (
        .gate_positive_10.top_20.precision >= 0.60
        and .gate_positive_10.top_20.mean_selected_oracle_percent >= 10
        and (
          [.rankers[]
           | select(
               .pairwise >= 0.55
               and .positive_folds >= 7
               and .states_above_base >= 0.60
             )]
          | length
        ) > 0
      )
    }
  ' > "$ANALYSIS_ROOT/QUICK_RESULT.json"

touch "$ANALYSIS_ROOT/ANALYSIS_COMPLETE"
cat "$ANALYSIS_ROOT/QUICK_RESULT.json"
