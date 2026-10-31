#!/usr/bin/env bash
set -euo pipefail

AUDIT_ROOT=${AUDIT_ROOT:-transformer_ppo_runs/diagnostics/ccp_m_downstream_unique_v1_10inst_3seed_b400}
CHECKPOINT=${CHECKPOINT:-transformer_ppo_runs/diagnostics/v5r_joint_replace_smoke_b160_p20_seed9501/M/final_policy.pt}
OUT_ROOT=${OUT_ROOT:-transformer_ppo_runs/diagnostics/ccp_m_downstream_transformer_v3_safe_base}
FOLDS=${FOLDS:?set FOLDS to one or more space-separated folds, for example 'S0_002 S0_005'}

if ! test -f "$AUDIT_ROOT/AUDIT_COMPLETE.json"; then
  echo "MISSING M AUDIT: $AUDIT_ROOT/AUDIT_COMPLETE.json" >&2
  exit 1
fi
if ! test -f "$CHECKPOINT"; then
  echo "MISSING M CHECKPOINT: $CHECKPOINT" >&2
  exit 1
fi

mkdir -p "$OUT_ROOT/logs"
read -r -a FOLD_ARRAY <<< "$FOLDS"

python3 -u train_m_downstream_transformer.py \
  --audit-root "$AUDIT_ROOT" \
  --checkpoint "$CHECKPOINT" \
  --out "$OUT_ROOT" \
  --folds "${FOLD_ARRAY[@]}" \
  --epochs 12 \
  --batch-size 4 \
  --learning-rate 3e-5 \
  --weight-decay 1e-4 \
  --rank-temperature 2.0 \
  --gate-threshold 10.0 \
  --training-objective safe_base_anchor \
  --score-scale 10.0 \
  --safe-quantile 0.25 \
  --safe-margin 0.10 \
  --safe-negative-weight 3.0 \
  --seed 9991 \
  --device cpu
