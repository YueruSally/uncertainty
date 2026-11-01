#!/usr/bin/env bash
set -euo pipefail

FORMAL_ROOT=${FORMAL_ROOT:-transformer_ppo_runs/formal/ccp_im_directed_formal_v2_pop250_gen200_10newinst_3seed_s200_oos5000}
ANALYSIS_ROOT=${ANALYSIS_ROOT:-$FORMAL_ROOT/analysis}
MAX_WORKERS=${MAX_WORKERS:-4}

for SHARD in 0 1 2; do
  if ! test -f "$FORMAL_ROOT/shard_${SHARD}/SHARD_COMPLETE.json"; then
    echo "MISSING FORMAL SHARD: $SHARD" >&2
    exit 1
  fi
done

mkdir -p "$ANALYSIS_ROOT"
python3 -u analyze_im_ccp_formal_oos_v1.py \
  --plan "$FORMAL_ROOT/FORMAL_PLAN.json" \
  --formal-root "$FORMAL_ROOT" \
  --out "$ANALYSIS_ROOT" \
  --max-workers "$MAX_WORKERS" \
  | tee "$ANALYSIS_ROOT/analysis.log"

test -f "$ANALYSIS_ROOT/PIPELINE_COMPLETE"
echo "FORMAL BASE/I/M ANALYSIS COMPLETE"
