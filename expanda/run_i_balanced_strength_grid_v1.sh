#!/usr/bin/env bash
set -euo pipefail

INSTANCES=${INSTANCES:-/home/lunet/cokt2/uncertainty-transformer-ppo/expanda/transformer_ppo_runs/instances}
NORMALIZATION=${NORMALIZATION:-/home/lunet/cokt2/uncertainty-transformer-ppo/expanda/transformer_ppo_runs/reward_normalization.json}
REUSE_BASE_ROOT=${REUSE_BASE_ROOT:-transformer_ppo_runs/diagnostics/ccp_m_directed_balanced_validation_v1_10inst_10seed_b400}
OUT_ROOT=${OUT_ROOT:-transformer_ppo_runs/diagnostics/ccp_i_balanced_strength_grid_v1_10inst_3seed_b400}

for REQUIRED in \
  "$NORMALIZATION" \
  "$REUSE_BASE_ROOT/PIPELINE_COMPLETE"
do
  if ! test -e "$REQUIRED"; then
    echo "MISSING REQUIRED INPUT: $REQUIRED" >&2
    exit 1
  fi
done

mkdir -p "$OUT_ROOT"

echo "===== TEST I BALANCED STRENGTH CONTROL ====="
python3 -u -m unittest \
  tests.test_initialization_rule \
  > "$OUT_ROOT/tests.log" 2>&1

echo "===== 10-INSTANCE / 3-SEED I STRENGTH GRID ====="
python3 -u evaluate_i_directed_initialization.py \
  --instances "$INSTANCES" \
  --normalization "$NORMALIZATION" \
  --reuse-base-root "$REUSE_BASE_ROOT" \
  --out "$OUT_ROOT" \
  --instance-ids \
    train-S0-002 train-S0-005 train-S0-006 train-S0-007 train-S0-008 \
    train-S0-009 train-S0-010 train-S0-011 train-S0-012 train-S0-013 \
  --algorithm-seeds 777001 777002 777003 \
  --strategies \
    BASE \
    I_DIRECTED_BALANCED_P025 \
    I_DIRECTED_BALANCED_P050 \
    I_DIRECTED_BALANCED_P075 \
    I_DIRECTED_BALANCED \
  --max-workers 12 \
  --evaluation-budget 400 \
  --scenario-count 200 \
  --ccp-confidence 0.90 \
  --pop 20 \
  --gens 1000 \
  --device cpu \
  | tee "$OUT_ROOT/evaluation.log"

if ! jq -e '
  .completed == true
  and .instance_count == 10
  and .algorithm_seed_count == 3
  and .run_count == 150
  and ([.status_counts[]] | add) == 150
  and .scenario_exact == true
  and (.strategies | length) == 5
' "$OUT_ROOT/COMPLETE.json" >/dev/null; then
  echo "I BALANCED STRENGTH GRID FAILED" >&2
  exit 1
fi

touch "$OUT_ROOT/PIPELINE_COMPLETE"

echo "===== I BALANCED STRENGTH GRID COMPLETE ====="
cat "$OUT_ROOT/QUICK_RESULT.json"
