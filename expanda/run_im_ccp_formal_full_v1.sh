#!/usr/bin/env bash
set -euo pipefail

FORMAL_ROOT=${FORMAL_ROOT:-transformer_ppo_runs/formal/ccp_im_directed_formal_v3_pop250_gen200_20newinst_5seed_s200_oos5000}
SHARD_WORKERS=${SHARD_WORKERS:-2}
OOS_WORKERS=${OOS_WORKERS:-5}
export FORMAL_ROOT

mkdir -p "$FORMAL_ROOT"

echo "===== PREPARE AND FREEZE 20-INSTANCE / 5-SEED PLAN ====="
bash run_im_ccp_formal_prepare_v1.sh

echo "===== ONE FULL-BUDGET PAIRED BASE/I/M SMOKE ====="
python3 -u run_im_ccp_formal_shard_v1.py \
  --plan "$FORMAL_ROOT/FORMAL_PLAN.json" \
  --out "$FORMAL_ROOT/shard_0" \
  --shard 0 --workers 3 --device cpu --smoke-only \
  | tee "$FORMAL_ROOT/smoke.log"

test -f "$FORMAL_ROOT/shard_0/SMOKE_COMPLETE.json"

echo "===== 300-RUN FORMAL OPTIMIZATION: FIVE SHARDS ====="
PIDS=()
for SHARD in 0 1 2 3 4; do
  python3 -u run_im_ccp_formal_shard_v1.py \
    --plan "$FORMAL_ROOT/FORMAL_PLAN.json" \
    --out "$FORMAL_ROOT/shard_${SHARD}" \
    --shard "$SHARD" --workers "$SHARD_WORKERS" --device cpu \
    > "$FORMAL_ROOT/shard_${SHARD}.nohup.log" 2>&1 &
  PID=$!
  PIDS+=("$PID")
  echo "[FORMAL MASTER] shard=$SHARD pid=$PID"
done

FAILED=0
for INDEX in 0 1 2 3 4; do
  if ! wait "${PIDS[$INDEX]}"; then
    echo "[FORMAL MASTER] shard=$INDEX failed" >&2
    FAILED=1
  fi
done
if test "$FAILED" -ne 0; then
  echo "At least one formal shard failed; inspect shard logs." >&2
  exit 1
fi

for SHARD in 0 1 2 3 4; do
  test -f "$FORMAL_ROOT/shard_${SHARD}/SHARD_COMPLETE.json"
done

echo "===== 5000-SCENARIO INDEPENDENT OOS EVALUATION ====="
env MAX_WORKERS="$OOS_WORKERS" bash run_im_ccp_formal_finalize_v1.sh

test -f "$FORMAL_ROOT/analysis/PIPELINE_COMPLETE"
touch "$FORMAL_ROOT/FULL_PIPELINE_COMPLETE"

echo "===== FORMAL I/M EXPERIMENT COMPLETE ====="
cat "$FORMAL_ROOT/analysis/AVERAGE_RESULT.json"
