#!/usr/bin/env bash
set -euo pipefail

SOURCE_INSTANCES=${SOURCE_INSTANCES:-/home/lunet/cokt2/uncertainty-transformer-ppo/expanda/transformer_ppo_runs/instances}
NORMALIZATION=${NORMALIZATION:-/home/lunet/cokt2/uncertainty-transformer-ppo/expanda/transformer_ppo_runs/reward_normalization.json}
CHECKPOINT=${CHECKPOINT:-transformer_ppo_runs/diagnostics/v5r_joint_replace_smoke_b160_p20_seed9501/M/final_policy.pt}
FORMAL_ROOT=${FORMAL_ROOT:-transformer_ppo_runs/formal/ccp_im_directed_formal_v3_pop250_gen200_20newinst_5seed_s200_oos5000}

mkdir -p "$FORMAL_ROOT"

python3 -u -m unittest \
  tests.test_s0_final_test_instances \
  tests.test_im_ccp_formal_v1 \
  tests.test_initialization_rule \
  tests.test_m_late_replace_frequency \
  > "$FORMAL_ROOT/tests.log" 2>&1

python3 -u prepare_im_ccp_formal_v1.py \
  --source-instances "$SOURCE_INSTANCES" \
  --normalization "$NORMALIZATION" \
  --checkpoint "$CHECKPOINT" \
  --out "$FORMAL_ROOT" \
  | tee "$FORMAL_ROOT/prepare.log"

for SHARD in 0 1 2 3 4; do
  python3 -u run_im_ccp_formal_shard_v1.py \
    --plan "$FORMAL_ROOT/FORMAL_PLAN.json" \
    --out "$FORMAL_ROOT/shard_${SHARD}" \
    --shard "$SHARD" --workers 1 --device cpu --preflight-only
done

echo "FORMAL PLAN READY: $FORMAL_ROOT/FORMAL_PLAN.json"
