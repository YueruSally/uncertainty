#!/usr/bin/env bash
set -euo pipefail

INSTANCES=${INSTANCES:-transformer_ppo_runs/diagnostics/ccp_m_prescreen_scale_search_instances_v1}
NORMALIZATION=${NORMALIZATION:-/home/lunet/cokt2/uncertainty-transformer-ppo/expanda/transformer_ppo_runs/reward_normalization.json}
OUT_ROOT=${OUT_ROOT:-transformer_ppo_runs/diagnostics/ccp_c_k8_final_oracle_v1_3inst_10seed_b400}
MAX_WORKERS=${MAX_WORKERS:-2}

for REQUIRED in "$NORMALIZATION" \
  "$INSTANCES/prescreen-search-S0-001.json" \
  "$INSTANCES/prescreen-search-S0-002.json" \
  "$INSTANCES/prescreen-search-S0-003.json"
do
  if ! test -f "$REQUIRED"; then
    echo "MISSING REQUIRED INPUT: $REQUIRED" >&2
    exit 1
  fi
done

mkdir -p "$OUT_ROOT"

echo "===== TEST C-K8 COMPLETE CROSSOVER+MUTATION PAIRS ====="
python3 -u -m unittest \
  tests.test_c_k8_crossover \
  tests.test_crossover_rule \
  > "$OUT_ROOT/tests.log" 2>&1

echo "===== C-K8 FINAL-HV ORACLE / CCP100 / POP20 / B400 ====="
python3 -u experiments/run_c_k8_final_oracle_v1.py \
  --instances "$INSTANCES" \
  --normalization "$NORMALIZATION" \
  --out "$OUT_ROOT" \
  --workers "$MAX_WORKERS"

echo "===== C-K8 FINAL-HV SCREEN COMPLETE ====="
cat "$OUT_ROOT/RESULT_TABLE.csv"
