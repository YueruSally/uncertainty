#!/usr/bin/env bash
set -euo pipefail

NORMALIZATION=${NORMALIZATION:-/home/lunet/cokt2/uncertainty-transformer-ppo/expanda/transformer_ppo_runs/reward_normalization.json}
VALIDATION_INSTANCES=${VALIDATION_INSTANCES:-transformer_ppo_runs/diagnostics/ccp_m_ccp100_independent_v1_instances_5}
VALIDATION_ROOT=${VALIDATION_ROOT:-transformer_ppo_runs/diagnostics/ccp_m_ccp100_independent_v1_5inst_3seed_b800}
OUT_ROOT=${OUT_ROOT:-transformer_ppo_runs/diagnostics/ccp_m_ccp100_independent_oos_v1_5inst_3seed_s1000}
MAX_WORKERS=${MAX_WORKERS:-2}
MAX_ATTEMPTS=${MAX_ATTEMPTS:-3}

if ! [[ "$MAX_WORKERS" =~ ^[1-9][0-9]*$ ]] \
    || (( MAX_WORKERS > 2 )); then
  echo "MAX_WORKERS must be an integer in [1, 2]: $MAX_WORKERS" >&2
  exit 1
fi
if ! [[ "$MAX_ATTEMPTS" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_ATTEMPTS must be a positive integer: $MAX_ATTEMPTS" >&2
  exit 1
fi
for REQUIRED in \
  "$NORMALIZATION" \
  "$VALIDATION_INSTANCES/catalog.json" \
  "$VALIDATION_ROOT/COMPLETE.json" \
  "$VALIDATION_ROOT/VALIDATION_PASS"; do
  if ! test -f "$REQUIRED"; then
    echo "MISSING REQUIRED INPUT: $REQUIRED" >&2
    exit 1
  fi
done

mkdir -p "$OUT_ROOT"
exec 9>"$OUT_ROOT/pipeline.lock"
if ! flock -n 9; then
  echo "ANOTHER THREE-METHOD OOS PIPELINE IS ALREADY RUNNING" >&2
  exit 75
fi

echo "===== TEST THREE-METHOD INDEPENDENT OOS EVALUATOR ====="
python3 -u -m unittest \
  tests.test_m_directed_balanced_oos \
  tests.test_m_three_method_oos \
  > "$OUT_ROOT/tests.log" 2>&1

echo "===== THREE-METHOD / ONE-SCENARIO-SET / ONE-REFERENCE OOS ====="
if ! test -f "$OUT_ROOT/OOS_COMPLETE.json"; then
  ATTEMPT=1
  while (( ATTEMPT <= MAX_ATTEMPTS )); do
    echo "[M-THREE-OOS] attempt=$ATTEMPT/$MAX_ATTEMPTS"
    set +e
    python3 -u evaluate_m_three_method_oos.py \
      --instances "$VALIDATION_INSTANCES" \
      --normalization "$NORMALIZATION" \
      --validation-root "$VALIDATION_ROOT" \
      --out "$OUT_ROOT" \
      --instance-ids \
        final-test-S0-001 final-test-S0-002 final-test-S0-003 \
        final-test-S0-004 final-test-S0-005 \
      --algorithm-seeds 886001 886002 886003 \
      --scenario-count 1000 \
      --oos-seed 77000000 \
      --alpha 0.90 \
      --uncertainty-profile transport_border_saa_v1 \
      --max-workers "$MAX_WORKERS" \
      2>&1 | tee -a "$OUT_ROOT/evaluation.log"
    STATUS=${PIPESTATUS[0]}
    set -e
    if (( STATUS == 0 )) && test -f "$OUT_ROOT/OOS_COMPLETE.json"; then
      break
    fi
    if (( ATTEMPT == MAX_ATTEMPTS )); then
      echo "OOS EVALUATION FAILED AFTER $MAX_ATTEMPTS ATTEMPTS" >&2
      if (( STATUS == 0 )); then
        STATUS=1
      fi
      exit "$STATUS"
    fi
    ATTEMPT=$((ATTEMPT + 1))
  done
fi

touch "$OUT_ROOT/OOS_PIPELINE_COMPLETE"
if jq -e '.admission_pass == true' "$OUT_ROOT/OOS_COMPLETE.json" >/dev/null; then
  touch "$OUT_ROOT/OOS_PASS"
  echo "===== THREE-METHOD INDEPENDENT OOS PASSED ====="
else
  touch "$OUT_ROOT/OOS_REJECTED"
  echo "===== THREE-METHOD INDEPENDENT OOS DID NOT PASS ====="
fi
cat "$OUT_ROOT/QUICK_RESULT.json"
