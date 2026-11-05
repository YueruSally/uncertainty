#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
PYTHON=${PYTHON:-python3}
COLLECTION_ROOT=${COLLECTION_ROOT:-$PWD/transformer_ppo_runs/diagnostics/ccp_m_operator_feedback_v2_6inst_2seed_b400}
OFFLINE_ROOT=${OFFLINE_ROOT:-$PWD/transformer_ppo_runs/diagnostics/ccp_m_operator_offline_v1_384state}
PATH_ROOT=${PATH_ROOT:-$PWD/transformer_ppo_runs/diagnostics/ccp_m_operator_path_representation_v1_384state}
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
mkdir -p "$PATH_ROOT"
echo "===== TEST MATCHED SUMMARY / PATH TOKENS ====="
if ! "$PYTHON" -u -m unittest tests.test_m_operator_path_representation > "$PATH_ROOT/tests.log" 2>&1; then
  cat "$PATH_ROOT/tests.log"
  exit 1
fi
cat "$PATH_ROOT/tests.log"
echo "===== PAIRED REPRESENTATION STUDY / CCP100 DATA / ZERO NEW EVALUATIONS ====="
"$PYTHON" -u run_m_operator_path_representation_v1.py \
  --collection-root "$COLLECTION_ROOT" --offline-root "$OFFLINE_ROOT" --out "$PATH_ROOT"
echo "===== M PATH REPRESENTATION COMPLETE ====="
