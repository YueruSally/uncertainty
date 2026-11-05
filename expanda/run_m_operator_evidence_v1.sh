#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
PYTHON=${PYTHON:-python3}
COLLECTION_ROOT=${COLLECTION_ROOT:-$PWD/transformer_ppo_runs/diagnostics/ccp_m_operator_feedback_v2_6inst_2seed_b400}
OFFLINE_ROOT=${OFFLINE_ROOT:-$PWD/transformer_ppo_runs/diagnostics/ccp_m_operator_offline_v1_384state}
EVIDENCE_ROOT=${EVIDENCE_ROOT:-$PWD/transformer_ppo_runs/diagnostics/ccp_m_operator_evidence_v1_384state}
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
mkdir -p "$EVIDENCE_ROOT"
export MPLCONFIGDIR="$EVIDENCE_ROOT/matplotlib_cache"
echo "===== TEST NOISE / GROUPED LEARNING CURVE / NO NEW EVALUATIONS ====="
if ! "$PYTHON" -u -m unittest tests.test_m_operator_evidence > "$EVIDENCE_ROOT/tests.log" 2>&1; then
  cat "$EVIDENCE_ROOT/tests.log"
  exit 1
fi
cat "$EVIDENCE_ROOT/tests.log"
"$PYTHON" -u run_m_operator_evidence_v1.py \
  --collection-root "$COLLECTION_ROOT" --offline-root "$OFFLINE_ROOT" --out "$EVIDENCE_ROOT"
echo "===== M OPERATOR EVIDENCE COMPLETE ====="
