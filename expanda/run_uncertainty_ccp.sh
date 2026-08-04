#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_EXECUTABLE="$PYTHON_BIN"
elif [[ -x "../.venv/bin/python" ]]; then
  PYTHON_EXECUTABLE="../.venv/bin/python"
elif [[ -x ".venv/bin/python" ]]; then
  PYTHON_EXECUTABLE=".venv/bin/python"
else
  PYTHON_EXECUTABLE="python3"
fi

exec "$PYTHON_EXECUTABLE" baseline_uncertainty.py \
  --pop "${POP_SIZE:-200}" \
  --gens "${GENERATIONS:-300}" \
  --runs "${RUNS:-1}" \
  --mc-scenarios "${MC_SCENARIOS:-200}" \
  --out "${OUTPUT_DIR:-outputs/uncertainty_ccp}" \
  "$@"
