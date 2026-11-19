#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec python3 -B -u "$SCRIPT_DIR/run.py" "$@"
