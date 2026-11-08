#!/usr/bin/env bash
set -euo pipefail
SCRIPT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec python3 -u "$SCRIPT_ROOT/run.py" --stage all --workers "${MAX_WORKERS:-2}" "$@"
