#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${1:-$ROOT/expanda/learning_runs/segment_v2_offline_719a54d/models/segment_policy_model.joblib}"
OUT="${2:-$ROOT/expanda/learning_runs/segment_v2_pilot_3pairs}"
DATA="${3:-$ROOT/expanda/data/data_expanded.xlsx}"

if [[ ! -f "$MODEL" ]]; then
    echo "Missing promoted segment model: $MODEL" >&2
    exit 2
fi
if [[ -d "$OUT" ]] && [[ -n "$(find "$OUT" -mindepth 1 -print -quit)" ]]; then
    echo "Refusing to overwrite non-empty pilot directory: $OUT" >&2
    exit 2
fi
mkdir -p "$OUT"

for run in 1 2 3; do
    algorithm_seed=$((990000 + run))
    training_seed=$((991000 + run))
    policy_seed=$((10990000 + run))
    run_dir=$(printf "run%02d" "$run")
    echo "[$(date -Is)] $run_dir: Random"
    python "$ROOT/expanda/run_learning_ccp100.py" \
        --data "$DATA" --out "$OUT/$run_dir/random" \
        --pop 100 --gens 1000 --evaluation-budget 15000 \
        --algorithm-seed "$algorithm_seed" --training-seed "$training_seed" \
        --path-seed 0 --policy-seed "$policy_seed" --policy random
    echo "[$(date -Is)] $run_dir: Rule"
    python "$ROOT/expanda/run_learning_ccp100.py" \
        --data "$DATA" --out "$OUT/$run_dir/rule" \
        --pop 100 --gens 1000 --evaluation-budget 15000 \
        --algorithm-seed "$algorithm_seed" --training-seed "$training_seed" \
        --path-seed 0 --policy-seed "$policy_seed" --policy rule --epsilon .10
    echo "[$(date -Is)] $run_dir: LearningSegment"
    python "$ROOT/expanda/run_learning_ccp100.py" \
        --data "$DATA" --out "$OUT/$run_dir/learning_segment" \
        --pop 100 --gens 1000 --evaluation-budget 15000 \
        --algorithm-seed "$algorithm_seed" --training-seed "$training_seed" \
        --path-seed 0 --policy-seed "$policy_seed" --policy learning-segment \
        --model "$MODEL" --epsilon .10
done

python - "$OUT" <<'PY'
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
rows = []
for run in range(1, 4):
    for method in ("random", "rule", "learning_segment"):
        path = root / f"run{run:02d}" / method / "COMPLETE.json"
        row = json.loads(path.read_text())
        if not row["completed"] or row["stop_reason"] != "evaluation_budget":
            raise SystemExit(f"Incomplete budget-controlled run: {path}")
        rows.append({"run": run, "method": method,
                     "evaluations": row["ccp_evaluation_count"],
                     "effective": row["effective_modifications"],
                     "pareto": row["final_pareto_size"],
                     "seconds": row["optimisation_seconds"]})
(root / "PILOT_COMPLETE.json").write_text(
    json.dumps({"completed": True, "oos_used": False, "runs": rows}, indent=2) + "\n")
print("SEGMENT V2 THREE-PAIR PILOT COMPLETE")
PY
