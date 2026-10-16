# RF-Reward formal training (stage D)

Stage D uses training instances only.  It performs no formal-test optimization,
OOS evaluation or hypervolume computation.

## Frozen design

- 450 S0-S8 training manifests: 50 per configuration.
- Three independent collection trajectories.
- At most 5,000 actual CCP100 evaluations per instance, population 100.
- 1,350 collection runs and at most 6,750,000 evaluations in total.
- Collection policy: `0.5 * p_rule + 0.5 * uniform_eligible`.
- Exact immediate archive reward and the C+ feature schema are unchanged.
- Primary fitting is unweighted.  Logged propensities are retained for overlap
  diagnostics and an optional, separately declared sensitivity analysis.
- RF parameters are frozen from C+ before stage D: 200 trees, depth 8, minimum
  leaf 5 and feature fraction 0.5.
- Five configuration-balanced, instance-grouped folds evaluate the fixed model
  diagnostically.  They do not select or alter its parameters.

The runner validates source, catalog and normalization hashes; rejects non-train
manifests; fixes independent collection/model seed namespaces; limits numerical
library threads per collection worker; and resumes only metadata-identical
completed runs.  It never overwrites a partial non-empty run.

## Preflight

```bash
cd /home/lunet/cokt2/uncertainty-transformer-ppo
conda activate ccp100ppo

export RFD_INSTANCES="$PWD/expanda/transformer_ppo_runs/instances"
export RFD_NORMALIZATION="$PWD/expanda/transformer_ppo_runs/reward_normalization.json"
export RFD_OUT="$PWD/expanda/transformer_ppo_runs/rf_reward_formal_training_v1"
export RFD_LOG="$PWD/expanda/transformer_ppo_runs/logs/rf_reward_formal_training_v1.log"
export RFD_CATALOG_DIGEST="$(python -c 'import json,os; print(json.load(open(os.path.join(os.environ["RFD_INSTANCES"], "catalog.json")))["catalog_digest"])')"
export RFD_NORMALIZATION_DIGEST="de73404449c2b612609ad5248f815793f99b676e4260e2068ae403f043addbcd"

printf '%s  %s\n' "$RFD_NORMALIZATION_DIGEST" "$RFD_NORMALIZATION" | sha256sum -c -

python -u expanda/run_rf_reward_formal_training_batch.py \
  --instances "$RFD_INSTANCES" \
  --normalization "$RFD_NORMALIZATION" \
  --out "$RFD_OUT" \
  --catalog-digest "$RFD_CATALOG_DIGEST" \
  --normalization-digest "$RFD_NORMALIZATION_DIGEST" \
  --workers 4 --rf-jobs 4 --preflight-only
```

Preflight must report 450 instances per trajectory, three trajectories, 1,350
collections, 5,000 evaluations per instance and a 6,750,000 upper bound.

Before launch, verify adequate disk space and that no unrelated CPU experiment
would contend with the four collection workers:

```bash
df -h /home/lunet/cokt2
pgrep -af 'python.*(rf_reward|transformer_ppo|ccp100)' || true
```

## Launch and resume

```bash
mkdir -p "$(dirname "$RFD_LOG")"

nohup python -u expanda/run_rf_reward_formal_training_batch.py \
  --instances "$RFD_INSTANCES" \
  --normalization "$RFD_NORMALIZATION" \
  --out "$RFD_OUT" \
  --catalog-digest "$RFD_CATALOG_DIGEST" \
  --normalization-digest "$RFD_NORMALIZATION_DIGEST" \
  --workers 4 --rf-jobs 4 \
  >> "$RFD_LOG" 2>&1 < /dev/null &

echo $! > "$RFD_OUT.pid"
echo "PID=$(cat "$RFD_OUT.pid")"
echo "LOG=$RFD_LOG"
```

The same command safely resumes completed collections and validated completed
models.  A failed or interrupted non-empty run stops for inspection instead of
being overwritten.

Monitor with:

```bash
tail -f "$RFD_LOG"
```

Successful completion creates `FORMAL_TRAINING_COMPLETE.json`, three
`FORMAL_TRAJECTORY_COMPLETE.json` files, three `rf_reward.joblib` models and all
source/model/data hashes.  Stage E must not begin until these artifacts are
audited.
