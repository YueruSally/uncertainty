# RF-Reward C+ training-only evidence gate

This run is a medium-scale gate between stages C and D.  It is not formal RF
training or testing and reads no formal-test outcome or OOS hypervolume.

## Frozen scope

- Source: the existing 450-manifest S0-S8 training catalog only.
- Selection: five manifests per configuration, selected deterministically with
  seed `82000000`; 45 instances per trajectory.
- Collection: three independent trajectories, 1,000 actual CCP100 evaluations
  per instance, population 100, `rho=0.5` Rule/uniform eligible mixture.
- Upper bound: 135,000 actual CCP100 evaluations.
- Fitting: unweighted immediate-reward regression.  Each trajectory produces
  one independent RF model.
- Validation: five configuration-balanced, instance-grouped folds.  Every fold
  holds out one instance from each S0-S8 configuration.
- Gate report: RF versus the corresponding training-fold mean predictor on
  instance-macro MAE and RMSE, separately for all three trajectories.

The batch validates the catalog, source workbook and normalization hashes before
starting.  Completed instances are skipped only when their instance, seeds,
budget, collection mixture and normalization provenance all match.  A non-empty
incomplete run is preserved and causes a stop for inspection.

## Server launch

```bash
cd /home/lunet/cokt2/uncertainty-transformer-ppo
conda activate ccp100ppo

export CPLUS_INSTANCES="$PWD/expanda/transformer_ppo_runs/instances"
export CPLUS_NORMALIZATION="$PWD/expanda/transformer_ppo_runs/reward_normalization.json"
export CPLUS_OUT="$PWD/expanda/transformer_ppo_runs/rf_reward_cplus_v1"
export CPLUS_LOG="$PWD/expanda/transformer_ppo_runs/logs/rf_reward_cplus_v1.log"
export CPLUS_CATALOG_DIGEST="$(python -c 'import json,os; print(json.load(open(os.path.join(os.environ["CPLUS_INSTANCES"], "catalog.json")))["catalog_digest"])')"
export CPLUS_NORMALIZATION_DIGEST="$(sha256sum "$CPLUS_NORMALIZATION" | cut -d ' ' -f1)"

mkdir -p "$(dirname "$CPLUS_LOG")"

nohup python -u expanda/run_rf_reward_cplus_batch.py \
  --instances "$CPLUS_INSTANCES" \
  --normalization "$CPLUS_NORMALIZATION" \
  --out "$CPLUS_OUT" \
  --catalog-digest "$CPLUS_CATALOG_DIGEST" \
  --normalization-digest "$CPLUS_NORMALIZATION_DIGEST" \
  --instances-per-config 5 \
  --trajectories 3 \
  --evaluation-budget 1000 \
  --uniform-mixture 0.5 \
  --workers 4 \
  --rf-jobs 4 \
  --pop 100 \
  --gens 1000 \
  >> "$CPLUS_LOG" 2>&1 < /dev/null &

echo $! > "$CPLUS_OUT.pid"
echo "PID=$(cat "$CPLUS_OUT.pid")"
echo "LOG=$CPLUS_LOG"
```

Monitor with:

```bash
tail -f "$CPLUS_LOG"
```

The exact same launch command safely resumes matching completed collections.
Final evidence is written to `BATCH_COMPLETE.json`; each trajectory also has a
`TRAJECTORY_COMPLETE.json`, model, training summary and per-instance logs.

Passing or failing this development gate does not authorize stages D or E.
Results must be reviewed before changing the RF definition or launching formal
training.
