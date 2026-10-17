# RF-Reward formal test and five-method analysis (stage E)

This stage is frozen after the successful stage-D audit and before reading any
five-method formal hypervolume.  It has two separated entry points:

1. `run_rf_reward_formal_test_batch.py` audits the three completed four-method
   blocks without loading their legacy HV values, then performs only the 360
   paired RF optimizations and their OOS-5000 evaluations.
2. `analyze_rf_reward_five_method_formal.py` refuses to run until all 360 RF
   tasks are complete.  It then calculates new additive five-method metrics.

Neither entry point modifies the existing Random, Rule, Transformer or MLP
directories.  Their previous four-method HV files are provenance-checked by
hash but ignored because they used a different normalization definition.

## Frozen formal protocol

- 120 test instances, 10 per S0-S11 configuration.
- Three paired blocks; RF trajectory/model 1, 2 and 3 map to blocks 1, 2 and 3.
- 15,000 actual CCP100 evaluations per optimization, population 100.
- 360 RF optimizations and at most 5,400,000 CCP100 evaluations.
- Algorithm, path, CCP100 and OOS seeds are copied from the corresponding
  completed four-method block and verified from its metadata.
- Final decisions are the feasible first nondominated front of the final
  population. OOS uses the manifest's common 5,000-scenario master.
- RF prediction uses one CPU job per optimization worker. This changes only
  inference parallelism, not the frozen trees or action argmax.
- No partial HV is calculated during the 360 RF runs.

The analysis uses empirical OOS q90 cost, emission and makespan. For each test
instance it pools all local OOS nondominated fronts from five methods and three
blocks, extracts their global nondominated set, and uses its componentwise
minimum and maximum as the one common normalization range. The formula is
`(objective-minimum)/scale`, without clipping. An exactly zero span uses scale
one, so every value in that coordinate maps to zero. The primary normalized HV
reference is `(1.1,1.1,1.1)`; `(1.05,1.05,1.05)` and `(1.2,1.2,1.2)` are
descriptive sensitivities.

The six primary hypotheses are RF–Rule, Transformer–Rule and Transformer–RF in
S0-S8 and S9-S11. Each test-instance difference is the mean of its three paired
block differences. Mean differences receive a 20,000-replicate percentile
bootstrap CI that resamples instances within each fixed configuration. A
two-sided 100,000-draw sign-flip randomization test targets the mean paired
difference, and Holm adjusts the family of six p-values. Win/tie/loss uses a
`1e-12` tie tolerance. Random and MLP remain in the five-method descriptive
table; per-block/model-seed performance is retained.

Supporting process metrics use all mutation attempts, including no-candidate
skips, as the common denominator. Effective modification, no-candidate,
no-change, mutation-failure, repair-failure and rollback categories are shown
explicitly and may overlap. Optimization time, inference time, evaluation
counts and stage-D training cost are reported separately and never replace the
primary OOS-HV endpoint.

## Locate the three existing blocks

After updating the server, list only completion metadata; do not open any old
HV or statistics file:

```bash
cd /home/lunet/cokt2/uncertainty-transformer-ppo
conda activate ccp100ppo

find "$PWD/expanda/transformer_ppo_runs" -name BATCH_COMPLETE.json -print |
while read -r file; do
  python -c 'import json,sys; p=sys.argv[1]; d=json.load(open(p)); print(p, "formal=", d.get("formal"), "replication=", d.get("replication"), "instances=", d.get("completed_instance_count"))' "$file"
done
```

Set the three roots containing those verified completion files:

```bash
export RFE_ROOT="$PWD"
export RFE_PYTHON="$(command -v python)"
export RFE_INSTANCES="$PWD/expanda/transformer_ppo_runs/instances"
export RFE_NORMALIZATION="$PWD/expanda/transformer_ppo_runs/reward_normalization.json"
export RFE_TRAINING="$PWD/expanda/transformer_ppo_runs/rf_reward_formal_training_v1"
export RFE_BLOCK1="ABSOLUTE_ROOT_FOR_EXISTING_REPLICATION_1"
export RFE_BLOCK2="ABSOLUTE_ROOT_FOR_EXISTING_REPLICATION_2"
export RFE_BLOCK3="ABSOLUTE_ROOT_FOR_EXISTING_REPLICATION_3"
export RFE_OUT="$PWD/expanda/transformer_ppo_runs/rf_reward_formal_test_v1"
export RFE_LOG="$PWD/expanda/transformer_ppo_runs/logs/rf_reward_formal_test_v1.log"
```

## Metadata-only preflight

```bash
"$RFE_PYTHON" -u expanda/run_rf_reward_formal_test_batch.py \
  --instances "$RFE_INSTANCES" \
  --normalization "$RFE_NORMALIZATION" \
  --rf-training-root "$RFE_TRAINING" \
  --existing-block "1=$RFE_BLOCK1" \
  --existing-block "2=$RFE_BLOCK2" \
  --existing-block "3=$RFE_BLOCK3" \
  --out "$RFE_OUT" \
  --workers 4 \
  --preflight-only
```

Preflight must report 1,440 existing runs, 1,440 existing OOS results, three
expected RF model hashes, 360 RF optimizations, 360 RF OOS evaluations, a
5,400,000 evaluation upper bound, and `formal_hv_values_read: false`. If any
old run, checkpoint, source front, OOS file, seed or digest differs, it stops.

## Launch/resume RF formal testing

Use a new output directory. The process never overwrites a partial directory;
an interrupted non-empty task must be inspected before resumption.

```bash
mkdir -p "$(dirname "$RFE_LOG")"

nohup "$RFE_PYTHON" -u expanda/run_rf_reward_formal_test_batch.py \
  --instances "$RFE_INSTANCES" \
  --normalization "$RFE_NORMALIZATION" \
  --rf-training-root "$RFE_TRAINING" \
  --existing-block "1=$RFE_BLOCK1" \
  --existing-block "2=$RFE_BLOCK2" \
  --existing-block "3=$RFE_BLOCK3" \
  --out "$RFE_OUT" \
  --workers 4 \
  >> "$RFE_LOG" 2>&1 < /dev/null &

echo $! > "$RFE_OUT.pid"
echo "PID=$(cat "$RFE_OUT.pid")"
echo "LOG=$RFE_LOG"
```

Completion is represented by `RF_FORMAL_COMPLETE.json`. It must state 360
completed tasks and `hv_computed: false`.

## Five-method analysis — only after completion audit

Do not run this command while the formal batch is active or incomplete:

```bash
export RFE_ANALYSIS="$PWD/expanda/transformer_ppo_runs/rf_reward_five_method_analysis_v1"

"$RFE_PYTHON" -u expanda/analyze_rf_reward_five_method_formal.py \
  --instances "$RFE_INSTANCES" \
  --normalization "$RFE_NORMALIZATION" \
  --rf-training-root "$RFE_TRAINING" \
  --existing-block "1=$RFE_BLOCK1" \
  --existing-block "2=$RFE_BLOCK2" \
  --existing-block "3=$RFE_BLOCK3" \
  --rf-formal-root "$RFE_OUT" \
  --out "$RFE_ANALYSIS"
```

All outputs are new files. `raw_oos_source_manifest.json` preserves hashes and
paths for every raw OOS JSON/NPZ source; `per_instance_oos_bounds.json` freezes
the 120 common bounds; `five_method_oos_hv_rows.json` contains primary and
sensitivity metrics; `formal_statistics.json` contains the paired analyses.
`auxiliary_process_metrics.json` contains consistently denominated process and
cost results.
`ANALYSIS_COMPLETE.json` records all output hashes.
