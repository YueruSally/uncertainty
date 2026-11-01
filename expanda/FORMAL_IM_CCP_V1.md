# Formal CCP evaluation of directed I and M

This protocol is frozen before execution. It compares three methods only:

- `BASE`
- `I_DIRECTED_BALANCED` (`RULE-I`, `joint_balanced`)
- `M_DIRECTED_BALANCED` (probability-zero model controller plus scheduled
  `replace`, `joint_balanced`, from generation 100)

The design uses 20 newly generated and previously unseen S0 instances, five
new algorithm seeds, population 250, 200 generations, and exactly 50,250 real
objective evaluations per run. Optimization uses 200 frozen CCP scenarios at
q90. Fixed final fronts are then re-evaluated on 5,000 new scenarios per
instance. Inference averages the five seeds inside each instance and treats
the 20 instances as the independent units.

## Expected time

- Preparation and preflight: about 2--5 minutes.
- One full-budget paired smoke triplet: about 30--60 minutes.
- Five optimization shards, with two workers each: about 9--15 hours total
  wall time on the current server.
- Independent OOS evaluation and final analysis: about 30--60 minutes.

## 1. Prepare once

```bash
cd /home/lunet/cokt2/uncertainty-learning-components/expanda

export FORMAL_ROOT=transformer_ppo_runs/formal/ccp_im_directed_formal_v3_pop250_gen200_20newinst_5seed_s200_oos5000

bash run_im_ccp_formal_prepare_v1.sh
```

Preparation generates and freezes the unseen instances, hashes all scientific
inputs and implementations, and writes `FORMAL_PLAN.json`. Do not edit the
scientific code, normalization, checkpoint, instances, or plan after this.

## 2. Recommended one-command execution

The master runner prepares and freezes the plan, runs the full-budget smoke,
resumes those three runs into the 300-run optimization, and finally performs
the independent OOS evaluation:

```bash
nohup bash run_im_ccp_formal_full_v1.sh \
  > "$FORMAL_ROOT/full_pipeline.nohup.log" 2>&1 &
```

## 3. Manual full-budget smoke before the 300-run experiment

Run one exactly paired Base/I/M triplet first. These are real formal runs at
population 250, 200 generations and 50,250 evaluations, not a reduced-budget
substitute. Expected wall time with three workers is about 30--60 minutes.
The completed triplet is retained and reused by shard 0.

```bash
nohup python3 -u run_im_ccp_formal_shard_v1.py \
  --plan "$FORMAL_ROOT/FORMAL_PLAN.json" \
  --out "$FORMAL_ROOT/shard_0" --shard 0 --workers 3 --device cpu \
  --smoke-only \
  > "$FORMAL_ROOT/smoke.nohup.log" 2>&1 &
```

The only success signal is:

```bash
test -f "$FORMAL_ROOT/shard_0/SMOKE_COMPLETE.json" && \
  echo "FORMAL I/M SMOKE COMPLETE"
```

Do not start the full shards if the process has stopped without that file.

## 4. Manual five-shard execution

Shard 0:

```bash
nohup python3 -u run_im_ccp_formal_shard_v1.py \
  --plan "$FORMAL_ROOT/FORMAL_PLAN.json" \
  --out "$FORMAL_ROOT/shard_0" --shard 0 --workers 2 --device cpu \
  > "$FORMAL_ROOT/shard_0.nohup.log" 2>&1 &
```

Shard 1:

```bash
nohup python3 -u run_im_ccp_formal_shard_v1.py \
  --plan "$FORMAL_ROOT/FORMAL_PLAN.json" \
  --out "$FORMAL_ROOT/shard_1" --shard 1 --workers 2 --device cpu \
  > "$FORMAL_ROOT/shard_1.nohup.log" 2>&1 &
```

Shard 2:

```bash
nohup python3 -u run_im_ccp_formal_shard_v1.py \
  --plan "$FORMAL_ROOT/FORMAL_PLAN.json" \
  --out "$FORMAL_ROOT/shard_2" --shard 2 --workers 2 --device cpu \
  > "$FORMAL_ROOT/shard_2.nohup.log" 2>&1 &
```

Run matching commands for shards 3 and 4. The recommended master runner does
this automatically and is less error-prone.

Each shard is resumable. Re-running the same command validates and reuses every
complete task; an interrupted task is quarantined before being rerun.

Shard 0 automatically reuses the three completed smoke runs.

## 5. Monitor

```bash
for SHARD in 0 1 2 3 4; do
  echo "===== SHARD $SHARD ====="
  jq '{completed_task_count,task_count,estimated_remaining_hours,last_task}' \
    "$FORMAL_ROOT/shard_${SHARD}/shard_progress.json" 2>/dev/null \
    || tail -n 5 "$FORMAL_ROOT/shard_${SHARD}.nohup.log"
done
```

Do not run final analysis until all five `SHARD_COMPLETE.json` files exist.

## 6. Independent OOS evaluation and formal analysis

```bash
export MAX_WORKERS=4
nohup bash run_im_ccp_formal_finalize_v1.sh \
  > "$FORMAL_ROOT/finalize.nohup.log" 2>&1 &
```

The final concise result is written to
`$FORMAL_ROOT/analysis/QUICK_RESULT.json`; the complete statistical result and
report are `FORMAL_RESULT.json` and `REPORT.md`. For the short table containing
only average HV change and average runtime, use the directly generated
`AVERAGE_RESULT.json`.

This protocol evaluates deterministic teacher rules. It does not claim that a
Transformer or PPO policy learned these improvements.
