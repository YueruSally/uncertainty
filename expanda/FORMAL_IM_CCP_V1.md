# Formal CCP evaluation of directed I and M

This protocol is frozen before execution. It compares three methods only:

- `BASE`
- `I_DIRECTED_BALANCED` (`RULE-I`, `joint_balanced`)
- `M_DIRECTED_BALANCED` (probability-zero model controller plus scheduled
  `replace`, `joint_balanced`, from generation 100)

The design uses 10 newly generated and previously unseen S0 instances, three
new algorithm seeds, population 250, 200 generations, and exactly 50,250 real
objective evaluations per run. Optimization uses 200 frozen CCP scenarios at
q90. Fixed final fronts are then re-evaluated on 5,000 new scenarios per
instance. Inference averages the three seeds inside each instance and treats
the 10 instances as the independent units.

## Expected time

- Preparation and preflight: about 2--5 minutes.
- Three optimization shards, with two workers each: about 5--7 hours total
  wall time on the current server.
- Independent OOS evaluation and final analysis: about 20--40 minutes.

## 1. Prepare once

```bash
cd /home/lunet/cokt2/uncertainty-learning-components/expanda

export FORMAL_ROOT=transformer_ppo_runs/formal/ccp_im_directed_formal_v2_pop250_gen200_10newinst_3seed_s200_oos5000

bash run_im_ccp_formal_prepare_v1.sh
```

Preparation generates and freezes the unseen instances, hashes all scientific
inputs and implementations, and writes `FORMAL_PLAN.json`. Do not edit the
scientific code, normalization, checkpoint, instances, or plan after this.

## 2. Full-budget smoke before the 90-run experiment

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

## 3. Run three terminals concurrently

Terminal 1:

```bash
nohup python3 -u run_im_ccp_formal_shard_v1.py \
  --plan "$FORMAL_ROOT/FORMAL_PLAN.json" \
  --out "$FORMAL_ROOT/shard_0" --shard 0 --workers 2 --device cpu \
  > "$FORMAL_ROOT/shard_0.nohup.log" 2>&1 &
```

Terminal 2:

```bash
nohup python3 -u run_im_ccp_formal_shard_v1.py \
  --plan "$FORMAL_ROOT/FORMAL_PLAN.json" \
  --out "$FORMAL_ROOT/shard_1" --shard 1 --workers 2 --device cpu \
  > "$FORMAL_ROOT/shard_1.nohup.log" 2>&1 &
```

Terminal 3:

```bash
nohup python3 -u run_im_ccp_formal_shard_v1.py \
  --plan "$FORMAL_ROOT/FORMAL_PLAN.json" \
  --out "$FORMAL_ROOT/shard_2" --shard 2 --workers 2 --device cpu \
  > "$FORMAL_ROOT/shard_2.nohup.log" 2>&1 &
```

Each shard is resumable. Re-running the same command validates and reuses every
complete task; an interrupted task is quarantined before being rerun.

Shard 0 automatically reuses the three completed smoke runs.

## 4. Monitor

```bash
for SHARD in 0 1 2; do
  echo "===== SHARD $SHARD ====="
  jq '{completed_task_count,task_count,estimated_remaining_hours,last_task}' \
    "$FORMAL_ROOT/shard_${SHARD}/shard_progress.json" 2>/dev/null \
    || tail -n 5 "$FORMAL_ROOT/shard_${SHARD}.nohup.log"
done
```

Do not run final analysis until all three `SHARD_COMPLETE.json` files exist.

## 5. Independent OOS evaluation and formal analysis

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
