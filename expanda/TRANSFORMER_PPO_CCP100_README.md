# Transformer-PPO CCP100 mutation-location study

This branch adds an isolated implementation for the next CCP100 study.  It
does not modify the frozen V1/V2 NSGA-II, mutation, repair, CCP100 evaluation,
or environmental-selection functions.  The new controller uses the existing
`ACTIVE_MUTATION_LOGGER` hook.

Do not run these commands on macOS.  Use the remote VS Code terminal on
`cokt2@lnx-grid-41`.  Every smoke, training and formal run must use a fresh
output directory; every long-running stage must use `nohup`.

## Safety and scientific invariants

- The five operators remain equiprobable.  PPO chooses only an eligible
  location after the operator has been sampled.
- A no-eligible event does not resample the operator and does not create a PPO
  action/reward tuple.
- The external nondominated archive is reward-only.  It is never passed to
  NSGA-II environmental selection.
- A post-crossover child is evaluated and inserted into the reward archive
  before the action.  Crossover improvement therefore cannot be credited to
  mutation-location PPO.
- OOS-5000 scenarios are used only after optimization and never for training,
  reward normalization or checkpoint selection.
- Checkpoints are selected by a predeclared training budget, not by test/OOS
  performance.

## 1. Synchronize in a separate server worktree

After this branch has been pushed:

```bash
cd /home/lunet/cokt2/uncertainty
git fetch origin --prune
git worktree add -b codex/transformer-ppo-ccp100 \
  /home/lunet/cokt2/uncertainty-transformer-ppo \
  origin/codex/transformer-ppo-ccp100
cd /home/lunet/cokt2/uncertainty-transformer-ppo
git status --short --branch
git rev-parse HEAD
```

If the local branch already exists, omit `-b` and use the local branch name as
the final argument.

## 2. Audit the existing server environment

Do not install or upgrade anything before recording this output:

```bash
python3 --version
python3 -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
nvidia-smi
```

The inherited code requires Python 3.10 or newer.  The exact PyTorch build must
match the server driver/CUDA environment; `requirements_transformer_ppo.txt`
intentionally does not pin a speculative CUDA wheel.

## 3. Unit tests

```bash
cd /home/lunet/cokt2/uncertainty-transformer-ppo/expanda
python3 -m unittest \
  tests.test_multi_instance_ccp100 \
  tests.test_learning_ccp100_logging \
  tests.test_learning_policy \
  tests.test_transformer_ppo_framework
```

## 4. Freeze release-time design before generating data

The source workbook has ET=0 for all 40 batches.  The generator therefore has
no hidden/default release-time distribution.  The supervisor-approved maximum
and grid step must be supplied explicitly:

```bash
mkdir -p expanda/transformer_ppo_runs/logs
nohup python3 -u expanda/generate_transformer_ppo_instances.py \
  --out expanda/transformer_ppo_runs/instances \
  --release-time-max-h APPROVED_MAX \
  --release-time-step-h APPROVED_STEP \
  > expanda/transformer_ppo_runs/logs/instances.log 2>&1 < /dev/null &
```

This produces 450 S0-S8 training manifests and 120 S0-S11 formal-test
manifests.  Train/test OD pools, centi-TEU grids and release-time slots are
disjoint.  Each manifest freezes instance, path, CCP100 and OOS seeds.

## 5. Reward normalization

Run predeclared training-only Rule pilots, then freeze normalized-objective
bounds.  OOS results and test instances are rejected:

```bash
nohup python3 -u expanda/run_transformer_ppo_rule_normalization_batch.py \
  --instances expanda/transformer_ppo_runs/instances \
  --catalog-digest FROZEN_CATALOG_DIGEST \
  --evaluation-budget PREDECLARED_SHORT_EPISODE_BUDGET \
  --workers 4 \
  --out expanda/transformer_ppo_runs/rule_normalization \
  > expanda/transformer_ppo_runs/logs/rule_normalization_batch.log 2>&1 < /dev/null &
```

The batch runner gives every instance fixed algorithm and policy seeds, limits
each worker to one numerical-library thread, keeps a separate log per instance,
and safely resumes runs that already have a validated `COMPLETE.json`.

```bash
python3 expanda/freeze_transformer_ppo_normalization.py \
  --runs expanda/transformer_ppo_runs/rule_normalization/runs \
  --instances expanda/transformer_ppo_runs/instances \
  --out expanda/transformer_ppo_runs/reward_normalization.json
```

The normalized objectives are cost/total TEU, emissions/total TEU and
makespan/instance time horizon.

## 6. Training

Do not launch formal training until server environment, release-time design,
normalization and smoke outputs have been inspected.  One policy seed is:

```bash
nohup python3 -u expanda/train_transformer_ppo_ccp100.py \
  --instances expanda/transformer_ppo_runs/instances \
  --normalization expanda/transformer_ppo_runs/reward_normalization.json \
  --architecture transformer --policy-seed 1 \
  --evaluation-budget PREDECLARED_SHORT_EPISODE_BUDGET \
  --out expanda/transformer_ppo_runs/training/transformer_seed1 \
  > expanda/transformer_ppo_runs/logs/transformer_seed1.log 2>&1 < /dev/null &
```

Repeat with three predeclared policy seeds and with `--architecture mlp` for
the structural ablation.  The formal inference entry point is
`run_ppo_policy_ccp100.py`; it loads a checkpoint with `training=False`.

Training commits an atomic checkpoint and `training_progress.json` after every
instance.  If a process or server is interrupted, rerun the identical command
with `--resume`.  All scientific arguments must remain identical; the loader
verifies the frozen configuration, completed episode markers and checkpoint
SHA-256 before continuing.  A partly written next episode is moved to
`interrupted_episodes/` rather than overwritten:

```bash
nohup python3 -u expanda/train_transformer_ppo_ccp100.py \
  --instances expanda/transformer_ppo_runs/instances \
  --normalization expanda/transformer_ppo_runs/reward_normalization.json \
  --architecture transformer --policy-seed 1 \
  --evaluation-budget PREDECLARED_SHORT_EPISODE_BUDGET \
  --out expanda/transformer_ppo_runs/training/transformer_seed1 \
  --resume \
  >> expanda/transformer_ppo_runs/logs/transformer_seed1.log 2>&1 < /dev/null &
```

Treat each policy seed as its own predeclared paired replication block; do not
select the best seed using test or OOS results.  Within a block, use the same
algorithm seed for all four methods.  The Random and Rule wrapper automatically
uses the instance's frozen path and CCP100 seeds:

```bash
python3 expanda/run_transformer_ppo_control_ccp100.py \
  --instance TEST_INSTANCE.json --policy rule --algorithm-seed PAIRED_SEED \
  --out FRESH_RULE_OUTPUT
```

## 7. OOS evaluation and paired inference

The OOS wrapper refuses training manifests and automatically uses the test
manifest's frozen 5,000-scenario seed.  Run it once for each frozen method run:

```bash
python3 expanda/run_transformer_ppo_oos.py \
  --instance TEST_INSTANCE.json --run METHOD_RUN --out FRESH_OOS_OUTPUT
```

After all four methods for an instance finish, compute their common normalized
OOS hypervolumes:

```bash
python3 expanda/compute_transformer_ppo_oos_hv.py \
  --instance TEST_INSTANCE.json \
  --normalization expanda/transformer_ppo_runs/reward_normalization.json \
  --oos random=RANDOM_OOS/oos_5000.json \
  --oos rule=RULE_OOS/oos_5000.json \
  --oos transformer=TRANSFORMER_OOS/oos_5000.json \
  --oos mlp=MLP_OOS/oos_5000.json \
  --out INSTANCE_RESULT/oos_hv.json
```

Once one replication block contains exactly ten new test instances for every
S0-S11 configuration, run the per-configuration paired Wilcoxon tests and Holm
correction:

```bash
python3 expanda/analyze_transformer_ppo_formal.py \
  --input-root REPLICATION_BLOCK_RESULTS \
  --out REPLICATION_BLOCK_RESULTS/formal_statistics.json
```
