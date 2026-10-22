# NSGA-II learning-components experiment

This is the implementation and execution record for the new five-method
study: `Base`, `I`, `C`, `M`, and frozen `I+C+M`.

## Frozen scientific contract

- Begin with one deterministic scene and disjoint training/test instances.
  There is no validation split, early stopping, or best-checkpoint selection.
- I preserves Base's representable initial-individual structure.  PPO chooses
  the identities of `k` distinct paths; Base still chooses `k`, generates the
  split shares, and normalizes them.
- C is called only for common-node crossover.  Its selected cut is followed by
  Base mutation, repair, and the original final-offspring evaluation.  There
  is no post-crossover intermediate objective evaluation.
- M samples each of the five operators with probability 0.2 and learns only
  the legal target position.
- One action reward is the joint archive increment
  `HV(A_t union Y_t) - HV(A_t)`.  `A_t` is read before any member of `Y_t` is
  inserted, then the whole batch is committed atomically.
- The reward has no feasibility, repair, no-change, survival, or diversity
  shaping and is not clipped.  Only a fixed positive scaling is allowed;
  tiny negative floating-point noise is set to zero.
- The reward archive resets for every independent NSGA-II run and never takes
  part in environmental selection.
- Zero legal candidates use Base's skip/retry/fallback without a policy
  transition.  One legal candidate is executed directly without a policy
  transition.  Two or more legal candidates invoke the model.
- All methods use the same real-objective evaluation budget.  Inference is not
  an evaluation, and candidates may not be evaluated before selection.
- CCP is introduced only after the deterministic experiment works.  CCP100
  uses the 90th order statistic for all three objectives; common OOS-5000 uses
  the 4500th order statistic and reports OOS q90-HV.

## Implementation gates

1. **Reward/archive foundation**: pure scaled HV reward, atomic `Y_t`, reset,
   no clipping, and contract tests.
2. **Deterministic runner**: fixed single-scene train/test manifests and equal
   real-evaluation accounting.
3. **M controller**: remove the pilot's action-before evaluation and retain
   only operator-then-position learning.
4. **C controller**: common-node cut policy with reward settled after Base
   mutation, repair, and final evaluation.
5. **I controller**: sequential distinct-path choice with one terminal reward
   per completed individual.
6. **Shared data definitions**: graph, chromosome/path, population, masks, and
   normalization; model parameters remain separate.
7. **Five-method smoke and deterministic formal runs**.
8. **CCP100 retraining/OOS-5000**, followed by cross-scene evaluation.

Do not launch a formal or long training run until all gates required by that
stage are marked complete in Git and the smoke manifest passes.

## GitHub and VS Code workflow

The implementation branch is `codex/nsga2-learning-components`, based on
commit `47bada1` from `codex/transformer-ppo-ccp100`.

On the server, keep this study in its own worktree so that other experiments
and uncommitted files remain untouched:

```bash
cd /home/lunet/cokt2/uncertainty
git fetch origin --prune
git worktree add /home/lunet/cokt2/uncertainty-learning-components \
  origin/codex/nsga2-learning-components
cd /home/lunet/cokt2/uncertainty-learning-components
git status --short --branch
git rev-parse HEAD
```

For an existing worktree, update without creating a merge commit:

```bash
cd /home/lunet/cokt2/uncertainty-learning-components
git fetch origin --prune
git pull --ff-only
```

Run the reward/archive contract tests before any smoke run:

```bash
cd /home/lunet/cokt2/uncertainty-learning-components/expanda
python3 -m unittest tests.test_learning_reward_archive
```

Every future long command documented here will use a fresh output directory
and the form:

```bash
nohup python3 -u SCRIPT.py ARGS \
  > FRESH_OUTPUT/logs/RUN.log 2>&1 < /dev/null &
```

Generated results, checkpoints, and logs stay outside Git unless a later
commit explicitly adds a small manifest or aggregate report.
