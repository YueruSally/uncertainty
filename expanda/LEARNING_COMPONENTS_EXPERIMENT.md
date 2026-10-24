# NSGA-II learning-components experiment

This is the implementation and execution record for the new five-method
study: `Base`, `I`, `C`, `M`, and frozen `I+C+M`.

## Frozen scientific contract

- Begin with one deterministic scene and disjoint training/test instances.
  There is no validation split, early stopping, or best-checkpoint selection.
- I preserves Base's representable initial-individual structure.  PPO chooses
  the identities of `k` distinct paths; Base still chooses `k`, generates the
  split shares, and normalizes them.
- C is called only for common-node crossover.  For the allocation pair that
  Base chooses in each Batch, the model chooses among legal common nodes.  If
  one crossover contains several learnable Batch cuts, they form one decision
  sequence: intermediate rewards are zero and only the final learnable cut
  receives the joint terminal reward.  The selected cuts are followed by Base
  mutation, repair, and the original final-offspring evaluation.  There is no
  post-crossover intermediate objective evaluation.
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

## Current implementation status

- Gate 1 is implemented and covered by
  `tests.test_learning_reward_archive`.
- Deterministic/CCP100 evaluation modes now share the episode runner.
  Deterministic training requires one explicit `--configuration` so the first
  stage cannot accidentally mix scenes.
- The M PPO controller no longer evaluates a crossed child before mutation.
  Zero-candidate decisions create no action, and a single legal candidate is
  executed without model inference or a PPO transition.
- The C controller now preserves the 0.5 common-node / 0.5 structural-operator
  choice, controls only legal common-node cuts, and atomically settles the two
  children after Base mutation, repair, and final evaluation.  C has its own
  checkpoint/model parameters and raw two-parent chromosome tokens.
- The I controller retains capacity-aware and greedy seeds unchanged.  For
  each remaining random individual, Base still draws `k`, consumes its paired
  random sample, generates split shares, and normalizes the allocations; I
  sequentially chooses `k` distinct path identities.  Intermediate rewards
  are zero and the evaluated complete individual supplies one terminal HV
  reward.  I uses a separate checkpoint and a return horizon covering the
  complete construction sequence.
- The first-version shared `shared-network-tokens-v1` definition is now
  implemented for all three policies.  Nodes, edges, Batch terminals,
  chromosomes, candidate paths, and population state use fixed common
  definitions; the Transformer encodes the node/edge attribute tokens and
  links candidate paths or cut nodes to the encoded network.  A GNN is a later
  network-encoder option and is not required for the deterministic first
  version.  Each component still has its own chromosome/action tokens and its
  own model parameters.  M chromosome/action tokens no longer contain decoded
  path cost, emission, travel time, or Rule probability.  Population summaries
  use the same implementation wherever the corresponding state exists.
- Gate 6 is complete: server development smokes for M, C, and I each consumed
  exactly 200 evaluations with the first-version network-token Transformer.
- The Base budget controller, frozen I+C+M coordinator, and paired five-method
  deterministic runner are implemented.  The learning controllers consume
  the shadow Base position draw before replacing it, so equal algorithm seeds
  remain paired beyond the learned decision.
- Gate 7 wiring and continuous-training smokes are complete.  The completed
  small-budget S0 screening protocol used 50 `train-S0` instances, 10 disjoint
  `test-S0` instances, population 20, and 400 real-objective evaluations per
  independent run.  These results must not be described as the final
  full-scale experiment until the budget audit and training-only convergence
  pilot in `STAGE1_AUDIT_PLAN.md` are complete.
  `prepare_deterministic_s0_normalization.py` freezes reward/final-HV bounds
  from deterministic Base fronts on the 50 training instances only.  It does
  not read validation or test instances and does not clip transformed values.
- The three formal policies are trained separately with policy seed 301 and
  the same shuffled 50-instance order, with the last checkpoint used directly.
  The server completion artifacts reported 20,000 evaluations per component
  and final checkpoint SHA-256 values beginning `f44c670d` (M), `57e3e8ba`
  (I), and `2e9297b0` (C).
- `run_five_method_deterministic_batch.py` is the fail-closed, resumable
  ten-instance test entry point.  It verifies checkpoint provenance and the
  common training order, runs Base/I/C/M/frozen-I+C+M with paired seeds, checks
  zero policy updates during inference, and writes `results.json`,
  `results.csv`, and `analysis.json`.  The analysis uses final-population HV,
  paired bootstrap intervals, an exact paired sign-flip test, and one Holm
  family for the four comparisons against Base.  Run its `--preflight-only`
  mode before the long command to validate all frozen inputs without creating
  result files.

Do not launch CCP or cross-scene work until the S0 deterministic batch has a
valid `BATCH_COMPLETE.json` and its results have been inspected.

## GitHub and VS Code workflow

The implementation branch is `codex/nsga2-learning-components`, based on
commit `47bada1` from `codex/transformer-ppo-ccp100`.

On the server, keep this study in its own worktree so that other experiments
and uncommitted files remain untouched:

```bash
cd /home/lunet/cokt2/uncertainty
git fetch origin --prune
git worktree add -b codex/nsga2-learning-components \
  /home/lunet/cokt2/uncertainty-learning-components \
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

If the worktree was previously created directly from `origin/...` and reports
`HEAD (no branch)`, attach it once before pulling:

```bash
cd /home/lunet/cokt2/uncertainty-learning-components
git switch -c codex/nsga2-learning-components \
  --track origin/codex/nsga2-learning-components
```

Generated instance catalogs and reward-normalization files are not Git
objects.  A new worktree therefore does not inherit them from an older
worktree.  Locate the existing artifacts and pass their absolute paths:

```bash
find /home/lunet/cokt2 -type f \
  -path '*/transformer_ppo_runs/reward_normalization.json' -print
find /home/lunet/cokt2 -type f \
  -path '*/transformer_ppo_runs/instances/catalog.json' -print
```

Do not copy or regenerate these artifacts for a smoke test.  Reuse the
existing catalog and training-only normalization by absolute path, while
writing the new output under the learning-components worktree.

Run the reward/archive contract tests before any smoke run:

```bash
cd /home/lunet/cokt2/uncertainty-learning-components/expanda
python3 -m unittest tests.test_learning_reward_archive
```

On the server environment with the full scientific dependencies installed,
also run the regression and evaluation-order tests:

```bash
python3 -m unittest \
  tests.test_initialization_learning \
  tests.test_crossover_learning \
  tests.test_multi_instance_ccp100 \
  tests.test_transformer_ppo_framework
```

Every future long command documented here will use a fresh output directory
and the form:

```bash
nohup python3 -u SCRIPT.py ARGS \
  > FRESH_OUTPUT/logs/RUN.log 2>&1 < /dev/null &
```

Generated results, checkpoints, and logs stay outside Git unless a later
commit explicitly adds a small manifest or aggregate report.
