# CCP100 RF-Reward stages A-C report

Date: 2026-10-15. This is a development report, not a frozen formal-analysis
specification. No formal-test hypervolume was read or computed.

## Project state and isolation

- The user's primary checkout was on `codex/add-expanda-experiments` at
  `58692bd`, behind its remote by six commits, with one modified tracked file
  and multiple untracked files. It was not switched or modified.
- The target branch and remote-tracking reference both pointed to `47bada1`
  (`Add resumable paired formal evaluation batch`). Its registered worktree at
  `/private/tmp/uncertainty-transformer-ppo` no longer existed and was reported
  as prunable.
- Work was performed in the isolated clean clone
  `/private/tmp/uncertainty-rf-reward` at `47bada1`.
- No repository-level `AGENTS.md` exists.
- SSH attempts to the three configured Loughborough hosts reached
  authentication but were rejected. Server branch, processes, result counts,
  checkpoints and hashes therefore remain unverified. No historical completion
  count is used in this report.

## Stage A: implementation audit

### Confirmed invariants

- NSGA-II initialization, crossover, five equiprobable mutation operators,
  encoding repair, feasibility boost and environmental selection remain in
  `baseline_uncertainty.py`. The location controller uses the existing logger
  hook.
- The post-crossover child is evaluated and inserted into a reward-only archive
  before the location action. The archive is excluded from environmental
  selection.
- The exact immediate reward is the post-action marginal normalized 3-D archive
  hypervolume, followed by this priority order: repair failure/revert `-0.10`,
  no decision change `-0.01`, infeasible `-0.05 + 0.025 * clipped violation
  improvement`, positive HV divided by `0.01`, otherwise zero, all clipped to
  `[-1, 1]`.
- Rule, Transformer-PPO and MLP-PPO share the same absolute eligible mask. A
  no-eligible event does not resample the operator and creates no action/reward
  tuple. Frozen PPO inference uses eligible-action argmax.
- The optimization budget counts actual CCP100 evaluations, including
  initialization, pre-mutation child evaluation and boost. Cached identical
  decisions do not consume the budget. The driver reserves four evaluations
  before an offspring pair, so a run may stop up to three evaluations below the
  cap without exceeding it.
- Final exported decisions are feasible rank-0 members of the final population,
  not members of the reward archive.
- PPO checkpoints contain model, optimizer and owned RNG states; formal loading
  verifies architecture, policy seed, catalog/normalization digests and SHA-256.

### Log reuse decision

- Existing PPO `mutation_outcomes.jsonl` records containing `reward`,
  `reward_hv_delta`, `reward_archive_hv_before` and
  `reward_archive_hv_after` can retain their exact labels after their server
  hashes and configuration provenance are verified.
- Old Random/Rule logs that contain only before/after q90 objectives, survival,
  first-front or parent-relation labels cannot be converted into RF-Reward
  labels. A before/after objective difference is not the archive marginal reward.
- Unexecuted candidates have no counterfactual outcome label and are never
  assigned zero. The new dataset contains one row per executed eligible action.
- Existing PPO logs do not contain the complete flattened RF population/archive
  state used here. They could support a reduced-feature sensitivity dataset, but
  they are not accepted as the primary RF training dataset without a separately
  frozen reduced schema.

### Formal-analysis gaps found

- The current OOS-HV script supports four methods and uses training-only Rule
  normalization bounds. It does not yet implement five-method, three-block,
  per-instance pooled OOS bounds.
- The current statistics code analyzes ten instances separately within each
  configuration and one block. It does not first average the three paired blocks
  within an instance, does not include RF, and does not implement the requested
  S0-S8 versus S9-S11 primary analyses.
- These gaps were not changed in stages A-C because the HV formula and primary
  testing family require confirmation before implementation.

## Stage B: training-only position diagnostic

Development-only catalog: one instance per S0-S8 training configuration and
one per S0-S11 development test configuration, generated with master seed
`63000000` and an explicitly smoke-only 48-hour/6-hour release grid. Reward
normalization was built from three 180-evaluation training-only Rule runs.

For each diagnosed event, the pre-mutation child, sampled operator, reward
archive and candidate set were frozen. Up to eight eligible positions were
sampled uniformly without replacement. Each position was executed three times;
the same deterministic new-value seed was used across positions within a
repeat. Probe evaluations were excluded from the trajectory budget and counted
separately.

Results across `train-S0-001` to `train-S2-001`:

- 24 events, 186 distinct positions, 558 probe evaluations.
- 15/24 events (62.5%) had a nonzero range in mean reward across positions.
- Mean within-event reward range: 0.1665; median: 0.01735.
- Reward mean 0.05752, median 0, positive rate 23.1%, negative rate 5.73%,
  range `[-0.0514, 1.0]`.
- All probed positions were Rule-eligible and effective in this small sample;
  5.73% became infeasible. No repair failure, revert or no-change probe occurred.
- An exploratory RF predicting event-centered reward from action features used
  event-grouped cross-validation. Its MAE was 0.06535 versus the zero baseline
  0.04824, R-squared was -0.109 and prediction/label correlation was 0.178.

Interpretation: position-level reward differences exist in this small training
sample, but this diagnostic does not show that the current pre-action features
generalize well enough to predict the within-event ordering.

## Stage C: RF-Reward smoke validation

### Implemented definition

- Fixed-length features use only the pre-action child token summary, selected
  candidate token, all-candidate summary, operator one-hot, population/pending
  offspring/archive summaries and search progress. `p_rule` is an explicit input.
- Collection behavior is
  `p_log(a|s) = (1-rho) p_rule(a|s) + rho / |eligible(s)|`, with smoke
  `rho=0.5`. The exact logging propensity is stored for every executed action.
- Primary fitting is unweighted RandomForestRegressor on exact immediate reward.
  The rationale is to estimate `E[reward | pre-action features]` under explicit
  overlap without imposing high-variance inverse-propensity weights. Propensities
  remain available for overlap diagnostics and an optional, separately frozen
  clipped-weight sensitivity analysis.
- Hyperparameters are selected only by instance-grouped validation. Formal
  tuning has not been frozen.
- Frozen inference predicts every Rule-eligible candidate and chooses reward
  argmax. Ties are resolved by larger `p_rule`, then stable candidate order.
- No-eligible events retain the existing no-action behavior. Failure, revert,
  no-change and infeasible executed actions are not filtered. Unit tests cover
  failure-row retention; the smoke sample happened to contain no repair failure,
  revert or no-change action.

### Smoke results

- Six training instances, 1,783 actual CCP100 evaluations, 236 mutation events,
  149 executed eligible actions and 87 no-eligible skips.
- All 149 executed actions had exact labels; 18 infeasible-after samples were
  retained. Reward mean 0.01065, standard deviation 0.09704, positive fraction
  10.1%, negative fraction 12.1%, range `[-0.05190, 1.0]`.
- Logging propensity: minimum 0.01913, median 0.05, maximum 0.5; diagnostic
  un-clipped IPS effective sample size 118.3/149.
- Best smoke hyperparameters: 200 trees, depth 8, minimum leaf 5, 0.5 feature
  fraction. Instance-grouped mean MAE/RMSE were 0.02982/0.06399 versus
  training-fold mean baselines 0.03456/0.06679. This is pipeline evidence only.
- Model SHA-256:
  `567858513b3246b01b015de12061603a76716e82c4d284e5a73672d12868f21c`.
  Training-data digest:
  `1f0263fdc2c187ced2ba60e647348465ba2fb51566d0d8fd0e4c8e003bcf582d`.
- Optimization smoke on an independent development test manifest used 297/300
  actual evaluations, 38 mutation events, 31 effective modifications and seven
  no-eligible skips. It exported 11 feasible final-population rank-0 decisions.
- OOS smoke evaluated exactly those 11 decisions on the manifest's common 5,000
  scenarios. OOS master digest:
  `f96aafaa8381b7c2f2d11452bf8807586386e824397e71997e7c5f1eee9efdcd`.
  No HV was computed.

Development stages A-C used 3,703 actual CCP100 evaluations in total: 533 for
normalization, 532 diagnostic trajectory evaluations, 558 counterfactual probe
evaluations, 1,783 collection evaluations and 297 RF optimization evaluations.
The OOS smoke additionally used 55,000 scenario-decision evaluations.

### Independent server smoke verification

The committed pipeline was independently rerun on `lnx-grid-41` from commit
`77aa6b2`, using Python 3.10.22, PyTorch 2.5.1+cu121 and scikit-learn 1.5.2.
All 48 targeted tests passed.  The training-only reward normalizer used by this
run had SHA-256
`de73404449c2b612609ad5248f815793f99b676e4260e2068ae403f043addbcd`.
The background pipeline exited normally and no traceback, error or exception
was present in its log.

The server position diagnostic used one training instance, eight fixed
child/operator events, eight eligible positions per event and three repetitions:

- 192 probe evaluations and 64 distinct positions; all probes made an effective
  modification in this sample.
- Seven of eight events had a nonzero mean-reward range across positions.  The
  mean and median within-event ranges were 0.02869 and 0.02369.
- The exploratory event-grouped action-feature RF had MAE 0.007394 versus
  0.006083 for the event-centered zero baseline, R-squared 0.0297 and
  prediction/label correlation 0.292.  Thus the server diagnostic again found
  position differences, but did not show useful out-of-event prediction.

The server RF collection used three training instances and retained 83 executed
actions.  Four actions were infeasible after mutation.  There were no mutation
failures, repair failures, reverts, unchanged decisions or ineffective actions
in this particular sample.  The maximum discrepancy between the stored archive
HV delta and `archive_hv_after - archive_hv_before` was exactly zero.  Logged
propensities ranged from 0.01726 to 0.5 and the diagnostic un-clipped IPS
effective sample size was 63.82/83.

The selected RF had instance-macro validation MAE/RMSE 0.03867/0.08322, while
the corresponding training-fold mean baseline achieved 0.02462/0.06895.
Therefore the RF was worse than the simple baseline on both selection metrics
in this small server sample.  The serialized model hash, independently matched
against its training summary, was
`91f39bda2756e1b8cf96d3a01282f9ffab330c6112da860ffcb2f95b1567a811`.

The independent RF policy smoke used 297/300 actual evaluations, exported six
feasible final-population rank-0 decisions and evaluated those six decisions on
the development manifest's common 5,000 OOS scenarios.  It computed no HV.

The server gate conclusion is therefore **engineering pass, predictive-evidence
insufficient**.  Exact-label generation, propensity logging, fitting,
serialization, inference, budget accounting, final-front export and OOS source
flow all passed.  Neither the one-instance diagnostic nor the three-instance RF
validation supports a performance claim or justifies launching the full formal
collection without the planned design confirmation.

Relevant tests: 48 passed, one PyTorch-dependent test skipped because the local
temporary environment has no PyTorch. A broad legacy test discovery produced
352 passes, one skip, three failures and 37 setup errors in unrelated A-lite
tests due missing historical commit `369585e` in the grafted branch and a
pre-existing `CONFIDENCE_ONTIME` API mismatch. The target README test set plus
new RF tests passes.

## Proposed stages D-E (not started or frozen)

### RF collection and fitting

1. Use the verified formal S0-S8 catalog (450 instances) and the already frozen
   reward normalizer if its digest matches the PPO training provenance.
2. Generate three independent collection trajectories. Each trajectory uses a
   predeclared instance order, algorithm-seed namespace, policy seed and
   `rho=0.5` mixture, with at most 5,000 actual evaluations per instance.
3. This is at most 2.25 million evaluations per trajectory and 6.75 million for
   three RF models. Record every run configuration, instance order, seeds,
   actual evaluation count, elapsed time, source hashes and data digest.
4. Within each trajectory, tune only on its training instances using
   configuration-balanced, instance-grouped folds. Freeze the grid, selection
   metric and RF seed before generation. Refit the selected model on all 450
   instances and store model/data/config hashes.
5. Keep unweighted regression as the proposed primary fit. Predeclare one
   optional sensitivity fit using stabilized, clipped propensity weights only if
   the supervisor wants it; do not replace the primary fit based on its result.

### Formal test

- Three RF models map to the three existing paired blocks: 120 instances x 3 =
  360 RF optimizations, at most 5.4 million actual evaluations.
- Match each block's algorithm seed, path seed, CCP100 seed and OOS-5000 master
  to Random, Rule, Transformer and MLP. Reuse an existing run only after method,
  instance, seeds, budget, final-front semantics and hashes pass validation.
- Missing existing-method runs may be filled under the original protocol only
  after the server inventory is re-audited. Existing outputs are never replaced.

### Proposed primary inference

- Within instance and comparison, compute the three block-paired differences and
  average them, making the test instance the primary unit.
- Analyze S0-S8 (90 instances) and S9-S11 (30 instances) separately. Preserve
  configuration strata in a stratified instance bootstrap for the mean paired
  difference and 95% interval; also report each configuration and each model seed.
- Proposed primary comparisons are RF vs Rule, Transformer vs Rule and
  Transformer vs RF. Use two Holm families (three hypotheses within S0-S8 and
  three within S9-S11). Report raw/adjusted p-values, mean/median differences and
  win/tie/loss. Other comparisons are secondary/descriptive unless separately
  predeclared.

## Items requiring confirmation before D or E

1. Server access and a fresh inventory of branch/commit, running processes,
   catalogs, normalizer, six final PPO checkpoints and all existing run counts.
2. Formal RF collection mixture `rho=0.5`, three fully independent trajectories,
   unweighted primary fitting and the optional clipped-weight sensitivity rule.
3. The exact grouped-CV split, tuning grid and selection metric.
4. Whether the formal reward normalizer already used by PPO is reused unchanged.
5. The exact OOS-HV boundary formula. The current code is not the proposed
   five-method pooled formula. In particular, confirm whether per-instance
   min/max are taken from the pooled global nondominated set or from the union of
   all per-run nondominated fronts, the zero-span rule, no-clipping rule, primary
   reference `(1.1,1.1,1.1)`, and sensitivity references.
6. The instance-aggregated primary hypotheses, two Holm families and stratified
   bootstrap described above.
7. Resource allocation and launch order for the 6.75M training plus 5.4M RF test
   evaluation upper bounds.
