# Stage-1 audit before the full-scale deterministic experiment

This stage does not launch another NSGA-II run. It audits the completed S0
artifacts before a population/evaluation-budget decision is made.

## Budget provenance already established

- Commit `ffbb6fa` introduced the ten-instance five-method batch and described
  population 20 with 400 real-objective evaluations as the formal S0 protocol.
- Both values were supplied explicitly by the run commands. The training
  command requires an evaluation budget; population defaults to 100, so 20/400
  did not arise from an unnoticed runtime default.
- The preceding development smoke used 200 evaluations. No scientific or
  convergence justification for promoting 20/400 to a formal budget is stored
  in the commit or experiment record.
- The original deterministic NSGA-II code retains population 250 and 200
  generations (`baseline3_desktop_original.py`, marked "Stage 1 tuned (F5)").

Until the intended budget is confirmed and a training-only convergence pilot
is complete, the 20/400 results are **small-budget deterministic screening
results**, not the final full-scale experiment.

The external question to resolve is:

> Was the original formal NSGA-II setting population 250 and 200 generations
> (roughly 50,000 objective evaluations), and must the learning-component study
> retain this scale, or may it use a smaller budget justified in advance by a
> training-only convergence and resource pilot?

## Existing-artifact audit

After pulling this branch on the server, run the read-only audit below. Replace
only `FIRST_TEST_BATCH` if its directory name differs. No checkpoint is loaded
and no optimisation is launched.

```bash
cd /home/lunet/cokt2/uncertainty-learning-components/expanda

I_TRAIN=transformer_ppo_runs/s0_formal_i_seed301_b400_p20_30b0fac
C_TRAIN=transformer_ppo_runs/s0_formal_c_seed301_b400_p20_30b0fac
M_TRAIN=transformer_ppo_runs/s0_formal_m_seed301_b400_p20_30b0fac
FIRST_TEST_BATCH=transformer_ppo_runs/s0_formal_five_method_test_seed301_b400_p20_ffbb6fa
TEST_REPEATS=transformer_ppo_runs/s0_formal_five_method_10seeds_seed301_b400_p20_2ec52dd
AUDIT_OUT=transformer_ppo_runs/s0_stage1_existing_artifact_audit

python3 audit_learning_components_stage1.py \
  --i-training "$I_TRAIN" \
  --c-training "$C_TRAIN" \
  --m-training "$M_TRAIN" \
  --first-test-batch "$FIRST_TEST_BATCH" \
  --test-repeats "$TEST_REPEATS" \
  --out "$AUDIT_OUT"

cat "$AUDIT_OUT/audit.md"
```

The report contains:

- I/C/M training episode, model-action, reward-event and PPO-update counts;
- nonzero-reward fractions using the component-specific event denominator;
- frozen-policy candidate counts, entropy, distance from uniform selection,
  paired shadow-random differences, single/no-candidate and fallback counts;
- an equal-real-evaluation check and method runtime/inference breakdown.

Two measurements cannot be recovered from old artifacts: per-update PPO
batch/loss/KL traces, and separate objective-evaluation versus feature-building
time. They require additional logging followed by a small training-only
diagnostic, not a rerun of the current test set.
