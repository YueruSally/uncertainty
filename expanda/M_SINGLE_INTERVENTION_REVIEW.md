# M single-intervention review

## Review scope

This branch contains the mutation-control implementation used to test whether
one learned replacement-path decision can change the final NSGA-II archive.
The diagnostic holds the instance, algorithm seed, evaluation budget, and all
other mutation decisions fixed to the paired Base run. The model is used at
exactly one global mutation-call index.

Branch: `codex/ppo-global-actor-audit`

## Successful causal cases

All results below use deterministic evaluation, population 20, and a budget of
400 real objective evaluations.

| Instance | Algorithm seed | Model event | Paired Base final HV | One-shot final HV | Relative final HV change |
|---|---:|---:|---:|---:|---:|
| S0_006 | 777002 | 14 | 0.4370430019 | 0.4731757823 | +8.27% |
| S0_005 | 777002 | 16 | 0.3668900883 | 0.4281877307 | +16.71% |
| S0_006 | 777001 | 38 | 0.3501321489 | 0.4428436774 | +26.48% |

The percentage is the change in **final normalized archive hypervolume**
relative to the paired Base run:

`100 * (one_shot_final_hv - base_final_hv) / base_final_hv`

These are selected successful cases, not an average result and not evidence of
a stable 30,000-evaluation improvement. Other audited one-shot states produced
negative or neutral final changes. Their purpose is to establish that the M
action space and the controller-to-NSGA-II path can produce material downstream
effects when a useful replacement action is selected.

## Code to review

- `transformer_ppo/controller.py`: exact one-event model gating, Base
  passthrough, selected replacement-path execution, and mutation diagnostics.
- `transformer_ppo/episode.py`: propagation and reporting of the one-event
  intervention setting.
- `run_learning_components_deterministic.py`: frozen-policy diagnostic entry
  point and `--model-choice-event` validation.
- `multi_instance_logging.py`: executed mutation outcome and environmental
  survival logging.
- `tests/test_transformer_ppo_framework.py`: exact one-event intervention
  tests.
- `tests/test_learning_components_v2_controls.py`: exact Base-passthrough and
  replacement-path control tests.

The main implementation milestones are:

- `b939aa2`: exact one-shot mutation intervention.
- `3f69f49`: exact Base semantics for non-model mutation calls.
- `dd7fca1`: diagnostic logging for unevaluated Base passthrough calls.
- `1f0585e`: full mutation policy-state logging for downstream labels.
- `729e5e4`: counterfactual-panel analysis.

## Generic reproduction command

The instance JSON, training-only normalization file, and matching M checkpoint
are required. Checkpoints and full run directories are intentionally not stored
in Git because they are generated artifacts.

```bash
python3 -u run_learning_components_deterministic.py \
  --instance "$INSTANCE" \
  --normalization "$NORMALIZATION" \
  --out "$OUT" \
  --component M \
  --checkpoint "$CHECKPOINT" \
  --algorithm-seed "$ALGORITHM_SEED" \
  --inference-action-mode sample \
  --inference-policy-seed "$POLICY_SEED" \
  --model-choice-event "$EVENT" \
  --evaluation-budget 400 \
  --pop 20 \
  --gens 1000
```

The paired Base run uses the same instance, normalization, algorithm seed,
budget, and population, with `--component BASE`. For a causal comparison, also
verify the non-target mutation calls follow Base and compare
`final_archive_lineage.json` where appropriate.

## Current interpretation and next step

The control implementation is capable of applying and preserving a beneficial
mutation decision, but the learned scorer does not yet identify such decisions
reliably across states and instances. The next experiment therefore trains on
the paired one-generation HV advantage, adds nonlinear state-candidate
interactions, and falls back to exact Base behaviour when predicted advantage
is not sufficiently positive.
