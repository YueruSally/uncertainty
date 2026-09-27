# Post-V1 multi-instance CCP100 experiment

Run these commands **only in the remote VS Code terminal on `cokt2@lnx-grid-41`**.
Each formal optimisation/OOS stage runs under `nohup`. Do not reuse an output
directory or alter a plan, instance, model, or strategy after freezing it.
The original V1 branch and `formal_weighted_v6_30runs` are untouched.

## Definitions

`deadline_window_alpha` means `LT = ET + max(alpha * (original_LT - ET), 1 hour)`.
It is not the CCP objective quantile (`alpha=.90` in the older runner).
CCP100 uncertainty sources and distributions are unchanged. `instance_id`
identifies a fixed batch configuration; the CCP training scenario digest
identifies the 100 random samples within one optimisation run.

Random selects from every original position. Rule samples with the original
hierarchical probability conditioned on eligible positions. Learning applies
the same mask and reweights Rule probabilities by model scores, with epsilon
mass on Rule **inside the mask**. If none is eligible, no location is selected
and the operator is not resampled. The V1 files and models are never reused.

`parent_relation` uses fixed midpoint score
`P(dominates) + 0.5 * P(incomparable)`, reflecting an explicit ordinal
loss/tie/win utility. Invalid/equal-objective moves are excluded from that
three-class model and retained as failures in the two binary targets.

## Prepare and audit

From the repository root, after checking the branch is
`codex/multi-instance-ccp100` and `git status --short` is empty:

```bash
python -m pip install -r requirements.txt
cd expanda
python -m unittest tests.test_multi_instance_ccp100 tests.test_learning_ccp100_logging tests.test_learning_policy
cd ..
mkdir -p expanda/multi_instance_runs/logs
nohup python -u expanda/multi_instance_catalog.py \
  --out expanda/multi_instance_runs/instances \
  > expanda/multi_instance_runs/logs/instances.log 2>&1 < /dev/null &
```

Wait for completion and inspect the log and all 17 instance JSON files.
The generator checks that every OD has paths in the unchanged path builder.
It samples training/validation instances independently and uses unseen OD
pairs and new quantities in T1-T5. Then freeze the plan:

```bash
python expanda/multi_instance_plan.py make \
  --instances expanda/multi_instance_runs/instances \
  --out expanda/multi_instance_runs/plan.json
```

Run one small-budget smoke in the remote server, separately from the formal
254-run count:

```bash
nohup python -u expanda/run_multi_instance_ccp100.py \
  --instance expanda/multi_instance_runs/instances/S0.json \
  --out expanda/multi_instance_runs/smoke/S0_rule \
  --policy rule --pop 20 --gens 5 --evaluation-budget 500 \
  > expanda/multi_instance_runs/logs/smoke.log 2>&1 < /dev/null &
```

Inspect `COMPLETE.json`, candidate probabilities, skips, and outcomes. The
audit is four full-budget optimisations (S0/S9, Random/Rule), each with 1000
and extended 5000 OOS evaluations. Record actual optimisation and OOS times
before launching the 50 training runs:

```bash
nohup python -u expanda/multi_instance_plan.py run \
  --plan expanda/multi_instance_runs/plan.json --stage audit \
  --root expanda/multi_instance_runs/jobs \
  > expanda/multi_instance_runs/logs/audit.log 2>&1 < /dev/null &
```

## Train and choose the validation strategy

Run each stage only after the prior log reports every job as `DONE`.

```bash
nohup python -u expanda/multi_instance_plan.py run \
  --plan expanda/multi_instance_runs/plan.json --stage train \
  --root expanda/multi_instance_runs/jobs \
  > expanda/multi_instance_runs/logs/train.log 2>&1 < /dev/null &

nohup python -u expanda/multi_instance_plan.py run \
  --plan expanda/multi_instance_runs/plan.json --stage validation_rule \
  --root expanda/multi_instance_runs/jobs \
  > expanda/multi_instance_runs/logs/validation_rule.log 2>&1 < /dev/null &
```

Freeze validation normalisation from Rule, before running Learning:

```bash
python expanda/multi_instance_analysis.py freeze-bounds \
  --root expanda/multi_instance_runs/jobs \
  --instances expanda/multi_instance_runs/instances --split validation \
  --out expanda/multi_instance_runs/validation_bounds.json

python expanda/multi_instance_training.py build \
  --runs-root expanda/multi_instance_runs/jobs \
  --out expanda/multi_instance_runs/dataset
python expanda/multi_instance_training.py train \
  --dataset expanda/multi_instance_runs/dataset \
  --out expanda/multi_instance_runs/models

nohup python -u expanda/multi_instance_plan.py run \
  --plan expanda/multi_instance_runs/plan.json --stage validation_labels \
  --root expanda/multi_instance_runs/jobs \
  --models expanda/multi_instance_runs/models \
  > expanda/multi_instance_runs/logs/validation_labels.log 2>&1 < /dev/null &

python expanda/multi_instance_diagnostics.py \
  --root expanda/multi_instance_runs/jobs \
  --out expanda/multi_instance_runs/validation_policy_diagnostics.json

python expanda/multi_instance_analysis.py choose-label \
  --root expanda/multi_instance_runs/jobs \
  --bounds expanda/multi_instance_runs/validation_bounds.json \
  --models expanda/multi_instance_runs/models \
  --out expanda/multi_instance_runs/label_choice_1000.json
```

If the choice status is `needs_5000_extension`, evaluate only its two fixed
labels and Rule on scenarios 1001-5000, without rerunning optimisation:

```bash
nohup python -u expanda/multi_instance_extend_oos.py \
  --plan expanda/multi_instance_runs/plan.json \
  --root expanda/multi_instance_runs/jobs \
  --choice expanda/multi_instance_runs/label_choice_1000.json \
  > expanda/multi_instance_runs/logs/validation_extension.log 2>&1 < /dev/null &

python expanda/multi_instance_analysis.py choose-label \
  --root expanda/multi_instance_runs/jobs \
  --bounds expanda/multi_instance_runs/validation_bounds.json \
  --models expanda/multi_instance_runs/models --size 5000 \
  --initial expanda/multi_instance_runs/label_choice_1000.json \
  --out expanda/multi_instance_runs/label_choice_final.json
```

Otherwise, copy the frozen `label_choice_1000.json` to
`label_choice_final.json`. Then run epsilon=0 on the same validation pairs:

```bash
nohup python -u expanda/multi_instance_plan.py run \
  --plan expanda/multi_instance_runs/plan.json --stage epsilon_zero \
  --root expanda/multi_instance_runs/jobs \
  --models expanda/multi_instance_runs/models \
  --lock expanda/multi_instance_runs/label_choice_final.json \
  > expanda/multi_instance_runs/logs/epsilon_zero.log 2>&1 < /dev/null &

python expanda/multi_instance_analysis.py freeze-epsilon \
  --root expanda/multi_instance_runs/jobs \
  --bounds expanda/multi_instance_runs/validation_bounds.json \
  --label-choice expanda/multi_instance_runs/label_choice_final.json \
  --out expanda/multi_instance_runs/policy_lock.json
```

## Five held-out tests

The frozen model/epsilon must remain unchanged. Complete 100 Random/Rule
controls first, freeze per-instance test bounds, then run 50 Learning jobs.

```bash
nohup python -u expanda/multi_instance_plan.py run \
  --plan expanda/multi_instance_runs/plan.json --stage test_controls \
  --root expanda/multi_instance_runs/jobs \
  > expanda/multi_instance_runs/logs/test_controls.log 2>&1 < /dev/null &

python expanda/multi_instance_analysis.py freeze-bounds \
  --root expanda/multi_instance_runs/jobs \
  --instances expanda/multi_instance_runs/instances --split test \
  --out expanda/multi_instance_runs/test_bounds.json

nohup python -u expanda/multi_instance_plan.py run \
  --plan expanda/multi_instance_runs/plan.json --stage test_learning \
  --root expanda/multi_instance_runs/jobs \
  --models expanda/multi_instance_runs/models \
  --lock expanda/multi_instance_runs/policy_lock.json \
  > expanda/multi_instance_runs/logs/test_learning.log 2>&1 < /dev/null &

python expanda/multi_instance_test_report.py \
  --root expanda/multi_instance_runs/jobs \
  --bounds expanda/multi_instance_runs/test_bounds.json \
  --lock expanda/multi_instance_runs/policy_lock.json \
  --out expanda/multi_instance_runs/test_report.json
```

Run counts: 4 audit + 50 training Rule + 10 validation Rule + 30 label
validation + 10 epsilon ablation + 100 test controls + 50 test Learning =
254 optimisations. OOS evaluations and metadata generation are additional
timed work, not additional optimisation runs. Each formal run caps actual
CCP100 evaluations at 15,000 and requires a budget stop. The final report
audits paired seeds and instance, CCP, and OOS digests before statistics.
