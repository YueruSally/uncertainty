# EV–CCP OOS supervisor-revision audit

## Scope and conclusion

The current Run-1 optimisation-stage artifacts contain 259 EV solutions and
263 CCP30 solutions.  The 263-row CCP30 q90 comparison already contains every
current original CCP30 solution.  Its non-contiguous numeric suffixes are not
evidence of post-OOS filtering: `candidate_rows()` assigns the suffix before
removing exact duplicate decisions, so skipped suffixes are expected.

The current Run-1 solutions cannot be given the complete requested mean,
median and q90 table solely from the formal common-validation files.  Those
files contain an older Run-1 candidate version.  A validation-only rerun of all
522 current fixed decisions is required for a coherent complete table; EV or
CCP optimisation is not required.  Revalidating only the numerically “missing”
IDs would be incorrect because none of the older Run-1 records has the same
decision fingerprint as the current record with that ID.

## 1. Original EV solutions

File:
`formal_ev_vs_ccp_s30_30runs/run_01/EV/final_feasible_nondominated.json`.

Count: 259 unique persisted decisions.  IDs range from
`formal-run-01-EV-solution-0000` to `...-0295`, with 37 skipped numeric
suffixes.  The completion marker independently records `candidate_count=259`.

## 2. Original CCP30 solutions

File:
`formal_ev_vs_ccp_s30_30runs/run_01/CCP30/final_feasible_nondominated.json`.

Count: 263 unique persisted decisions.  IDs range from
`formal-run-01-CCP30-solution-0000` to `...-0298`, with these 36 skipped
suffixes:

`0008, 0018, 0020, 0023, 0030, 0031, 0039, 0043, 0050, 0054, 0067,
0079, 0115, 0123, 0132, 0133, 0152, 0163, 0199, 0200, 0218, 0227,
0247, 0249, 0254, 0255, 0256, 0260, 0265, 0278, 0282, 0290, 0292,
0293, 0296, 0297`.

## 3. S=5,000 validation implementation

The formal common validation is implemented in
`run_formal_ev_vs_ccp_s30_30runs.py` after all optimisation completions.  It
restores each fixed decision, evaluates it on the common seed-930001 S=5,000
ScenarioSet, verifies the decision fingerprint, and writes
`validation/scenario_level_results.csv.gz` and
`validation/per_candidate_summary.json`.

Run-1-specific validation is implemented in `run_run1_oos_validation.py`.
CCP30 training-q90 versus OOS-q90 validation is implemented in
`run_ccp30_q90_oos_validation.py`.

## 4. Post-validation filtering

The formal evaluator writes all candidates before any reporting operation.
However, legacy reporting did apply additional non-dominated analyses:

- `postprocess_formal_ev_vs_ccp.py` constructs global and per-run OOS-mean and
  OOS-q90 non-dominated fronts for metrics.
- the previous `run_run1_oos_validation.py` calculated scenario-wise Pareto
  appearances and an OOS-mean non-dominated flag.
- `run_ev_ccp_oos_pilot.py` writes OOS mean/q90 Pareto subsets.

Those operations explain filtered *front/report* files, but they do not explain
the 263-row CCP30 comparison.  The revised Run-1 output path no longer calls a
non-dominated operation and writes one statistics row per loaded original.

## 5. Why CCP30 has 263 rows

`run_ccp30_q90_oos_validation.py` loads the current 263-row optimisation file
directly and does no non-dominated operation.  Its existing CSV has 263 unique
IDs and its ID set exactly equals the source file's ID set.

The gaps arise in `run_ev_ccp_oos_pilot.py:candidate_rows`: it enumerates the
already selected optimisation front, uses that enumeration index in the ID,
then skips an individual when its decision fingerprint has already been seen.
Thus gaps mean exact-decision duplicates were omitted at optimisation-result
persistence, not that solutions were removed after OOS evaluation.

## 6. Reuse decision

- Reusable for current CCP30: the existing 263-row
  `ccp30_q90_s30_vs_s5000.csv` (all source IDs match), including cost,
  emissions and makespan.
- Reusable for current Run-1 plots: the 522-row
  `run1_oos_solution_objective_summary.csv` contains means for all 259 EV and
  263 CCP30 current solutions.
- Not sufficient for the requested complete table: it has no medians and no
  complete q90 set.
- Not reusable as current per-solution full statistics:
  `run1_oos_solution_stats.csv` has 545 rows from an older candidate version
  (269 EV and 276 CCP30).  Although some IDs overlap, zero overlapping rows
  have the same decision fingerprint as the current source.
- The formal 15,664-candidate summary/scenario file likewise represents that
  older Run-1 version: compared with current sources, it has 47 missing current
  IDs and 69 unexpected old IDs.

Therefore the safe revision is a validation-only rerun of all 522 fixed Run-1
solutions on the unchanged seed-930001 scenarios.  No optimisation, uncertainty
model, distribution, parameter, or Scheme-B accounting change is needed.

## Code revision

`run_run1_oos_validation.py` now defaults to the additive directory
`run1_oos_validation_all_original`, emits mean/median/q90 for all three
objectives for every source solution, produces the three requested 2D plots
from OOS means, and retains a clearly named 3D supporting plot.  Its revised
execution path performs no Pareto/non-dominated filtering.

`run_ccp30_q90_oos_validation.py` now also defaults to that additive directory,
derives its expected count from the source rather than hard-coding 263, matches
by `solution_id`, and records `post_validation_pareto_filtering=false`.

Both validation paths retain the existing ScenarioSet construction and Scheme
B evaluation, including waiting emissions in total emissions and regional
carbon cost.
