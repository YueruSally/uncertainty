# One full-size paired EV/CCP pilot

This is one pilot replicate per method, not the formal 30-run experiment.
The active model is the quantile-objective formulation documented in
`../CCP_SEMANTICS_AUDIT.md`; punctuality is diagnostic only.

## Frozen design

- population 300; generations 500; alpha 0.90
- shared NSGA-II seed: 861337
- S50 master training seed: 862050; CCP30 is its exact first-30 prefix
- independent S5000 validation seed: 869999
- common path library and structural option ordering; path-library seed 0

## Runtime and candidate counts

| Method | Optimisation seconds | Candidates | Unique fingerprints |
|---|---:|---:|---:|
| EV | 1012.926 | 264 | 264 |
| CCP30 | 962.188 | 269 | 269 |
| CCP50 | 958.221 | 270 | 270 |

Stage C took 37.487 seconds for 803 candidates x 5000 scenarios = 4,015,000
candidate-scenario evaluations. No fingerprints are shared between methods.
Every pre/post Stage C fingerprint matches.

The historical S200 pilot runtime was 650.3 seconds at population 300 and 500
generations. Descriptive runtime ratios are EV 1.56x, CCP30 1.48x, and CCP50
1.47x. These are not controlled scaling estimates: the historical run used 20
batches and a different chance-feasibility implementation, whereas this run
uses 40 batches and the current quantile-objective model. Vectorised scenario
simulation and path caching also make runtime non-linear in S.

## Generalisation delta ranges

Percentage deltas are `(OOS - training) / training * 100`.

| Method / primary comparison | Cost | Emission | Makespan |
|---|---:|---:|---:|
| EV expected-input to OOS mean | +0.537% to +1.661% | numerical zero | +0.701% to +15.708% |
| CCP30 training q90 to OOS q90 | +0.130% to +1.285% | 0% | -2.810% to +9.517% |
| CCP50 training q90 to OOS q90 | -0.071% to +1.306% | 0% | +0.329% to +9.683% |

## Common held-out Pareto views

| Method | OOS-mean front | OOS-q90 front |
|---|---:|---:|
| EV | 50 | 68 |
| CCP30 | 7 | 16 |
| CCP50 | 71 | 64 |

This single pilot does not establish statistical superiority.

## Punctuality diagnostic

Minimum per-batch OOS on-time probabilities ranged from 0 to 0.0304 for EV,
0.0042 to 0.0410 for CCP30, and 0.0178 to 0.0410 for CCP50. These values use
`completion <= Batches.LT`; they are diagnostics and are not CCP feasibility.

## Emission check

All 803 fixed candidates have scenario-invariant emission. For each candidate,
mean, median, q90, minimum, and maximum emission are equal up to floating-point
summation representation. Emission differs between decisions because their
paths, modes, and flows differ. No emission-model change was made.

## Numerical/methodological observations

- Exact decision-fingerprint retention produces large fronts (803 candidates),
  so the raw Stage C table contains 4,015,000 rows.
- S30 and S50 runtimes are almost equal under the current vectorised and cached
  evaluator; scenario count alone is not a useful runtime predictor here.
- Paired seeds align stochastic operations initially, but fitness-dependent
  selection naturally causes later RNG call paths and populations to diverge.
- No claim comparing methods statistically is justified from one replicate.
