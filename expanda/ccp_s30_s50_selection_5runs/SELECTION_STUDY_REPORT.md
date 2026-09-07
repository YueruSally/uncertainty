# CCP S30/S50 selection study report

This study contains exactly five new paired full-size replicates (population 300,
500 generations, alpha 0.90), with no EV runs. It is not the formal experiment.
The common S=5000 validation set (seed 879999) is selection-only and must not be
reused as the formal experiment's final validation set.

| Rep | Algorithm / training seed | Runtime S30 / S50 (s) | Candidates S30 / S50 | HV S30 / S50 | IGD+ S30 / S50 | Median cost-q90 APE S30 / S50 (%) | Median makespan-q90 APE S30 / S50 (%) |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | 871001 / 872001 | 981.905 / 1001.814 | 281 / 262 | 1.113780 / 1.179903 | 0.020501 / 0.031123 | 0.2490 / 0.1005 | 0.9750 / 0.9505 |
| 2 | 871002 / 872002 | 967.050 / 975.106 | 261 / 259 | 0.924620 / 1.351363 | 0.167971 / 0.000051 | 1.4112 / 0.8919 | 3.1609 / 1.7619 |
| 3 | 871003 / 872003 | 979.268 / 969.757 | 278 / 274 | 1.017285 / 0.418644 | 0.000000 / 0.357638 | 0.5646 / 1.0532 | 3.9344 / 4.0834 |
| 4 | 871004 / 872004 | 972.212 / 980.824 | 261 / 276 | 1.317661 / 0.964532 | 0.000650 / 0.096925 | 0.3953 / 0.4602 | 4.3049 / 5.3027 |
| 5 | 871005 / 872005 | 999.408 / 992.775 | 282 / 273 | 1.344058 / 1.343025 | 0.007361 / 0.031857 | 1.5488 / 1.2612 | 7.8895 / 7.5153 |

S30 has higher HV in three replicates and lower IGD+ in four. S50 has lower
median absolute cost-q90 and makespan-q90 generalisation error in three
replicates each. Runtime is effectively similar: the median absolute paired
difference is 0.886%, below the predeclared 5% engineering threshold.

The evidence is mixed, but it leans toward S30 for the formal study because the
primary validated Pareto-set metrics favour S30 more consistently, while S50's
generalisation-error advantage occurs only 3/5 times and does not translate into
consistent Pareto quality. This is an engineering selection, not a claim of
statistical significance.

All 2,707 candidates had unique fingerprints within their source run, and all
13,535,000 candidate-scenario evaluations preserved those fingerprints. Fixed
candidate emission was scenario-invariant in every case. Punctuality values are
retained only as diagnostics and were not used as CCP feasibility.
