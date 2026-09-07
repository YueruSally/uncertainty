# Fixed CCP30 decision q90 stability

263 fixed decisions. All original/independent thresholds share the same OOS set.
This is conditional on the existing Run-1 decisions, not a repeated-optimisation experiment.
The original training sample selected these decisions; independent samples only re-estimate them.
Coverage concerns separate objective thresholds, not on-time or joint feasibility.
Seeds use nested prefixes of a master sample; changing the maximum S changes the random stream.
Solution-level observations share scenarios and are correlated. Seed SD below is descriptive, not a confidence interval.

| S | Objective | Seeds | Mean coverage | SD of seed means | Mean absolute gap (pp) |
|---|---|---|---|---|---|
| 30 | cost | 10 | 88.17% | 5.71% | 5.246 |
| 30 | emission | 10 | 87.19% | 3.68% | 3.406 |
| 30 | makespan | 10 | 88.01% | 4.03% | 3.948 |
| 100 | cost | 10 | 89.68% | 2.38% | 2.306 |
| 100 | emission | 10 | 88.87% | 2.34% | 2.108 |
| 100 | makespan | 10 | 89.46% | 1.38% | 1.619 |
| 300 | cost | 10 | 89.77% | 1.80% | 1.520 |
| 300 | emission | 10 | 89.30% | 1.56% | 1.404 |
| 300 | makespan | 10 | 90.35% | 0.62% | 0.991 |

90% is the nominal target. ceil(0.9*S)/(S+1) is only an expectation reference for an independently fixed decision with continuous iid outcomes; it is not a guarantee or a universal correction. Timetable outcomes can have ties.
Neither closeness to 90% nor improved estimates demonstrates that independently re-optimised CCP models improve.
