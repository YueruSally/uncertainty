# CCP30 fixed-decision distribution similarity

This experiment compares nested training-sample objective distributions with one common OOS-5000 reference. It does not re-optimise decisions or filter solutions.

Primary metrics are Jensen-Shannon distance, normalised Wasserstein distance and the KS statistic. KL and the asymptotic two-sided KS p-value are supplementary. A p-value above 0.05 means that this test did not detect a difference; it does not prove equivalence.

| S | Objective | Median JS | Median normalised W1 | Median KS | Mean absolute mean difference (%) | KS p>0.05 | Mean absolute coverage gap (pp) |
|---|---|---:|---:|---:|---:|---:|---:|
| 30 | cost | 0.233826 | 0.184342 | 0.156133 | 0.342 | 98.0% | 5.246 |
| 30 | emission | 0.217602 | 0.199100 | 0.164100 | 0.017 | 97.7% | 3.406 |
| 30 | makespan | 0.219509 | 0.146640 | 0.130800 | 0.851 | 99.8% | 3.948 |
| 100 | cost | 0.136175 | 0.088828 | 0.086800 | 0.176 | 87.3% | 2.306 |
| 100 | emission | 0.131180 | 0.097292 | 0.079000 | 0.005 | 99.9% | 2.108 |
| 100 | makespan | 0.133254 | 0.081752 | 0.075400 | 0.549 | 99.7% | 1.619 |
| 300 | cost | 0.084067 | 0.061024 | 0.049967 | 0.120 | 96.8% | 1.520 |
| 300 | emission | 0.084282 | 0.062481 | 0.051667 | 0.004 | 95.9% | 1.404 |
| 300 | makespan | 0.082868 | 0.051788 | 0.043467 | 0.388 | 96.2% | 0.991 |

## Independent OOS-5000 reference baseline

The table below compares a second independent OOS sample with the common OOS reference. It estimates the non-zero distance expected from Monte Carlo sampling even when both samples use the same uncertainty model.

| Objective | Median JS | Median normalised W1 | Median KS | Mean absolute mean difference (%) | KS p>0.05 |
|---|---:|---:|---:|---:|---:|
| cost | 0.038045 | 0.040751 | 0.022600 | 0.096 | 99.2% |
| emission | 0.031509 | 0.025074 | 0.021600 | 0.001 | 100.0% |
| makespan | 0.036654 | 0.036579 | 0.020600 | 0.359 | 99.6% |

## Pre-declared practical assessment

Candidate: **S=100**

S=100 meets the declared coverage-gap tolerance and captures the required share of the distribution-distance improvement from S=30 to S=300.

This assessment is descriptive and conditional on the existing 263 Run-1 decisions. Distribution similarity alone does not prove that a separately re-optimised CCP model has equivalent solution quality.
