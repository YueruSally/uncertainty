# CCP30 fixed-decision distribution similarity

This experiment compares nested training-sample objective distributions with one common OOS-5000 reference. It does not re-optimise decisions or filter solutions.

Primary metrics are Jensen-Shannon distance, normalised Wasserstein distance and the KS statistic. KL is supplementary because it is directional and sensitive to histogram construction; the program uses OOS-defined common bins and Jeffreys smoothing.

| S | Objective | Median JS | Median normalised W1 | Median KS | Mean absolute coverage gap (pp) |
|---|---|---:|---:|---:|---:|
| 30 | cost | 0.233826 | 0.184342 | 0.156133 | 5.246 |
| 30 | emission | 0.217602 | 0.199100 | 0.164100 | 3.406 |
| 30 | makespan | 0.219509 | 0.146640 | 0.130800 | 3.948 |
| 100 | cost | 0.136175 | 0.088828 | 0.086800 | 2.306 |
| 100 | emission | 0.131180 | 0.097292 | 0.079000 | 2.108 |
| 100 | makespan | 0.133254 | 0.081752 | 0.075400 | 1.619 |
| 300 | cost | 0.084067 | 0.061024 | 0.049967 | 1.520 |
| 300 | emission | 0.084282 | 0.062481 | 0.051667 | 1.404 |
| 300 | makespan | 0.082868 | 0.051788 | 0.043467 | 0.991 |

## Pre-declared practical assessment

Candidate: **S=100**

S=100 meets the declared coverage-gap tolerance and captures the required share of the distribution-distance improvement from S=30 to S=300.

This assessment is descriptive and conditional on the existing 263 Run-1 decisions. Distribution similarity alone does not prove that a separately re-optimised CCP model has equivalent solution quality.
