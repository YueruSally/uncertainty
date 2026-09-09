# CCP100 versus CCP300 re-optimisation pilot

This descriptive pilot uses 2 paired repetitions, population 100, 100 generations and one fresh common OOS-5000 sample. It is not a formal inferential experiment.

All original training Pareto candidates are retained. OOS nondominated fronts are used only for HV/IGD+ geometry and do not filter the saved candidate set.

| Replicate | Method | Runtime (s) | Candidates | Mean HV | Mean IGD+ | q90 HV | q90 IGD+ | Cost cov. | Emission cov. | Makespan cov. | Mean abs. gap (pp) |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | CCP100 | 69.56 | 59 | 1.145276 | 0.024542 | 1.373985 | 0.000000 | 84.26% | 80.47% | 87.48% | 5.951 |
| 4 | CCP300 | 78.85 | 64 | 0.654263 | 0.163309 | 0.785440 | 0.239576 | 85.57% | 88.00% | 89.46% | 2.537 |
| 5 | CCP100 | 68.43 | 24 | 0.471383 | 0.270932 | 0.595412 | 0.366466 | 91.72% | 86.43% | 89.71% | 1.964 |
| 5 | CCP300 | 78.72 | 29 | 0.424924 | 0.349558 | 0.537102 | 0.441714 | 89.84% | 91.27% | 89.58% | 0.898 |

## Descriptive paired win counts

| Metric | Preferred | CCP100 wins | CCP300 wins | Ties |
|---|---|---:|---:|---:|
| oos_mean_hypervolume | higher | 2 | 0 | 0 |
| oos_mean_igd_plus | lower | 2 | 0 | 0 |
| oos_q90_hypervolume | higher | 2 | 0 | 0 |
| oos_q90_igd_plus | lower | 2 | 0 | 0 |
| all_objectives_mean_absolute_coverage_gap_pp | lower | 0 | 2 | 0 |
| optimisation_runtime_seconds | lower | 2 | 0 | 0 |

With only 2 paired repetitions, win counts and differences are descriptive. Do not report significance tests or claim that either sample size is definitively superior.
