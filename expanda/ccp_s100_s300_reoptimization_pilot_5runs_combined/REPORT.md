# CCP100 versus CCP300 re-optimisation pilot

This descriptive pilot uses 5 paired repetitions, population 100, 100 generations and one fresh common OOS-5000 sample. It is not a formal inferential experiment.

All original training Pareto candidates are retained. OOS nondominated fronts are used only for HV/IGD+ geometry and do not filter the saved candidate set.

| Replicate | Method | Runtime (s) | Candidates | Mean HV | Mean IGD+ | q90 HV | q90 IGD+ | Cost cov. | Emission cov. | Makespan cov. | Mean abs. gap (pp) |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | CCP100 | 69.51 | 27 | 0.595373 | 0.328806 | 0.647894 | 0.275007 | 86.74% | 93.75% | 83.32% | 4.563 |
| 1 | CCP300 | 79.40 | 79 | 1.106827 | 0.060575 | 1.197531 | 0.033883 | 88.93% | 89.58% | 89.20% | 0.835 |
| 2 | CCP100 | 71.03 | 30 | 0.657247 | 0.262001 | 0.646350 | 0.208419 | 80.28% | 84.52% | 83.65% | 7.182 |
| 2 | CCP300 | 80.51 | 38 | 1.224198 | 0.025527 | 1.049837 | 0.099194 | 87.70% | 89.66% | 87.65% | 1.739 |
| 3 | CCP100 | 69.33 | 53 | 0.691873 | 0.245060 | 0.798116 | 0.145097 | 83.52% | 88.25% | 87.60% | 3.564 |
| 3 | CCP300 | 80.46 | 71 | 0.567705 | 0.391174 | 0.684465 | 0.230313 | 90.51% | 85.72% | 89.72% | 1.857 |
| 4 | CCP100 | 69.56 | 59 | 0.850713 | 0.144333 | 0.901847 | 0.078031 | 84.26% | 80.47% | 87.48% | 5.951 |
| 4 | CCP300 | 78.85 | 64 | 0.517697 | 0.393308 | 0.609182 | 0.229498 | 85.57% | 88.00% | 89.46% | 2.537 |
| 5 | CCP100 | 68.43 | 24 | 0.418173 | 0.436691 | 0.506547 | 0.288052 | 91.72% | 86.43% | 89.71% | 1.964 |
| 5 | CCP300 | 78.72 | 29 | 0.418472 | 0.472651 | 0.507983 | 0.318502 | 89.84% | 91.27% | 89.58% | 0.898 |

## Descriptive paired win counts

| Metric | Preferred | CCP100 wins | CCP300 wins | Ties |
|---|---|---:|---:|---:|
| oos_mean_hypervolume | higher | 2 | 3 | 0 |
| oos_mean_igd_plus | lower | 3 | 2 | 0 |
| oos_q90_hypervolume | higher | 2 | 3 | 0 |
| oos_q90_igd_plus | lower | 3 | 2 | 0 |
| all_objectives_mean_absolute_coverage_gap_pp | lower | 0 | 5 | 0 |
| optimisation_runtime_seconds | lower | 5 | 0 | 0 |

With only 5 paired repetitions, win counts and differences are descriptive. Do not report significance tests or claim that either sample size is definitively superior.
