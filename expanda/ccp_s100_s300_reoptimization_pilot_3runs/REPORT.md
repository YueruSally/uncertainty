# CCP100 versus CCP300 re-optimisation pilot

This descriptive pilot uses 3 paired repetitions, population 100, 100 generations and one fresh common OOS-5000 sample. It is not a formal inferential experiment.

All original training Pareto candidates are retained. OOS nondominated fronts are used only for HV/IGD+ geometry and do not filter the saved candidate set.

| Replicate | Method | Runtime (s) | Candidates | Mean HV | Mean IGD+ | q90 HV | q90 IGD+ | Cost cov. | Emission cov. | Makespan cov. | Mean abs. gap (pp) |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | CCP100 | 69.51 | 27 | 0.595373 | 0.332701 | 0.633143 | 0.268758 | 86.74% | 93.75% | 83.32% | 4.563 |
| 1 | CCP300 | 79.40 | 79 | 1.106827 | 0.060658 | 1.146041 | 0.026850 | 88.93% | 89.58% | 89.20% | 0.835 |
| 2 | CCP100 | 71.03 | 30 | 0.657247 | 0.266180 | 0.541536 | 0.220891 | 80.28% | 84.52% | 83.65% | 7.182 |
| 2 | CCP300 | 80.51 | 38 | 1.224198 | 0.025794 | 1.023517 | 0.096730 | 87.70% | 89.66% | 87.65% | 1.739 |
| 3 | CCP100 | 69.33 | 53 | 0.691873 | 0.248342 | 0.719777 | 0.139717 | 83.52% | 88.25% | 87.60% | 3.564 |
| 3 | CCP300 | 80.46 | 71 | 0.567705 | 0.391078 | 0.623809 | 0.220863 | 90.51% | 85.72% | 89.72% | 1.857 |

## Descriptive paired win counts

| Metric | Preferred | CCP100 wins | CCP300 wins | Ties |
|---|---|---:|---:|---:|
| oos_mean_hypervolume | higher | 1 | 2 | 0 |
| oos_mean_igd_plus | lower | 1 | 2 | 0 |
| oos_q90_hypervolume | higher | 1 | 2 | 0 |
| oos_q90_igd_plus | lower | 1 | 2 | 0 |
| all_objectives_mean_absolute_coverage_gap_pp | lower | 0 | 3 | 0 |
| optimisation_runtime_seconds | lower | 3 | 0 | 0 |

With only three paired repetitions, win counts and differences are descriptive. Do not report significance tests or claim that either sample size is definitively superior.
