# Final formal EV versus CCP30 report

## Frozen formal experiment configuration

The frozen design used 30 paired replicates, population 300, 500 generations, α=0.9, direct expected-input EV, and active quantile-objective CCP30 with separate empirical ceil(0.9×30)-th order-statistic objectives. EV_r and CCP30_r shared algorithm seed r.

## Completion and validation

All 30 EV and all 30 CCP30 completion markers, configurations, candidate hashes, and paired seeds passed the immutable-input audit. Common held-out S5000 validation completed for 15,664 candidates in 855.203 s; every decision fingerprint was unchanged. Seed: `930001`. Digest: `9219266ad001420fffdcb93b3d135774ab893c9c8357fcdfbcf11520e29788cb`.

## Runtime summary

The 60 optimisation runs totalled 57095.058 s (951.584 ± 23.440 s; range 912.835–991.126 s). Stage C took 855.203 s. No optimisation or validation was rerun during post-processing.

## OOS run-level results and paired statistics

OOS-MEAN represents expected/risk-neutral performance; OOS-Q90 represents upper-tail/risk-aware performance. HV uses one pooled-candidate normalization per view and the fixed normalized reference point (1.2, 1.2, 1.2). IGD+ uses the corresponding global held-out nondominated front. Spacing is computed in the same normalized objective space. Detailed 60-row results and 30-row paired contrasts are in `metrics/`; descriptive and paired tests are in `statistics/`.

## OOS-MEAN

- hypervolume: EV 1.00344 ± 0.131127; CCP30 1.11263 ± 0.129648; W/T/L 7/0/23; Holm p=0.0120301; rank-biserial=-0.634409.
- igd_plus: EV 0.179151 ± 0.049333; CCP30 0.124456 ± 0.0514508; W/T/L 4/0/26; Holm p=0.000243962; rank-biserial=0.806452.
- spacing: EV 0.0129229 ± 0.00621781; CCP30 0.0113834 ± 0.00427961; W/T/L 12/0/18; Holm p=0.786367; rank-biserial=0.23871.
- nondominated_size: EV 110.533 ± 31.6846; CCP30 99.6 ± 30.7365; W/T/L 18/1/11; Holm p=0.659704; rank-biserial=0.294253.

## OOS-Q90

- hypervolume: EV 0.931844 ± 0.155553; CCP30 1.0202 ± 0.159431; W/T/L 9/0/21; Holm p=0.261317; rank-biserial=-0.406452.
- igd_plus: EV 0.196312 ± 0.0702537; CCP30 0.153938 ± 0.0584663; W/T/L 9/0/21; Holm p=0.111177; rank-biserial=0.488172.
- spacing: EV 0.0137271 ± 0.00704113; CCP30 0.0122724 ± 0.00517992; W/T/L 10/0/20; Holm p=0.786367; rank-biserial=0.234409.
- nondominated_size: EV 114.3 ± 31.6796; CCP30 106.133 ± 33.5556; W/T/L 17/0/13; Holm p=0.786367; rank-biserial=0.195699.

## Generalisation

EV training direct-expected-input → OOS mean: cost signed delta mean 1.2477% (mean APE 1.2477%); makespan signed delta mean 14.8041% (mean APE 14.8138%).

CCP30 training q90 → OOS q90: cost signed delta mean 1.0274% (mean APE 1.0675%); makespan signed delta mean 6.2132% (mean APE 6.2968%). Per-run and method summaries are in `metrics/generalisation_summary.csv`.

## Punctuality and emission diagnostics

Minimum-batch on-time probability relative to `Batches.LT` is retained only as a punctuality diagnostic (candidate median 0.0316, range 0.0000–0.1790); it is not CCP feasibility, reliability, or chance-constraint satisfaction. Emission is scenario-invariant under the current model (maximum within-candidate scenario span 0); its tiny floating-point training/OOS deltas are reported separately and are not treated as a generalisation-error discriminator.

## Scientific interpretation

The two held-out views answer different questions and must be interpreted separately. Under OOS-MEAN, CCP30 won 23/30 HV pairs and 26/30 IGD+ pairs; both differences survived Holm correction, so the evidence does not support an EV expected-performance advantage in this experiment. Under OOS-Q90, CCP30 won 21/30 pairs for both HV and IGD+, but neither comparison survived the eight-test Holm correction. Spacing was unresolved in both views. Thus the observed direction generally favours CCP30 for convergence/coverage, most clearly under mean evaluation, while tail-view inferential evidence is weaker. Nondominated-set size is descriptive, not intrinsically a quality score.

## Limitations

CCP30 uses active quantile-objective semantics: it minimizes three separate marginal empirical q90 objectives from only 30 training scenarios. It is not a joint chance constraint, and punctuality is not part of CCP feasibility. The global reference fronts are empirical fronts from the 15,664 validated formal candidates, not the unknown true Pareto fronts. The one common S5000 set supports paired comparisons but does not eliminate Monte Carlo error. Exact objective duplicates are deduplicated only for metric geometry; all nondominated provenance records are retained in the saved reference-front files.

## Reproducibility and checkpoints

Validation summary SHA-256: `4d4e28e984d572200ce10fe1ee2cf378279eea34f8df76b4980f92097b632adb`. Scenario-level SHA-256 recorded by the completed checkpoint (not reread): `9bb1c30802f78ab44920ae3bdcd969bfcd889d09343fe5719dfba5753d5ddda3`. Normalization bounds, epsilon handling, and HV reference are frozen in `metrics/normalisation.json`. Input audit checks candidate/configuration hashes for every run and verifies all 15,664 source IDs.
