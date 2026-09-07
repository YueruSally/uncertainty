# Refined Operational Uncertainty Experiments

This second-stage experiment follows the coarse screening result. It uses a fixed path-library topology across baseline and scenarios, so route-change metrics reflect optimization responses rather than repeated random path generation.

## Run Configuration

- Mode: `refined`
- Data: `/home/lunet/cokt2/uncertainty/expanda/data/data_expanded.xlsx`
- Expected batches: `40`
- Population: `120`
- Generations: `80`
- Seeds: `2026, 7, 99, 1000`
- Single-factor candidates: `M1, M2`
- Fine levels: `r1, r2, r3, r4, r5`
- Combination levels: `r3, r5`

## Interpretation

- M1 is boundary bottleneck congestion: border capacity pressure creates endogenous delay through a BPR function.
- M2 is corridor/link disruption.
- M1+M2 is the main cascade scenario: disruption pushes flow toward remaining borders and amplifies bottleneck congestion.
- On-time delivery is retained as a service KPI. Recovery cost is the main economic decision indicator; rerouted volume and border-share shift are supporting flow-reallocation indicators.
- B1 and B2 can be added as benchmarks, because line-haul travel-time and demand uncertainty are established uncertainty classes.
- C3 is not included by default because coarse screening showed the weakest KPI impact.

## Refined Ranking

| scenario_id | candidate_label | max_kpi_score | mean_kpi_score | max_rerouted_volume_share | mean_rerouted_volume_share | max_recovery_cost_delta_rel | mean_recovery_cost_delta_rel | monotonicity_score | recommended_role | role_priority |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M1 | Boundary bottleneck congestion | 0.265981 | 0.138533 | 0.289298 | 0.266038 | 0.0822845 | 0.0317444 | 0.75 | core | 0 |
| M2 | Corridor/link disruption | 0.148004 | 0.0720808 | 0.324848 | 0.282917 | 0.0708213 | 0.0284257 | 0.5 | core | 0 |

## Combination Scenarios

| scenario_id | scenario_type | candidate_label | level | intensity | runs | cost_delta_rel_mean | time_delta_rel_mean | emission_delta_rel_mean | on_time_drop_mean | on_time_rate_mean | late_h_per_teu_delta_mean | p95_lateness_delta_h_mean | route_change_ratio_mean | path_route_change_ratio_mean | rerouted_volume_share_mean | border_share_shift_mean | recovery_cost_delta_mean | recovery_cost_delta_rel_mean | infeasible_teu_mean | total_teu_mean | feasible_ratio_mean | runtime_s_mean | kpi_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M1+M2 | combination | Boundary bottleneck congestion + Corridor/link disruption | r3 | 3 | 4 | 0.0982919 | 0.00373282 | -0.253199 | 0.190283 | 0.809717 | 6.19104 | 42.9404 | 0.687319 | 0.687319 | 0.270679 | 0.439067 | 1.11456e+06 | 0.0982919 | 0 | 4455 | 0 | 17.8232 | 0.25796 |
| M1+M2 | combination | Boundary bottleneck congestion + Corridor/link disruption | r5 | 5 | 4 | 0.130955 | 0.00373282 | -0.225388 | 0.241152 | 0.758848 | 7.04132 | 43.4704 | 0.684173 | 0.684173 | 0.275269 | 0.439067 | 1.48636e+06 | 0.130955 | 0 | 4455 | 0 | 17.8914 | 0.293389 |

## Single-Factor and Level Summary

| scenario_id | scenario_type | candidate_label | level | intensity | runs | cost_delta_rel_mean | time_delta_rel_mean | emission_delta_rel_mean | on_time_drop_mean | on_time_rate_mean | late_h_per_teu_delta_mean | p95_lateness_delta_h_mean | route_change_ratio_mean | path_route_change_ratio_mean | rerouted_volume_share_mean | border_share_shift_mean | recovery_cost_delta_mean | recovery_cost_delta_rel_mean | infeasible_teu_mean | total_teu_mean | feasible_ratio_mean | runtime_s_mean | kpi_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M1+M2 | combination | Boundary bottleneck congestion + Corridor/link disruption | r3 | 3 | 4 | 0.0982919 | 0.00373282 | -0.253199 | 0.190283 | 0.809717 | 6.19104 | 42.9404 | 0.687319 | 0.687319 | 0.270679 | 0.439067 | 1.11456e+06 | 0.0982919 | 0 | 4455 | 0 | 17.8232 | 0.25796 |
| M1+M2 | combination | Boundary bottleneck congestion + Corridor/link disruption | r5 | 5 | 4 | 0.130955 | 0.00373282 | -0.225388 | 0.241152 | 0.758848 | 7.04132 | 43.4704 | 0.684173 | 0.684173 | 0.275269 | 0.439067 | 1.48636e+06 | 0.130955 | 0 | 4455 | 0 | 17.8914 | 0.293389 |
| M1 | single | Boundary bottleneck congestion | r1 | 1 | 4 | -0.0134222 | -0.00402858 | -0.0477393 | 0.00684624 | 0.993154 | 0.00554056 | 0 | 0.699531 | 0.699531 | 0.263804 | 0.153227 | -157694 | -0.0134222 | 0 | 4455 | 0.75 | 27.4111 | 0.0134222 |
| M1 | single | Boundary bottleneck congestion | r2 | 2 | 4 | 0.00502517 | 0.0244387 | -0.000981922 | 0.0349306 | 0.965069 | 0.151051 | 0.0185714 | 0.784576 | 0.784576 | 0.234773 | 0.172446 | 53265 | 0.00502517 | 0 | 4455 | 0.025 | 27.441 | 0.0349306 |
| M1 | single | Boundary bottleneck congestion | r3 | 3 | 4 | 0.0177063 | 0.047299 | -0.101902 | 0.121932 | 0.878068 | 1.85156 | 14.926 | 0.734904 | 0.734904 | 0.270066 | 0.322582 | 194674 | 0.0177063 | 0 | 4455 | 0 | 24.39 | 0.121932 |
| M1 | single | Boundary bottleneck congestion | r4 | 4 | 4 | 0.0822845 | 0.00267857 | -0.246838 | 0.179018 | 0.820982 | 5.62405 | 44.6849 | 0.668901 | 0.668901 | 0.272248 | 0.431975 | 932595 | 0.0822845 | 0 | 4455 | 0 | 17.3769 | 0.265981 |
| M1 | single | Boundary bottleneck congestion | r5 | 5 | 4 | 0.0671282 | 0.00373282 | -0.204826 | 0.184198 | 0.815802 | 4.9399 | 43.0753 | 0.708669 | 0.708669 | 0.289298 | 0.436987 | 756381 | 0.0671282 | 0 | 4455 | 0 | 17.8801 | 0.256401 |
| M2 | single | Corridor/link disruption | r1 | 1 | 4 | 0.00720956 | 0.00494185 | -0.000358418 | 0.021951 | 0.978049 | 0.179565 | 0 | 0.730613 | 0.730613 | 0.277276 | 0.221005 | 77575.8 | 0.00720956 | 0 | 4455 | 0.25 | 28.8275 | 0.021951 |
| M2 | single | Corridor/link disruption | r2 | 2 | 4 | -0.0120938 | -0.0185924 | -0.013822 | 0.00684624 | 0.993154 | 0.000508578 | 0 | 0.680839 | 0.680839 | 0.227835 | 0.15075 | -138749 | -0.0120938 | 0 | 4455 | 0.75 | 25.7218 | 0.0185924 |
| M2 | single | Corridor/link disruption | r3 | 3 | 4 | 0.0475074 | 0.0290436 | -0.0844166 | 0.117146 | 0.882854 | 2.66242 | 17.9719 | 0.687849 | 0.687849 | 0.317386 | 0.362043 | 534886 | 0.0475074 | 0 | 4455 | 0 | 20.6188 | 0.117146 |
| M2 | single | Corridor/link disruption | r4 | 4 | 4 | 0.0286838 | 0.00449878 | -0.00515656 | 0.0547102 | 0.94529 | 1.1529 | 6.2381 | 0.696518 | 0.696518 | 0.267242 | 0.211961 | 318298 | 0.0286838 | 0 | 4455 | 0 | 20.6098 | 0.0547102 |
| M2 | single | Corridor/link disruption | r5 | 5 | 4 | 0.0708213 | -0.000373162 | -0.0877752 | 0.148004 | 0.851996 | 3.52641 | 23.5918 | 0.719187 | 0.719187 | 0.324848 | 0.395063 | 809636 | 0.0708213 | 0 | 4455 | 0 | 19.2631 | 0.148004 |
