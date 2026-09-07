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
| M1 | Boundary bottleneck congestion | 0.265405 | 0.133681 | 0.266198 | 0.241399 | 0.1222 | 0.0593332 | 0.75 | core | 0 |
| M2 | Corridor/link disruption | 0.140317 | 0.0772695 | 0.304447 | 0.251662 | 0.107012 | 0.0555841 | 0.5 | core | 0 |

## Combination Scenarios

| scenario_id | scenario_type | candidate_label | level | intensity | runs | cost_delta_rel_mean | time_delta_rel_mean | emission_delta_rel_mean | on_time_drop_mean | on_time_rate_mean | late_h_per_teu_delta_mean | p95_lateness_delta_h_mean | route_change_ratio_mean | path_route_change_ratio_mean | rerouted_volume_share_mean | border_share_shift_mean | recovery_cost_delta_mean | recovery_cost_delta_rel_mean | infeasible_teu_mean | total_teu_mean | feasible_ratio_mean | runtime_s_mean | kpi_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M1+M2 | combination | Boundary bottleneck congestion + Corridor/link disruption | r3 | 3 | 4 | 0.644101 | 0.809601 | -0.1271 | 0.219467 | 0.774921 | 22.3939 | 224.631 | 0.684443 | 0.684443 | 0.284831 | 0.469503 | 7.15058e+06 | 0.644101 | 0 | 4455 | 0 | 18.1609 | 1.33709 |
| M1+M2 | combination | Boundary bottleneck congestion + Corridor/link disruption | r5 | 5 | 4 | 116708 | 237270 | -0.140568 | 0.307164 | 0.687225 | 4.71793e+06 | 3.08221e+07 | 0.679423 | 0.679423 | 0.300877 | 0.45035 | 1.29311e+12 | 116708 | 0 | 4455 | 0 | 17.4606 | 237270 |

## Single-Factor and Level Summary

| scenario_id | scenario_type | candidate_label | level | intensity | runs | cost_delta_rel_mean | time_delta_rel_mean | emission_delta_rel_mean | on_time_drop_mean | on_time_rate_mean | late_h_per_teu_delta_mean | p95_lateness_delta_h_mean | route_change_ratio_mean | path_route_change_ratio_mean | rerouted_volume_share_mean | border_share_shift_mean | recovery_cost_delta_mean | recovery_cost_delta_rel_mean | infeasible_teu_mean | total_teu_mean | feasible_ratio_mean | runtime_s_mean | kpi_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M1+M2 | combination | Boundary bottleneck congestion + Corridor/link disruption | r3 | 3 | 4 | 0.644101 | 0.809601 | -0.1271 | 0.219467 | 0.774921 | 22.3939 | 224.631 | 0.684443 | 0.684443 | 0.284831 | 0.469503 | 7.15058e+06 | 0.644101 | 0 | 4455 | 0 | 18.1609 | 1.33709 |
| M1+M2 | combination | Boundary bottleneck congestion + Corridor/link disruption | r5 | 5 | 4 | 116708 | 237270 | -0.140568 | 0.307164 | 0.687225 | 4.71793e+06 | 3.08221e+07 | 0.679423 | 0.679423 | 0.300877 | 0.45035 | 1.29311e+12 | 116708 | 0 | 4455 | 0 | 17.4606 | 237270 |
| M1 | single | Boundary bottleneck congestion | r1 | 1 | 4 | 0.0161124 | 0.0019482 | 0.0540093 | 0.00123457 | 0.993154 | -0.213991 | 0 | 0.681436 | 0.681436 | 0.234163 | 0.127194 | 180033 | 0.0161124 | 0 | 4455 | 0.570833 | 26.571 | 0.0161124 |
| M1 | single | Boundary bottleneck congestion | r2 | 2 | 4 | 0.0366859 | 0.0353034 | 0.0721253 | 0.0232889 | 0.971099 | 0.16531 | 0 | 0.712183 | 0.712183 | 0.266198 | 0.141851 | 408083 | 0.0366859 | 0 | 4455 | 0 | 26.6809 | 0.0366859 |
| M1 | single | Boundary bottleneck congestion | r3 | 3 | 4 | 0.0446705 | 0.0286901 | -0.0400958 | 0.10625 | 0.888138 | 1.64088 | 10.2715 | 0.735887 | 0.735887 | 0.236405 | 0.247747 | 495885 | 0.0446705 | 0 | 4455 | 0 | 23.4653 | 0.10625 |
| M1 | single | Boundary bottleneck congestion | r4 | 4 | 4 | 0.1222 | 0.00572411 | -0.190865 | 0.191469 | 0.802919 | 5.8706 | 44.588 | 0.654331 | 0.654331 | 0.24476 | 0.395988 | 1.3594e+06 | 0.1222 | 0 | 4455 | 0 | 17.2673 | 0.265405 |
| M1 | single | Boundary bottleneck congestion | r5 | 5 | 4 | 0.0769974 | 0.00572411 | -0.182488 | 0.161132 | 0.833257 | 4.63772 | 40.9836 | 0.614761 | 0.614761 | 0.225467 | 0.381398 | 857116 | 0.0769974 | 0 | 4455 | 0 | 17.0968 | 0.24395 |
| M2 | single | Corridor/link disruption | r1 | 1 | 4 | 0.0313302 | 0.00636167 | 0.0864104 | 0.00656566 | 0.987823 | -0.150795 | 0 | 0.703418 | 0.703418 | 0.21813 | 0.188857 | 347238 | 0.0313302 | 0 | 4455 | 0.25625 | 26.7388 | 0.0313302 |
| M2 | single | Corridor/link disruption | r2 | 2 | 4 | 0.00930342 | 0 | 0.0402594 | 0.00344633 | 0.990942 | -0.213827 | 0 | 0.666095 | 0.666095 | 0.22909 | 0.167385 | 101548 | 0.00930342 | 0 | 4455 | 0.5 | 26.1893 | 0.00930342 |
| M2 | single | Corridor/link disruption | r3 | 3 | 4 | 0.107012 | 0.00548313 | 0.0217315 | 0.085379 | 0.909009 | 2.73365 | 21.0662 | 0.677492 | 0.677492 | 0.304447 | 0.421392 | 1.1938e+06 | 0.107012 | 0 | 4455 | 0 | 20.9709 | 0.125394 |
| M2 | single | Corridor/link disruption | r4 | 4 | 4 | 0.0429161 | 0.0064431 | -0.0034886 | 0.0800032 | 0.914385 | 1.5542 | 8.125 | 0.679542 | 0.679542 | 0.232935 | 0.292779 | 475952 | 0.0429161 | 0 | 4455 | 0 | 20.1452 | 0.0800032 |
| M2 | single | Corridor/link disruption | r5 | 5 | 4 | 0.0873585 | 0.00211316 | -0.0130843 | 0.121802 | 0.872586 | 3.19358 | 23.5732 | 0.662372 | 0.662372 | 0.273706 | 0.355848 | 966832 | 0.0873585 | 0 | 4455 | 0 | 18.9136 | 0.140317 |
