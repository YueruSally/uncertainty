# Operational Uncertainty Coarse Screening

This experiment identifies decision-relevant operational uncertainty candidates for the subsequent stochastic/robust routing model.

## Run Configuration

- Data: `/home/lunet/cokt2/uncertainty/expanda/data/data_expanded.xlsx`
- Expected batches: `40`
- Population: `90`
- Generations: `50`
- Seeds: `2026, 7, 99`
- Levels: `low, medium, high`
- KPI threshold: `0.03`
- Route-change threshold: `0.1`

## Candidate Pool

| ID | Candidate | Mechanism | Literature class |
|---|---|---|---|
| C1 | Line-haul travel time uncertainty | realized_time | transit time / travel time reliability |
| C2 | Border processing and break-of-gauge time uncertainty | realized_time | service/transfer time + queueing delay |
| C3 | Terminal/transshipment handling time uncertainty | realized_time | transfer time / terminal handling time |
| C4 | Service capacity and slot availability uncertainty | service_feasibility | capacity uncertainty |
| C5 | Shipment volume uncertainty | shipment_planning | demand uncertainty |
| C6 | Corridor/link/node disruption uncertainty | network_disruption | link-node failure / disruption |

## Ranking

| candidate | candidate_label | max_kpi_score | max_route_change_ratio | retained_levels | retain_for_refined_screening |
| --- | --- | --- | --- | --- | --- |
| C5 | Shipment volume uncertainty | 0.228625 | 0.881727 | 3 | True |
| C6 | Corridor/link/node disruption uncertainty | 0.137689 | 0.839919 | 3 | True |
| C1 | Line-haul travel time uncertainty | 0.0502696 | 0.909995 | 3 | True |
| C4 | Service capacity and slot availability uncertainty | 0.0672364 | 0.888762 | 3 | True |
| C2 | Border processing and break-of-gauge time uncertainty | 0.0303779 | 0.884625 | 3 | True |
| C3 | Terminal/transshipment handling time uncertainty | 0.0105385 | 0.855144 | 3 | True |

## Selection Logic

A candidate is retained if it materially affects operational KPIs and/or materially changes routing decisions. Factors that only affect capacity-induced waiting should not be entered into the final model together with their realized delay representation, to avoid double counting the same mechanism.

## Level Summary

| candidate | candidate_name | candidate_label | level | runs | cost_delta_rel_mean | time_delta_rel_mean | emission_delta_rel_mean | on_time_drop_mean | late_h_per_teu_delta_mean | p95_lateness_delta_h_mean | route_change_ratio_mean | infeasible_teu_mean | feasible_ratio_mean | runtime_s_mean | kpi_score | kpi_material | decision_material | retain_for_refined_screening | screening_class |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C1 | line_haul_travel_time | Line-haul travel time uncertainty | high | 3 | 0.0438954 | 0.000294886 | 0.0602488 | -0.00135927 | 0.575608 | 0 | 0.872 | 0 | 0.333333 | 13.9269 | 0.0438954 | True | True | True | core_decision_relevant |
| C1 | line_haul_travel_time | Line-haul travel time uncertainty | low | 3 | -0.00461817 | 0.00055359 | -0.0267887 | 0.0210749 | 0.422848 | 0.0213183 | 0.829227 | 0 | 0 | 12.8961 | 0.0210749 | False | True | True | strategy_shift |
| C1 | line_haul_travel_time | Line-haul travel time uncertainty | medium | 3 | 0.0502696 | 0.0225238 | 0.0680576 | 0.0143659 | 0.543156 | 0 | 0.909995 | 0 | 0.0703704 | 13.6851 | 0.0502696 | True | True | True | core_decision_relevant |
| C2 | border_processing_time | Border processing and break-of-gauge time uncertainty | high | 3 | 0.0154089 | -0.00819383 | 0.0227898 | 0.00185344 | 0.128699 | 0 | 0.884625 | 0 | 0.333333 | 12.9928 | 0.0154089 | False | True | True | strategy_shift |
| C2 | border_processing_time | Border processing and break-of-gauge time uncertainty | low | 3 | 0.0202733 | 0.00740295 | 0.028374 | 0.0303779 | 0.189095 | 0.495714 | 0.864365 | 0 | 0.333333 | 12.715 | 0.0303779 | True | True | True | core_decision_relevant |
| C2 | border_processing_time | Border processing and break-of-gauge time uncertainty | medium | 3 | 0.0163696 | 0.00273535 | 0.0429987 | -0.00852974 | 0.0651523 | 0 | 0.849442 | 0 | 0.666667 | 13.3541 | 0.0163696 | False | True | True | strategy_shift |
| C3 | terminal_transshipment_time | Terminal/transshipment handling time uncertainty | high | 3 | -0.00355967 | 0.000108573 | -0.0068623 | 0.00649257 | 0.0523126 | 0 | 0.855144 | 0 | 0 | 13.7478 | 0.00649257 | False | True | True | strategy_shift |
| C3 | terminal_transshipment_time | Terminal/transshipment handling time uncertainty | low | 3 | -0.0105385 | 0.00204591 | -0.0174429 | -0.00142162 | 0.149414 | 0 | 0.830065 | 0 | 0.333333 | 13.5253 | 0.0105385 | False | True | True | strategy_shift |
| C3 | terminal_transshipment_time | Terminal/transshipment handling time uncertainty | medium | 3 | -0.0103745 | -0.00187695 | -0.0211947 | 0.00688365 | 0.0682142 | 0 | 0.827867 | 0 | 0 | 13.5464 | 0.0103745 | False | True | True | strategy_shift |
| C4 | capacity_slot_availability | Service capacity and slot availability uncertainty | high | 3 | 0.0242328 | 0.0672364 | 0.00836202 | 0.0558536 | 0.617481 | 1.74972 | 0.888762 | 0 | 0 | 13.6651 | 0.0672364 | True | True | True | core_decision_relevant |
| C4 | capacity_slot_availability | Service capacity and slot availability uncertainty | low | 3 | 0.00887129 | -0.00502623 | 0.021292 | 0.00223639 | 0.0838998 | 0 | 0.836679 | 0 | 0 | 12.6369 | 0.00887129 | False | True | True | strategy_shift |
| C4 | capacity_slot_availability | Service capacity and slot availability uncertainty | medium | 3 | 0.0287888 | 0.0062844 | 0.0202408 | 0.0487867 | 0.62825 | 0.532262 | 0.886359 | 0 | 0 | 12.9116 | 0.0487867 | True | True | True | core_decision_relevant |
| C5 | shipment_volume | Shipment volume uncertainty | high | 3 | 0.228625 | -0.0112909 | 0.223596 | 0.00889938 | 0.21469 | 0 | 0.862608 | 0 | 0 | 13.0904 | 0.228625 | True | True | True | core_decision_relevant |
| C5 | shipment_volume | Shipment volume uncertainty | low | 3 | 0.0639759 | 0.0152492 | 0.0891621 | 0.00520474 | 0.183579 | 0 | 0.845439 | 0 | 0.333333 | 12.8453 | 0.0639759 | True | True | True | core_decision_relevant |
| C5 | shipment_volume | Shipment volume uncertainty | medium | 3 | 0.180415 | 0.00191257 | 0.237451 | -0.0122596 | -0.0188584 | 0 | 0.881727 | 0 | 0.666667 | 13.7339 | 0.180415 | True | True | True | core_decision_relevant |
| C6 | corridor_link_node_disruption | Corridor/link/node disruption uncertainty | high | 3 | 0.0532462 | 0.00848223 | -0.0545595 | 0.137689 | 2.72195 | 16.7773 | 0.815837 | 0 | 0 | 10.9703 | 0.137689 | True | True | True | core_decision_relevant |
| C6 | corridor_link_node_disruption | Corridor/link/node disruption uncertainty | low | 3 | 0.0139275 | 0.00741958 | 0.0279707 | 0.0196409 | 0.183883 | 0.0247619 | 0.839919 | 0 | 0 | 12.4858 | 0.0196409 | False | True | True | strategy_shift |
| C6 | corridor_link_node_disruption | Corridor/link/node disruption uncertainty | medium | 3 | 0.0255229 | 0.00893196 | -0.12887 | 0.0762439 | 2.73788 | 18.0971 | 0.823933 | 0 | 0.233333 | 9.78201 | 0.114078 | True | True | True | core_decision_relevant |
