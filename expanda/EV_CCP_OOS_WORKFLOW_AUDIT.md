# EV/CCP out-of-sample workflow audit

## Workflow before this change

- `baseline_uncertainty.py` performs NSGA-II optimisation in its main routine.
  `evaluate_individual()` propagates every decision through the active frozen
  scenario set and `aggregate_scenario_objectives()` reduces the cost,
  emission, and makespan arrays.
- CCP fitness is the separate empirical order statistic for each objective at
  index `ceil(alpha*S)-1`. Hard feasibility currently covers missing
  allocations, missing timetables, arc capacity, and border-node capacity.
  Per-batch on-time probabilities are diagnostics, not hard constraints.
- An `ev` mode existed, but it optimised a Monte Carlo sample mean. That is not
  the direct expected-input EV requested by the revised design.
- The candidate-pool driver `run_ccp_candidate_pool.py` performs one NSGA-II
  optimisation per individual scenario, pools scenario Pareto sets by a
  SHA-256 signature of full path nodes, modes, and allocation shares, then
  re-evaluates fixed copies. Its EV and CCP labels are post-optimisation
  reductions and therefore are not suitable as the revised EV/CCP optimiser.
- Normal core exports use `pareto_points.json` and include objectives,
  allocations, feasibility diagnostics, and on-time probabilities. The core
  NSGA-II return value deduplicates its display Pareto by objective values; the
  revised pilot instead takes rank-0 feasible population members and
  deduplicates only by complete decision fingerprint.
- `run_ccp_candidate_pool.re_evaluate_pool()` was the reusable fixed-candidate
  evaluator. It deep-copies candidates, invokes no operators, and checks the
  decision signature before and after every scenario evaluation. The revised
  pilot uses the same underlying `evaluate_individual()` simulator more
  efficiently over a complete validation set and applies the same signature
  guard.
- Search operations (`repair_missing_allocations`, crossover, mutation,
  feasibility boost, and route/mode replacement) occur only inside NSGA-II.
  No existing fixed-candidate evaluation function calls them. The danger was
  workflow-level reuse of `run_nsga2()` during “validation”; the new Stage C
  has no population and no call path to NSGA-II.

## Revised flow

`run_ev_ccp_oos_pilot.py` independently optimises direct-input EV, CCP30, and
CCP50 decisions with identical NSGA-II settings and unchanged operators. S30
is an exact prefix of one S50 training set. The three candidate sets remain
logically separate and have no size cap. Every candidate is then frozen and
passed through one separately generated common validation set. Scenario-level
cost, emission, and makespan are retained, and fingerprints are checked before
and after evaluation.
