# EV/CCP out-of-sample workflow pilot

This is an implementation pilot, not a formal replication. It used population
20 and 5 generations once per method; it did not start the formal 30-run
experiment and did not modify the earlier S200 pilot.

EV optimised one environment constructed directly from the model's expected
uncertainty inputs: line-haul multipliers 1.0 and border-event delays at their
model means. CCP30 and CCP50 optimised the existing separate empirical 90th
percentile cost, emission, and makespan objectives over nested training
samples. The S30 sample is the exact prefix of the S50 master sample.

All 23 exported decisions were frozen and evaluated over one independently
generated common `S_val=5000` set. Stage C calls only the scenario
simulator/evaluator; it creates no population and invokes no repair, crossover,
mutation, or replacement operation. Signature guards fail if any path, mode,
or allocation share changes.

Files:

- `configuration.json`: exact settings, seeds, scenario digests, and nesting /
  independence checks.
- `ev/`, `ccp30/`, `ccp50/`: separate final feasible nondominated decisions.
- `candidate_provenance.json`: full decisions, source IDs, training objectives,
  feasibility/reliability diagnostics, seeds, and fingerprints.
- `validation/scenario_level_results.csv`: 5,000 realised cost, emission, and
  makespan values per candidate.
- `validation/per_candidate_summary.json`: mean, min, max, median, p90,
  per-batch reliability, signature checks, and validation Pareto ranks.
- `runtime_summary.json` and `pilot_summary.json`: timings and headline counts.

Emission is constant for every fixed candidate because the existing model has
zero waiting emissions; its remaining emission terms depend only on fixed
path distance, mode factors, and flow. Cost and makespan remain stochastic,
including normal timetable rollover when a disturbance misses a departure.
