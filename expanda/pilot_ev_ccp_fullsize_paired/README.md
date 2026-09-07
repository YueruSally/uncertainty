# EV/CCP out-of-sample workflow pilot

This is one paired pilot, not a formal replication series. It used population 300 and 500 generations once per method. EV optimises one environment constructed from the model's expected uncertainty inputs. CCP30 and CCP50 optimise empirical 90th-percentile objectives over nested training samples with shared algorithm seed 861337. All exported decisions are then frozen and evaluated over one independently generated common S_val=5000 set. Stage C calls only the scenario simulator/evaluator; it creates no population and invokes no search operator. Signature guards fail if any route, mode, or share changes.

Emissions are constant when waiting emissions are zero because the existing formula depends only on fixed path distance, mode emission factor, and flow.

See `FULLSIZE_PAIRED_PILOT_REPORT.md` for results. The scenario-level validation
table may be stored as `validation/scenario_level_results.csv.gz` to keep the
checkpoint manageable; it contains 4,015,000 data rows plus its header.
