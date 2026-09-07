# Current CCP mathematical-semantics audit

## Implemented optimisation model

For a fixed decision vector `x` and frozen training scenarios
`omega_1,...,omega_S`, `evaluate_individual()` constructs three realised
arrays:

- `C_s(x)`: transport, carbon, waiting, processing, transfer, and lateness cost;
- `E_s(x)`: transport emission plus waiting emission;
- `T_s(x)`: maximum batch completion time (makespan).

With `RISK_METRIC="ccp"`, the three NSGA-II minimisation objectives are

`Q_alpha^S(Z(x)) = sort(Z_1(x),...,Z_S(x))[ceil(alpha*S)-1]`,

applied independently to `C`, `E`, and `T`. At `alpha=0.90`, S30 selects the
27th sorted value, S50 the 45th, and S5000 the 4500th (one-based positions).
There is no joint multivariate quantile and no probability constraint encoded
by these three objective reductions.

Current hard feasibility is exactly:

- every batch has an allocation;
- every selected non-road service has required timetable data;
- no nominal-schedule arc-capacity excess;
- no nominal-schedule border-node-capacity excess.

Neither `batch_on_time_prob`, `LT`, nor `max_late_h` enters `hard_ok`, the
normalised violation, or the penalty. Thus the current executable model has no
punctuality chance constraint and no hard latest-arrival/maximum-lateness
constraint.

## Meaning of LT and the punctuality diagnostic

Every pilot batch has a positive finite `LT` in the workbook's `Batches`
sheet. In the uncertainty evaluator it has two roles:

1. scenario lateness `max(0, arrival_s - LT)` adds an economic late-delivery
   term to scenario cost; and
2. `mean(arrival_s <= LT)` is retained as a per-batch punctuality diagnostic.

The diagnostic is useful but is not CCP feasibility. `max_late_h` is loaded
and maximum observed lateness is reported, but the current optimiser does not
enforce it.

The deterministic predecessor `baseline3.py` differs: lateness contributes to
its penalty and `late_teu_h_total <= 1e-9` is part of deterministic hard
feasibility. This semantic difference must not be silently projected onto the
current uncertainty model.

## Research-formulation ambiguity

The current executable model is formulation **A: quantile-based objective
optimisation**. However, the preserved S200 implementation at commit
`032d23e9b31850bc0b3a2d146aca3a80ca6f0399` contained a separate
`CONFIDENCE_ONTIME`, `chance_vio`, maximum-lateness excess, and CCP-only hard
feasibility checks. Its run log also reports `ontime=0.9`.

Therefore repository history supplies evidence that formulation **B: a true
punctuality chance constraint** was previously intended, but the current code
does not implement B. Research intent cannot be resolved from the executable
code alone; choosing A or restoring/redesigning B is an explicit modelling
decision. No new constraint was added in this audit.

## Emission

For a fixed decision, transport emission is path distance times the fixed mode
factor and flow. The only scenario-varying emission term in the evaluator is
`wait_emis_g_per_teu_h * flow * scenario_wait_h`. The pilot workbook supplies
zero waiting emission, so every candidate satisfies

`mean(E) = median(E) = q90(E) = min(E) = max(E)`

up to floating-point representation. Emission would become scenario-dependent
only if a genuine nonzero waiting/idling emission coefficient were supplied,
or if a new scenario-dependent line-haul/emission-factor term were explicitly
added. Neither change was made.

## Full-run configuration correction

The preserved S200 pilot log records population **300**, generations **500**,
S **200**, and one run. “200x300” was incorrect shorthand. The intended
full-size single optimisation setting is population 300 and 500 generations;
formal replication count remains a separate decision. No larger optimisation
was run during this audit.
