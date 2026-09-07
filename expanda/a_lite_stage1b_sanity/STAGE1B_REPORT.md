# A-lite Stage-1B Report (sanity-run)

> **SANITY RUN -- PLUMBING/DATA-FLOW CHECK ONLY. Reduced candidates/seeds. Every gate value, D_c, flip rate, tripwire and bootstrap interval in this file is SCIENTIFICALLY UNINTERPRETABLE and must never be quoted, plotted or treated as a Stage-1B result.**

- Frozen methodology SHA-256: `df85d40b041f3a3882c2bca1adeec9b2b2b06b6f2452a920b4c23d9ab21a0188`
- Methodology freeze commit: `369585eb6d781dc4f69cf05960842bdded26ee09` (blob SHA-256 `df85d40b041f3a3882c2bca1adeec9b2b2b06b6f2452a920b4c23d9ab21a0188`)
- Grid validation: **VALIDATED** — 6/6 declared cells present exactly once and valid; imputed 0, dropped 0, reduced denominators 0
- Master seeds: [810001]
- S grid: [200, 500, 1000]   reference S_ref = 1000
- alpha = 0.9 (inclusive: p_hat >= alpha PASSES)
- Gates: **G1 AND G2dagger AND G3**, all STRICT `<`. There is NO INDETERMINATE category.
- **S1000 is an empirical reference, not ground truth.** No SAA or theoretical convergence claim is made for it; a PASS means the estimate and the label do not move materially between S and S1000 along a fixed realisation.
- **Bootstrap, LOCO and the 0.01 tripwire are non-gating** (DESCRIPTIVE ONLY / reporting only). They never create a PASS, a FAIL or an INDETERMINATE category and never block Stage 2.
- All-pair and KNIFE gross flip rates are **non-gating** diagnostics; only the OFF-stratum rate gates, as G3.

## Ladder decision

- selected S: **None**
- Stage 2 may proceed: **False**
- S500 gate consulted: None
- SANITY RUN -- no ladder decision is computed or valid.

## S200_vs_S1000

**Verdict: PASS** — G1 = 0.008000 (pass), max_c D_c = +0.002000 [S1-04, OFF] (pass), G3 = 0.0000 (0/1) (pass)

### Per-candidate diagnostics (all rows, non-gating except as noted)

| candidate | stratum | D_c | A_c | Dmax_c | Dmin_c | nflip_c | netflip_c | bound@S | bound@ref |
|---|---|---|---|---|---|---|---|---|---|
| S1-01 | KNIFE | -0.014000 | 0.014000 | -0.014000 | -0.014000 | 0 | +0 | 1 | 0 |
| S1-04 | OFF | +0.002000 | 0.002000 | +0.002000 | +0.002000 | 0 | +0 | 0 | 0 |

### Panel diagnostics (non-gating)

- mean |dp| = 0.008000, median = 0.008000, max = 0.014000
- mean signed dp (T) = -0.006000
- all-pair flips 0/2 (0.0000)
- KNIFE flips 0/1 (0.0000)  [non-gating]
- OFF flips 0/1 (0.0000)  [G3]
- false-feasible 0, false-infeasible 0, net imbalance +0
- exact p_hat=alpha mass: 1 at S, 0 at ref
- binding-batch-set changes 1, max-lateness changes 1, non-CCP hard-pass changes 0
- 0.01 tripwire: T = -0.006000 -> **not tripped** [NON-GATING: reporting/investigation only]
- sqrt-rate check: observed 0.008000 vs scale 0.063246 (implied constant +0.126491)
- ratio-matched to Stage-1: False
- seed-cluster bootstrap (DESCRIPTIVE ONLY): G1 {'lo': 0.008, 'hi': 0.008, 'median': 0.008}, G2dagger {'lo': 0.002, 'hi': 0.002, 'median': 0.002}

### Per-seed breakdown (all seeds, non-gating)

| seed | mean signed | mean abs | max abs | flips | false-feasible | false-infeasible | net imbalance |
|---|---|---|---|---|---|---|---|
| 810001 | -0.006000 | 0.008000 | 0.014000 | 0 | 0 | 0 | +0 |

### Leave-one-candidate-out (LOCO) — DESCRIPTIVE ONLY, never gates

| omitted | stratum | G1 | max_c D_c | OFF flip rate |
|---|---|---|---|---|
| S1-01 | KNIFE | 0.002000 | +0.002000 | 0.0000 |
| S1-04 | OFF | 0.014000 | -0.014000 | n/a |

## S500_vs_S1000

**Verdict: PASS** — G1 = 0.013000 (pass), max_c D_c = +0.014000 [S1-04, OFF] (pass), G3 = 0.0000 (0/1) (pass)

> **TRIPWIRE FIRED IN A NON-SCIENTIFIC RUN — DO NOT QUOTE.** The 0.01 optimism tripwire fired (T=0.013000, optimism) in a NON-SCIENTIFIC run. This confirms only that the tripwire and its reporting path work. There is NO accepted operating point, NO valid ladder decision and NO scientific verdict in such a run, so this value MUST NOT be quoted and MUST NOT be carried forward as a Stage-2 limitation. Only a full run can create that obligation.

### Per-candidate diagnostics (all rows, non-gating except as noted)

| candidate | stratum | D_c | A_c | Dmax_c | Dmin_c | nflip_c | netflip_c | bound@S | bound@ref |
|---|---|---|---|---|---|---|---|---|---|
| S1-01 | KNIFE | +0.012000 | 0.012000 | +0.012000 | +0.012000 | 0 | +0 | 0 | 0 |
| S1-04 | OFF | +0.014000 | 0.014000 | +0.014000 | +0.014000 | 0 | +0 | 0 | 0 |

### Panel diagnostics (non-gating)

- mean |dp| = 0.013000, median = 0.013000, max = 0.014000
- mean signed dp (T) = +0.013000
- all-pair flips 0/2 (0.0000)
- KNIFE flips 0/1 (0.0000)  [non-gating]
- OFF flips 0/1 (0.0000)  [G3]
- false-feasible 0, false-infeasible 0, net imbalance +0
- exact p_hat=alpha mass: 0 at S, 0 at ref
- binding-batch-set changes 1, max-lateness changes 1, non-CCP hard-pass changes 0
- 0.01 tripwire: T = +0.013000 -> **TRIPPED (optimism)** [NON-GATING: reporting/investigation only]
- sqrt-rate check: observed 0.013000 vs scale 0.031623 (implied constant +0.411096)
- ratio-matched to Stage-1: True, Stage-1 reference {'S100_vs_S200': {'ratio': 0.5, 'n_cells': 100, 'mean_abs_dp': 0.0155}}
- seed-cluster bootstrap (DESCRIPTIVE ONLY): G1 {'lo': 0.013, 'hi': 0.013, 'median': 0.013}, G2dagger {'lo': 0.014, 'hi': 0.014, 'median': 0.014}

### Per-seed breakdown (all seeds, non-gating)

| seed | mean signed | mean abs | max abs | flips | false-feasible | false-infeasible | net imbalance |
|---|---|---|---|---|---|---|---|
| 810001 | +0.013000 | 0.013000 | 0.014000 | 0 | 0 | 0 | +0 |

### Leave-one-candidate-out (LOCO) — DESCRIPTIVE ONLY, never gates

| omitted | stratum | G1 | max_c D_c | OFF flip rate |
|---|---|---|---|---|
| S1-01 | KNIFE | 0.014000 | +0.014000 | 0.0000 |
| S1-04 | OFF | 0.012000 | +0.012000 | n/a |

## S200_vs_S500  (DIAGNOSTIC ONLY)

**Verdict: NOT_A_GATE** — G1 = 0.019000 (pass), max_c D_c = -0.012000 [S1-04, OFF] (pass), G3 = 0.0000 (0/1) (pass)

### Per-candidate diagnostics (all rows, non-gating except as noted)

| candidate | stratum | D_c | A_c | Dmax_c | Dmin_c | nflip_c | netflip_c | bound@S | bound@ref |
|---|---|---|---|---|---|---|---|---|---|
| S1-01 | KNIFE | -0.026000 | 0.026000 | -0.026000 | -0.026000 | 0 | +0 | 1 | 0 |
| S1-04 | OFF | -0.012000 | 0.012000 | -0.012000 | -0.012000 | 0 | +0 | 0 | 0 |

### Panel diagnostics (non-gating)

- mean |dp| = 0.019000, median = 0.019000, max = 0.026000
- mean signed dp (T) = -0.019000
- all-pair flips 0/2 (0.0000)
- KNIFE flips 0/1 (0.0000)  [non-gating]
- OFF flips 0/1 (0.0000)  [G3]
- false-feasible 0, false-infeasible 0, net imbalance +0
- exact p_hat=alpha mass: 1 at S, 0 at ref
- binding-batch-set changes 0, max-lateness changes 0, non-CCP hard-pass changes 0
- 0.01 tripwire: T = -0.019000 -> **TRIPPED (pessimism)** [NON-GATING: reporting/investigation only]
- sqrt-rate check: observed 0.019000 vs scale 0.054772 (implied constant +0.346891)
- ratio-matched to Stage-1: False
- seed-cluster bootstrap (DESCRIPTIVE ONLY): G1 {'lo': 0.019, 'hi': 0.019, 'median': 0.019}, G2dagger {'lo': -0.012, 'hi': -0.012, 'median': -0.012}

### Per-seed breakdown (all seeds, non-gating)

| seed | mean signed | mean abs | max abs | flips | false-feasible | false-infeasible | net imbalance |
|---|---|---|---|---|---|---|---|
| 810001 | -0.019000 | 0.019000 | 0.026000 | 0 | 0 | 0 | +0 |

### Leave-one-candidate-out (LOCO) — DESCRIPTIVE ONLY, never gates

| omitted | stratum | G1 | max_c D_c | OFF flip rate |
|---|---|---|---|---|
| S1-01 | KNIFE | 0.012000 | -0.012000 | 0.0000 |
| S1-04 | OFF | 0.026000 | -0.026000 | n/a |

