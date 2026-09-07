# A-lite Stage-1B Report (full-run)

- Frozen methodology SHA-256: `df85d40b041f3a3882c2bca1adeec9b2b2b06b6f2452a920b4c23d9ab21a0188`
- Methodology freeze commit: `369585eb6d781dc4f69cf05960842bdded26ee09` (blob SHA-256 `df85d40b041f3a3882c2bca1adeec9b2b2b06b6f2452a920b4c23d9ab21a0188`)
- Grid validation: **VALIDATED** — 300/300 declared cells present exactly once and valid; imputed 0, dropped 0, reduced denominators 0
- Master seeds: [810001, 810002, 810003, 810004, 810005, 810006, 810007, 810008, 810009, 810010]
- S grid: [200, 500, 1000]   reference S_ref = 1000
- alpha = 0.9 (inclusive: p_hat >= alpha PASSES)
- Gates: **G1 AND G2dagger AND G3**, all STRICT `<`. There is NO INDETERMINATE category.
- **S1000 is an empirical reference, not ground truth.** No SAA or theoretical convergence claim is made for it; a PASS means the estimate and the label do not move materially between S and S1000 along a fixed realisation.
- **Bootstrap, LOCO and the 0.01 tripwire are non-gating** (DESCRIPTIVE ONLY / reporting only). They never create a PASS, a FAIL or an INDETERMINATE category and never block Stage 2.
- All-pair and KNIFE gross flip rates are **non-gating** diagnostics; only the OFF-stratum rate gates, as G3.

## Ladder decision

- selected S: **200**
- Stage 2 may proceed: **True**
- S500 gate consulted: False
- S200 vs S1000 passed G1 AND G2dagger AND G3. Per the frozen ladder S500 is NOT evaluated as a replacement gate decision; all its results remain part of the fixed study and are reported.

## S200_vs_S1000

**Verdict: PASS** — G1 = 0.015890 (pass), max_c D_c = +0.014800 [S1-07, OFF] (pass), G3 = 0.0800 (4/50) (pass)

### Per-candidate diagnostics (all rows, non-gating except as noted)

| candidate | stratum | D_c | A_c | Dmax_c | Dmin_c | nflip_c | netflip_c | bound@S | bound@ref |
|---|---|---|---|---|---|---|---|---|---|
| S1-01 | KNIFE | -0.011900 | 0.019100 | +0.022000 | -0.046000 | 5 | -3 | 3 | 0 |
| S1-02 | KNIFE | -0.008900 | 0.019300 | +0.022000 | -0.046000 | 5 | -3 | 2 | 0 |
| S1-03 | KNIFE | -0.008800 | 0.018800 | +0.026000 | -0.047000 | 5 | -5 | 0 | 1 |
| S1-04 | OFF | -0.008400 | 0.018800 | +0.026000 | -0.052000 | 4 | -4 | 0 | 1 |
| S1-05 | KNIFE | +0.005900 | 0.019500 | +0.031000 | -0.038000 | 2 | -2 | 1 | 1 |
| S1-06 | KNIFE | +0.005900 | 0.019500 | +0.031000 | -0.038000 | 2 | -2 | 1 | 1 |
| S1-07 | OFF | +0.014800 | 0.017800 | +0.030000 | -0.008000 | 0 | +0 | 0 | 0 |
| S1-08 | OFF | +0.003100 | 0.009300 | +0.022000 | -0.018000 | 0 | +0 | 0 | 0 |
| S1-09 | OFF | -0.000300 | 0.009300 | +0.014000 | -0.021000 | 0 | +0 | 0 | 0 |
| S1-10 | OFF | +0.001100 | 0.007500 | +0.017000 | -0.012000 | 0 | +0 | 0 | 0 |

### Panel diagnostics (non-gating)

- mean |dp| = 0.015890, median = 0.012000, max = 0.052000
- mean signed dp (T) = -0.000750
- all-pair flips 23/100 (0.2300)
- KNIFE flips 19/50 (0.3800)  [non-gating]
- OFF flips 4/50 (0.0800)  [G3]
- false-feasible 0, false-infeasible 4, net imbalance -4
- exact p_hat=alpha mass: 7 at S, 4 at ref
- binding-batch-set changes 29, max-lateness changes 3, non-CCP hard-pass changes 0
- 0.01 tripwire: T = -0.000750 -> **not tripped** [NON-GATING: reporting/investigation only]
- sqrt-rate check: observed 0.015890 vs scale 0.063246 (implied constant +0.251243)
- ratio-matched to Stage-1: False
- seed-cluster bootstrap (DESCRIPTIVE ONLY): G1 {'lo': 0.01333, 'hi': 0.018531000000000002, 'median': 0.01587}, G2dagger {'lo': 0.010389999999999995, 'hi': 0.0204, 'median': 0.0154}

### Per-seed breakdown (all seeds, non-gating)

| seed | mean signed | mean abs | max abs | flips | false-feasible | false-infeasible | net imbalance |
|---|---|---|---|---|---|---|---|
| 810001 | +0.006100 | 0.008900 | 0.022000 | 0 | 0 | 0 | +0 |
| 810002 | -0.000300 | 0.008900 | 0.015000 | 2 | 0 | 2 | -2 |
| 810003 | -0.019700 | 0.020900 | 0.052000 | 4 | 0 | 4 | -4 |
| 810004 | +0.015900 | 0.015900 | 0.030000 | 0 | 0 | 0 | +0 |
| 810005 | +0.011300 | 0.011300 | 0.028000 | 2 | 2 | 0 | +2 |
| 810006 | -0.011700 | 0.016700 | 0.025000 | 6 | 0 | 6 | -6 |
| 810007 | -0.019100 | 0.028500 | 0.038000 | 5 | 0 | 5 | -5 |
| 810008 | +0.019500 | 0.019500 | 0.031000 | 0 | 0 | 0 | +0 |
| 810009 | -0.014900 | 0.019900 | 0.050000 | 4 | 0 | 4 | -4 |
| 810010 | +0.005400 | 0.008400 | 0.022000 | 0 | 0 | 0 | +0 |

### Leave-one-candidate-out (LOCO) — DESCRIPTIVE ONLY, never gates

| omitted | stratum | G1 | max_c D_c | OFF flip rate |
|---|---|---|---|---|
| S1-01 | KNIFE | 0.015533 | +0.014800 | 0.0800 |
| S1-02 | KNIFE | 0.015511 | +0.014800 | 0.0800 |
| S1-03 | KNIFE | 0.015567 | +0.014800 | 0.0800 |
| S1-04 | OFF | 0.015567 | +0.014800 | 0.0000 |
| S1-05 | KNIFE | 0.015489 | +0.014800 | 0.0800 |
| S1-06 | KNIFE | 0.015489 | +0.014800 | 0.0800 |
| S1-07 | OFF | 0.015678 | +0.005900 | 0.1000 |
| S1-08 | OFF | 0.016622 | +0.014800 | 0.1000 |
| S1-09 | OFF | 0.016622 | +0.014800 | 0.1000 |
| S1-10 | OFF | 0.016822 | +0.014800 | 0.1000 |

## S500_vs_S1000

**Verdict: PASS** — G1 = 0.006110 (pass), max_c D_c = +0.000800 [S1-05, KNIFE] (pass), G3 = 0.0600 (3/50) (pass)

### Per-candidate diagnostics (all rows, non-gating except as noted)

| candidate | stratum | D_c | A_c | Dmax_c | Dmin_c | nflip_c | netflip_c | bound@S | bound@ref |
|---|---|---|---|---|---|---|---|---|---|
| S1-01 | KNIFE | -0.002800 | 0.006800 | +0.012000 | -0.026000 | 3 | -3 | 0 | 0 |
| S1-02 | KNIFE | -0.002400 | 0.007200 | +0.016000 | -0.026000 | 3 | -3 | 0 | 0 |
| S1-03 | KNIFE | -0.002100 | 0.007300 | +0.014000 | -0.019000 | 3 | -3 | 0 | 1 |
| S1-04 | OFF | -0.003900 | 0.009100 | +0.014000 | -0.024000 | 3 | -3 | 0 | 1 |
| S1-05 | KNIFE | +0.000800 | 0.006200 | +0.011000 | -0.017000 | 1 | -1 | 0 | 1 |
| S1-06 | KNIFE | +0.000800 | 0.006200 | +0.011000 | -0.017000 | 1 | -1 | 0 | 1 |
| S1-07 | OFF | -0.001200 | 0.006800 | +0.016000 | -0.011000 | 0 | +0 | 0 | 0 |
| S1-08 | OFF | -0.000900 | 0.004900 | +0.006000 | -0.012000 | 0 | +0 | 0 | 0 |
| S1-09 | OFF | -0.000200 | 0.003600 | +0.005000 | -0.010000 | 0 | +0 | 0 | 0 |
| S1-10 | OFF | -0.001400 | 0.003000 | +0.004000 | -0.010000 | 0 | +0 | 0 | 0 |

### Panel diagnostics (non-gating)

- mean |dp| = 0.006110, median = 0.004000, max = 0.026000
- mean signed dp (T) = -0.001330
- all-pair flips 14/100 (0.1400)
- KNIFE flips 11/50 (0.2200)  [non-gating]
- OFF flips 3/50 (0.0600)  [G3]
- false-feasible 0, false-infeasible 3, net imbalance -3
- exact p_hat=alpha mass: 0 at S, 4 at ref
- binding-batch-set changes 13, max-lateness changes 1, non-CCP hard-pass changes 0
- 0.01 tripwire: T = -0.001330 -> **not tripped** [NON-GATING: reporting/investigation only]
- sqrt-rate check: observed 0.006110 vs scale 0.031623 (implied constant +0.193215)
- ratio-matched to Stage-1: True, Stage-1 reference {'S100_vs_S200': {'ratio': 0.5, 'n_cells': 100, 'mean_abs_dp': 0.0155}}
- seed-cluster bootstrap (DESCRIPTIVE ONLY): G1 {'lo': 0.00469, 'hi': 0.00748, 'median': 0.00601}, G2dagger {'lo': -0.0001, 'hi': 0.0043, 'median': 0.0018}

### Per-seed breakdown (all seeds, non-gating)

| seed | mean signed | mean abs | max abs | flips | false-feasible | false-infeasible | net imbalance |
|---|---|---|---|---|---|---|---|
| 810001 | +0.006500 | 0.009100 | 0.016000 | 0 | 0 | 0 | +0 |
| 810002 | -0.006700 | 0.006700 | 0.012000 | 4 | 0 | 4 | -4 |
| 810003 | -0.007800 | 0.012800 | 0.026000 | 4 | 0 | 4 | -4 |
| 810004 | +0.001300 | 0.004100 | 0.016000 | 0 | 0 | 0 | +0 |
| 810005 | -0.000900 | 0.005300 | 0.011000 | 2 | 0 | 2 | -2 |
| 810006 | -0.008900 | 0.009700 | 0.017000 | 2 | 0 | 2 | -2 |
| 810007 | +0.001700 | 0.002300 | 0.008000 | 0 | 0 | 0 | +0 |
| 810008 | +0.001100 | 0.002300 | 0.004000 | 0 | 0 | 0 | +0 |
| 810009 | +0.001400 | 0.006000 | 0.016000 | 0 | 0 | 0 | +0 |
| 810010 | -0.001000 | 0.002800 | 0.006000 | 2 | 0 | 2 | -2 |

### Leave-one-candidate-out (LOCO) — DESCRIPTIVE ONLY, never gates

| omitted | stratum | G1 | max_c D_c | OFF flip rate |
|---|---|---|---|---|
| S1-01 | KNIFE | 0.006033 | +0.000800 | 0.0600 |
| S1-02 | KNIFE | 0.005989 | +0.000800 | 0.0600 |
| S1-03 | KNIFE | 0.005978 | +0.000800 | 0.0600 |
| S1-04 | OFF | 0.005778 | +0.000800 | 0.0000 |
| S1-05 | KNIFE | 0.006100 | +0.000800 | 0.0600 |
| S1-06 | KNIFE | 0.006100 | +0.000800 | 0.0600 |
| S1-07 | OFF | 0.006033 | +0.000800 | 0.0750 |
| S1-08 | OFF | 0.006244 | +0.000800 | 0.0750 |
| S1-09 | OFF | 0.006389 | +0.000800 | 0.0750 |
| S1-10 | OFF | 0.006456 | +0.000800 | 0.0750 |

## S200_vs_S500  (DIAGNOSTIC ONLY)

**Verdict: NOT_A_GATE** — G1 = 0.014500 (pass), max_c D_c = +0.016000 [S1-07, OFF] (pass), G3 = 0.0600 (3/50) (pass)

### Per-candidate diagnostics (all rows, non-gating except as noted)

| candidate | stratum | D_c | A_c | Dmax_c | Dmin_c | nflip_c | netflip_c | bound@S | bound@ref |
|---|---|---|---|---|---|---|---|---|---|
| S1-01 | KNIFE | -0.009100 | 0.016700 | +0.019000 | -0.037000 | 4 | +0 | 3 | 0 |
| S1-02 | KNIFE | -0.006500 | 0.016100 | +0.019000 | -0.037000 | 4 | +0 | 2 | 0 |
| S1-03 | KNIFE | -0.006700 | 0.017300 | +0.024000 | -0.037000 | 4 | -2 | 0 | 0 |
| S1-04 | OFF | -0.004500 | 0.013300 | +0.023000 | -0.034000 | 3 | -1 | 0 | 0 |
| S1-05 | KNIFE | +0.005100 | 0.018500 | +0.029000 | -0.037000 | 3 | -1 | 1 | 0 |
| S1-06 | KNIFE | +0.005100 | 0.018500 | +0.029000 | -0.037000 | 3 | -1 | 1 | 0 |
| S1-07 | OFF | +0.016000 | 0.016400 | +0.031000 | -0.002000 | 0 | +0 | 0 | 0 |
| S1-08 | OFF | +0.004000 | 0.010800 | +0.018000 | -0.021000 | 0 | +0 | 0 | 0 |
| S1-09 | OFF | -0.000100 | 0.009500 | +0.012000 | -0.023000 | 0 | +0 | 0 | 0 |
| S1-10 | OFF | +0.002500 | 0.007900 | +0.016000 | -0.015000 | 0 | +0 | 0 | 0 |

### Panel diagnostics (non-gating)

- mean |dp| = 0.014500, median = 0.012000, max = 0.037000
- mean signed dp (T) = +0.000580
- all-pair flips 21/100 (0.2100)
- KNIFE flips 18/50 (0.3600)  [non-gating]
- OFF flips 3/50 (0.0600)  [G3]
- false-feasible 1, false-infeasible 2, net imbalance -1
- exact p_hat=alpha mass: 7 at S, 0 at ref
- binding-batch-set changes 25, max-lateness changes 2, non-CCP hard-pass changes 0
- 0.01 tripwire: T = +0.000580 -> **not tripped** [NON-GATING: reporting/investigation only]
- sqrt-rate check: observed 0.014500 vs scale 0.054772 (implied constant +0.264733)
- ratio-matched to Stage-1: False
- seed-cluster bootstrap (DESCRIPTIVE ONLY): G1 {'lo': 0.01195, 'hi': 0.01711, 'median': 0.014365}, G2dagger {'lo': 0.0118, 'hi': 0.0201, 'median': 0.0162}

### Per-seed breakdown (all seeds, non-gating)

| seed | mean signed | mean abs | max abs | flips | false-feasible | false-infeasible | net imbalance |
|---|---|---|---|---|---|---|---|
| 810001 | -0.000400 | 0.011600 | 0.026000 | 0 | 0 | 0 | +0 |
| 810002 | +0.006400 | 0.010800 | 0.023000 | 2 | 2 | 0 | +2 |
| 810003 | -0.011900 | 0.012100 | 0.028000 | 0 | 0 | 0 | +0 |
| 810004 | +0.014600 | 0.014600 | 0.025000 | 0 | 0 | 0 | +0 |
| 810005 | +0.012200 | 0.012200 | 0.023000 | 4 | 4 | 0 | +4 |
| 810006 | -0.002800 | 0.007600 | 0.021000 | 4 | 0 | 4 | -4 |
| 810007 | -0.020800 | 0.028000 | 0.037000 | 5 | 0 | 5 | -5 |
| 810008 | +0.018400 | 0.018400 | 0.030000 | 0 | 0 | 0 | +0 |
| 810009 | -0.016300 | 0.022500 | 0.034000 | 4 | 0 | 4 | -4 |
| 810010 | +0.006400 | 0.007200 | 0.018000 | 2 | 2 | 0 | +2 |

### Leave-one-candidate-out (LOCO) — DESCRIPTIVE ONLY, never gates

| omitted | stratum | G1 | max_c D_c | OFF flip rate |
|---|---|---|---|---|
| S1-01 | KNIFE | 0.014256 | +0.016000 | 0.0600 |
| S1-02 | KNIFE | 0.014322 | +0.016000 | 0.0600 |
| S1-03 | KNIFE | 0.014189 | +0.016000 | 0.0600 |
| S1-04 | OFF | 0.014633 | +0.016000 | 0.0000 |
| S1-05 | KNIFE | 0.014056 | +0.016000 | 0.0600 |
| S1-06 | KNIFE | 0.014056 | +0.016000 | 0.0600 |
| S1-07 | OFF | 0.014289 | +0.005100 | 0.0750 |
| S1-08 | OFF | 0.014911 | +0.016000 | 0.0750 |
| S1-09 | OFF | 0.015056 | +0.016000 | 0.0750 |
| S1-10 | OFF | 0.015233 | +0.016000 | 0.0750 |

