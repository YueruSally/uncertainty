# A-lite Stage-1B — REPORT-LAYER ADDENDUM (mandatory disclosures)

Generated 2026-08-18T10:52:42+00:00

## Status

This addendum **amends the report layer only**. It adds disclosures the frozen
methodology requires (amendment sections 5, 7.1, 7.3) that the report renderer
did not emit when this study was executed.

- The raw experiment was **NOT** recomputed, replaced, or re-run.
- All eight original artifacts are byte-unchanged (hashes below).
- Every gate value, verdict, denominator and the ladder decision were re-derived
  in memory from the preserved raw rows with the amended renderer and are
  **identical**. These disclosures are additive.

| artifact | sha256 (unchanged) |
|---|---|
| `comparisons.json` | `8e18040db94be4c5ed00ffb41a3dfc92578e3a8e863a86811910bae9e8938c92` |
| `ladder_decision.json` | `d664523d102dae3f77146ff9dd6b379fa6489bd9c46df714252d39ebdca1e822` |
| `manifest.json` | `b6df4bf0f8f79ba96039425e6f2d0527c7dd6c901031d4a4c893279e063b81c1` |
| `nested_prefix_verification.json` | `8cf906fa5d74cb3356d3b3eed6c3fd95bda340e16a8dd60208be28651fe97c17` |
| `per_candidate_diagnostics.json` | `f94e64ce5b8413d5716f01e7bccb93313b2771e20357dc70aa85034eaee9345e` |
| `raw_rows.csv` | `d40bb786d4cfb6113b0b593f3d79d71e65b7c1d1643c38055c46fe5b6c81bc7b` |
| `raw_rows.json` | `fe1d22154ee1571a0d3a32a3275cc46651f12f9e80a074b408e69c86e3bbfe61` |
| `STAGE1B_REPORT.md` | `db8c001885c3226eef0fe912fa243f30f66fe7b124075af97912d965d182820b` |

## The Stage-1B result is unchanged

- `selected_S` = **200**, `stage2_may_proceed` = **True**, `s500_gate_consulted` = **False**.
- **S200_vs_S1000**: G1 = 1589/100000, max_c D_c = 37/2500 (argmax S1-07), G3 = 2/25 — verdict **PASS**.
- **S500_vs_S1000**: G1 = 611/100000, max_c D_c = 1/1250 (argmax S1-05), G3 = 3/50 — verdict **PASS**.
- **S200_vs_S500**: diagnostic only, `NOT_A_GATE`; it never enters the ladder.

## Mandatory disclosures, per comparison

### S200_vs_S1000  (GATING)

Per amendment sections 7.1 and 7.3 the gate verdict **stands exactly as computed**;
nothing below overturns it or creates an INDETERMINATE category.

- Candidates S1-05 and S1-06 are STRUCTURALLY DISTINCT decisions that produce an identical p_hat series in every (seed, S) cell of this comparison. Effective panel diversity for the CCP worst-batch statistic is therefore below the nominal candidate count, and candidate-level replication claims must be qualified. The frozen panel is NOT reselected and no denominator is reduced on this basis.
- The DESCRIPTIVE seed-cluster bootstrap band for max_c D_c [0.010390, 0.020400] STRADDLES its 0.02 threshold. Per amendment section 7.1 the gate verdict stands exactly as computed; this fragility is a stated limitation to carry forward and may never overturn a verdict or create an INDETERMINATE category.
- The DESCRIPTIVE seed-cluster bootstrap band for OFF-stratum flip rate [0.040000, 0.120000] STRADDLES its 0.1 threshold. Per amendment section 7.1 the gate verdict stands exactly as computed; this fragility is a stated limitation to carry forward and may never overturn a verdict or create an INDETERMINATE category.
- Leave-one-candidate-out: omitting S1-07, S1-08, S1-09, S1-10 would move OFF-stratum flip rate across its 0.1 threshold from PASS to FAIL, so this component depends materially on the full frozen panel. DESCRIPTIVE ONLY -- it never changes the computed verdict.

### S500_vs_S1000  (GATING)

Per amendment sections 7.1 and 7.3 the gate verdict **stands exactly as computed**;
nothing below overturns it or creates an INDETERMINATE category.

- max_c D_c = 1/1250 is attained by MORE THAN ONE candidate (S1-05, S1-06). The single reported argmax is a deterministic first-in-frozen-order choice and must NOT be read as the uniquely worst candidate.
- Candidates S1-05 and S1-06 are STRUCTURALLY DISTINCT decisions that produce an identical p_hat series in every (seed, S) cell of this comparison. Effective panel diversity for the CCP worst-batch statistic is therefore below the nominal candidate count, and candidate-level replication claims must be qualified. The frozen panel is NOT reselected and no denominator is reduced on this basis.

### S200_vs_S500  (DIAGNOSTIC ONLY — never gates)

Per amendment sections 7.1 and 7.3 the gate verdict **stands exactly as computed**;
nothing below overturns it or creates an INDETERMINATE category.

- Candidates S1-05 and S1-06 are STRUCTURALLY DISTINCT decisions that produce an identical p_hat series in every (seed, S) cell of this comparison. Effective panel diversity for the CCP worst-batch statistic is therefore below the nominal candidate count, and candidate-level replication claims must be qualified. The frozen panel is NOT reselected and no denominator is reduced on this basis.
- The DESCRIPTIVE seed-cluster bootstrap band for max_c D_c [0.011800, 0.020100] STRADDLES its 0.02 threshold. Per amendment section 7.1 the gate verdict stands exactly as computed; this fragility is a stated limitation to carry forward and may never overturn a verdict or create an INDETERMINATE category.

