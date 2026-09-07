# A-lite Stage 1B — Pre-Result Methodology Amendment 1 (G2†)

**Date:** 2026-08-17
**Status:** PRE-RESULT. Nothing implemented, nothing run, no Stage-1B artifact exists.
**Author:** Claude (main executor). To be reviewed by Codex MCP as independent reviewer.

**Purpose.** The prior document
`expanda/A_LITE_STAGE1B_CLAUDE_METHODOLOGY_REVIEW.md`
left exactly one gating item unresolved (O1: which statistic measures scenario-count
adequacy) and specified the Claude-side candidate as

    G2   | #(pass@S & fail@ref) - #(fail@S & pass@ref) | / 100  < 0.05

This amendment **supersedes exactly two items** of the prior review — (i) that G2 line, and
(ii) the gating role of the seed-cluster bootstrap indeterminacy band (§7.1 below) — and adds
the pre-run preconditions and diagnostics needed to make G2's replacement, **G2†**,
implementable without guessing. Everything else in the prior review stands unchanged. The
prior review is **not** edited; it remains the historical record of what was agreed at
commit-time.

**Scope discipline.** This amendment does not reopen, reword, or re-justify any of the
other agreed methodology items (Stage-1B necessity, S grid, candidate panel, seed policy,
nesting, comparison set, KNIFE/OFF strata, G1, G3, materiality values, held-out policy,
Stage-2 gate ladder, evaluation scope, limitations). Where this document restates them, it
restates them verbatim in substance for completeness only.

**The scientific acceptance gates are G1, G2† and G3, and nothing else.** No bootstrap
interval, no leave-one-candidate-out (LOCO) sensitivity analysis, no tripwire, and no other
statistic may create an acceptance category, override a gate, or block Stage 2
independently. This is stated once here and enforced throughout §7.

---

## 0. Version-mismatch record (Step 1)

Codex is **correct**. The saved review file
(`A_LITE_STAGE1B_CLAUDE_METHODOLOGY_REVIEW.md`, 406 lines,
md5 `2ad31e3a0ad5f3cbf7301194bd85dc03`) contains:

- the **older** `G2 = |net classification bias| / 100 < 0.05` (line 207), and
- **no** occurrence of `D_c`, `G2†`, a `0.01` non-gating tripwire, an explicit
  `p_hat >= alpha` tie-rule precondition, or an i.i.d. generator precondition.

The newer proposal existed only in conversation. That is the traceability defect this
amendment closes. The historical file is left byte-for-byte unmodified.

---

## 1. Notation and primitives

| symbol | meaning |
|---|---|
| `c` | **candidate index**. `c ∈ C = {S1-01, …, S1-10}`, the 10 frozen Stage-1 candidates (provenance `feb4cf2`, canonical sha256 `ce214df2…79f33`). `N_c = 10`, fixed. |
| `s` | **master-seed index**. `s ∈ Σ`, the 10 new pre-declared Stage-1B master seeds. `N_s = 10`, fixed. `s` indexes seeds, **never** scenarios. |
| `j` | **scenario replication index** inside one ScenarioSet, `j = 1 … S`. |
| `b` | **batch index**, `b ∈ B`, the 20 China→Europe batches. |
| `S` | scenario count under test, `S ∈ {200, 500}`. |
| `S_ref` | reference scenario count. **For every gating statistic, `S_ref = 1000` always.** |
| `α` | `0.90`, the CCP on-time confidence level (`CONFIDENCE_ONTIME`, `baseline_uncertainty.py:178`). |

**The estimator.** For candidate `c`, seed `s`, scenario count `S`:

    p_hat_{c,s,S}  =  min_{b ∈ B}  (1/S) · Σ_{j=1..S}  1[ A_{c,s,b,j} ≤ LT_b ]

i.e. `min_on_time_prob` as produced by production
(`baseline_uncertainty.py:2308-2310`), where `A_{c,s,b,j}` is batch `b`'s arrival hour
under scenario `j`. It is a **minimum over 20 batch-level empirical proportions**, not a
single binomial proportion. It takes values in `{k/S}`.

**The decision (see precondition B).**

    Pass_{c,s,S}  =  1[ chance_vio_total_{c,s,S} ≤ TOL_HARD ],   TOL_HARD = 1e-12

with `chance_vio_total = Σ_b max(0, α − p_b)`. This is the production predicate
(`a_lite_stage1_sensitivity.py:480`), reused unchanged.

---

## 2. G2† and D_c — exact definition (Step 2)

### 2.1 Definition

For a comparison `(S vs S_ref)` and each candidate `c`:

    D_c(S, S_ref)  =  (1 / N_s) · Σ_{s ∈ Σ}  ( p_hat_{c,s,S}  −  p_hat_{c,s,S_ref} )

and

    G2†(S, S_ref)  PASSES  ⟺  max_{c ∈ C}  D_c(S, S_ref)  <  0.02

### 2.2 The ten required clarifications, answered in order

1. **What `c` represents.** The candidate index over the 10 frozen Stage-1 candidates.
   The set is fixed in advance and is never filtered, reweighted, or extended. `N_c = 10`.
2. **What `s` represents.** The **master-seed** index over the 10 new pre-declared
   Stage-1B seeds. `s` is *not* a scenario index; the scenario index is `j`, and `j` is
   already fully consumed inside `p_hat` before `D_c` is formed.
3. **Signed or absolute.** `D_c` is **signed**. The inner term
   `p_hat_{c,s,S} − p_hat_{c,s,S_ref}` keeps its sign, and the average over seeds is taken
   on the signed values. No absolute value is applied anywhere in `D_c`. The
   absolute-difference companion statistic `A_c` is defined separately in §5 and is
   **non-gating**.
4. **Exact formula across the 10 new master seeds.** As in §2.1: an unweighted arithmetic
   mean over exactly the 10 master seeds of the within-(candidate, seed) paired signed
   difference. Each seed contributes exactly one paired difference, weight `1/10`. No
   pooling across candidates occurs inside `D_c` — pooling across candidates happens only
   through the outer `max`.
5. **Do positive values mean smaller-S optimism.** **Yes.** `D_c > 0` means candidate `c`'s
   estimated worst-batch on-time probability is on average **higher** at `S` than at
   `S_ref = 1000`. Because the CCP test is one-sided (`p_hat ≥ α` accepts), a higher
   estimate is a **more permissive** estimate. Positive `D_c` is therefore
   **smaller-S optimism** and is the direction that causes over-acceptance.
6. **Are negative values treated differently.** **Yes, deliberately.** `D_c < 0` is
   smaller-S **pessimism**: `S` under-states reliability relative to the reference and
   would reject candidates the reference accepts. That is conservative with respect to the
   Stage-2 accept decision, so it is **not gated** by G2†. It is **not ignored**: `min_c D_c`,
   the full signed per-candidate table, and `max_c A_c` are all mandatory reported
   diagnostics (§5), and two-sided magnitude is already gated globally by G1. G2† is
   one-sided **by construction**, matching the one-sided failure mode it targets (§2.3).
7. **How equality is treated.** The comparison is **strict**. `max_c D_c = 0.02` exactly is
   a **FAIL**. This matches the pre-declared Stage-1 materiality convention recorded in
   `a_lite_stage1_results/manifest.json`: *"Thresholds are inclusive (>=)"* for declaring an
   effect MATERIAL, so a statistic at or above 0.02 is material and cannot pass. Pass
   requires `< 0.02`. (Same strictness convention as the unchanged G1 and G3.)
8. **Exact denominator.** The denominator inside `D_c` is **`N_s = 10`**, the number of
   master seeds — a fixed constant, **not** the number of rows found, **not** 100, and
   **never** renormalised to a surviving-row count. The outer `max` is over exactly
   `N_c = 10` candidates. If fewer than 10 valid seeds or 10 valid candidates are present,
   the run fails closed per item 10; the denominator is never reduced to accommodate loss.
9. **Is the reference always S1000.** **For gating, yes, always.** Both gated comparisons
   (`S200 vs S1000` and `S500 vs S1000`) use `S_ref = 1000`. `D_c(200, 500)` is computed and
   reported for the `S200 vs S500` comparison, but that comparison is **diagnostic only and
   never gates** (unchanged from the prior review, §F/§G). No gating decision anywhere in
   Stage 1B uses a reference other than S1000.
10. **How missing / non-finite rows fail closed.** An explicit pre-check runs **before** any
    `D_c` is computed. A cell `(c, s, S)` is **VALID** iff:
    (a) exactly one result row exists for the triple (duplicates ⇒ invalid);
    (b) `p_hat_{c,s,S}` is finite and lies in `[0, 1]`;
    (c) `chance_vio_total_{c,s,S}` is finite and `≥ 0`;
    (d) the stored `chance_constraint_pass` equals the recomputed production predicate.
    The run requires **all** `N_c × N_s × |{200, 500, 1000}| = 300` cells valid — not merely
    the two scenario counts entering a given comparison — because all three comparisons are
    mandatory outputs. If any cell is invalid: `D_c` is **not** computed on a reduced
    denominator, values are **not** imputed, cells are **not** dropped, and NaN propagation
    is **not** relied upon. Instead: **if any of the 300 cells is invalid, the entire
    Stage-1B run is INVALID; neither S200 nor S500 may pass, and Stage 2 must not start. No
    scientific result artifacts may be written. A dedicated validation-failure artifact must
    be written containing only the invalid status and every offending `(c, s, S)` triple.**
    The hierarchy of §7 does **not** fall through to `S500 vs S1000` on an invalid run — an
    invalid run yields no comparison verdicts at all. Fail-closed is unconditional: it
    applies even if the surviving cells would have passed.

### 2.3 The scientific failure mode G2† is intended to detect

Finding F2 of the prior review established that `p_hat` is **not** an unbiased estimator
that merely gets noisier at small `S`. It is a **minimum over 20 dependent batch-level
proportions**. Each individual batch proportion is an ordinary empirical proportion and is
*not* itself rounded or biased. **Stage 1 observed the smaller-`S` minimum displaced upward
relative to its larger-`S` counterpart; the effect concerns the distribution of the minimum
over dependent proportions and the coarse `1/S` grid.** The upward direction is an observed
property of this panel and grid, **not** a guaranteed mechanical consequence of taking a
minimum on a coarser grid. Stage-1 measured mean
`p_hat` of 0.9270 / 0.9218 / 0.9183 at S = 50 / 100 / 200, with the number of tied binding
batches falling from 1.67 to 1.06 over the same range. The bias is **systematic, one-sided
(optimistic), and had not plateaued** at the largest `S` tested.

A systematic optimism of this kind is a **per-candidate property**, because it is driven by
how many batches sit near that particular candidate's binding constraint. Two aggregation
choices can conceal it:

- **Cancellation.** A global *signed* mean lets a `+0.03` candidate be offset by a `−0.03`
  candidate and report `0.00`.
- **Dilution.** A global mean over 100 cells divides one badly biased candidate's signal by
  ten. A candidate carrying a persistent `+0.02` optimism moves the 100-cell mean by only
  `+0.002`, an order of magnitude below any threshold on that scale.

G1 (mean **absolute** paired difference) removes cancellation but not dilution, and by
discarding sign it also cannot distinguish systematic optimism from symmetric knife-edge
noise — which F1 showed is the dominant contributor in this boundary-enriched panel.

**Stage 2 accepts or rejects candidates one at a time.** A single candidate whose CCP
probability is systematically over-estimated by 0.02 at the operating `S` is a real
decision error for that candidate, whatever the panel average does. G2† therefore asks the
decision-relevant question directly: *is there any candidate for which running at `S`
instead of `S_ref` systematically inflates the accept statistic by a material amount?* The
`max_c` takes the worst candidate; the signed inner term restricts attention to the
dangerous direction; the mean over seeds averages out realisation noise so that the
statistic estimates a **systematic** per-candidate offset rather than a single unlucky draw.

**Formal relation to the alternatives** (proved, not asserted):
`max_c D_c ≥ (1/N_c) Σ_c D_c = T`, the global signed mean (§4). So G2† at 0.02 is uniformly
at least as strict as a global-signed-mean test at 0.02. Conversely `max_c D_c` and G1 are
**not** ordered in either direction: with one candidate at `+0.03` and nine at `0.00` with
no within-candidate spread, `G1 = 0.003` (pass) while `max_c D_c = 0.03` (fail). That case
is exactly the failure mode above, and it is why G2† is a genuine addition rather than a
restatement of G1.

### 2.4 Honest disclosure — retrospective behaviour on Stage 1

The instruction was not to choose the formula to make Stage-1's S=100 pass. The formula was
derived from F2's structure, as argued in §2.3. Its retrospective behaviour is reported
here so the reviewer can judge that claim, **not** as justification. Recomputed from
`a_lite_stage1_results/raw_rows.csv` with `S_ref = 200` (Stage 1's largest available
reference, the structural analogue of S1000):

| comparison | `max_c D_c` (argmax) | `min_c D_c` | G1 | G2† verdict |
|---|---|---|---|---|
| S50 vs S200 | **+0.0265** (S1-05) | −0.0070 | 0.0240 | **FAIL** |
| S100 vs S200 | **+0.0135** (S1-05) | −0.0090 | 0.0155 | pass |

Three points a reviewer should weigh:

- G2† does **not** rescue S=50: it fails on G2† independently at 0.0265, in addition to
  failing the unchanged G1 at 0.0240. A rule tuned to be permissive would not do this.
- G2† does **not** change any Stage-1 verdict relative to the superseded G2. Both the old
  G2 (`+0.02 ≤ 0.05`, pass) and G2† (`0.0135 < 0.02`, pass) accept S=100 vs S=200. The
  already-disclosed disagreement with Codex on Stage-1's S=100 is therefore **unchanged** by
  this amendment; it is not newly introduced by it, and it is not enlarged by it.
- The sign choice is not what produces the pass. Under a two-sided variant
  (`max_c |D_c| < 0.02`) the verdicts are identical: 0.0265 FAIL at S=50, 0.0135 pass at
  S=100. So the one-sided form was not selected to secure a Stage-1 outcome.

**Known conservatism of the `max` statistic — disclosed, not corrected.** `max_c D_c` is a
maximum over 10 noisy per-candidate means, so it is biased **upward** under pure noise. From
Stage-1's dispersion, the per-cell paired difference has sd of order 0.02, giving each `D_c`
a standard error of order `0.02/√10 ≈ 0.006`; the maximum of 10 independent zero-mean draws
at that scale has expectation of order `1.5 × 0.006 ≈ 0.009`, a non-trivial fraction of the
0.02 threshold. (The true clustering is crossed, so this is an order-of-magnitude figure,
not a calibrated one.) G2† therefore errs toward **rejection** — toward *not* starting
Stage 2. That is the safe direction for this decision. It is the reason the seed-cluster
bootstrap and LOCO analyses (§7.3, items 12–13) must accompany `max_c D_c` in the report —
**as descriptive robustness context only; they never alter the G2† verdict** (§7.1). It is
stated here
rather than patched away.

---

## 3. Justification of the 0.02 threshold (Step 3)

**Classification, stated precisely (this is deliberately narrower than an earlier draft of
this section claimed).** The **numeric value** 0.02 is inherited unchanged from the
pre-declared Stage-1 probability-materiality scale, and is intentionally aligned with the
existing G1 decision scale. The **pairing of that value with the `max_c D_c` functional is
new**, and is predeclared here, before any Stage-1B execution:

> G2† newly predeclares, before Stage-1B execution, the pairing of `max_c D_c` with
> threshold `0.02`. The numeric value is reused from Stage 1 as a decision-relevance scale,
> but its application to this max-statistic was not predeclared in Stage 1 and is not
> statistically calibrated for the maximum's sampling distribution. The rule is
> intentionally conservative; its upward noise bias can cause false inadequacy findings and
> does not estimate a family-wise error rate.

Precisely:

- **Provenance of the value.** 0.02 is recorded in `a_lite_stage1_results/manifest.json`
  under `predeclared_materiality.mean_abs_prob_delta_vs_S200 = 0.02`, declared before Stage-1
  results were inspected. This amendment reuses that constant **numerically unchanged**.
- **What is new.** The **functional** it is applied to. Stage 1 predeclared 0.02 for a
  *global mean absolute* difference, with a mean/resolution rationale. G2† applies it to a
  *per-candidate worst-case signed mean*. The unit is identical in both uses — a difference
  of two CCP probability estimates on the probability scale — and the strict-`<` convention
  is identical, but the aggregation, and therefore the statistic's sampling distribution, is
  not the one for which 0.02 was originally declared. Calling this "inheritance" of a
  threshold would overstate it; it is inheritance of a **scale**, plus a **new predeclared
  pairing**.
- **Consequence, stated rather than smoothed over.** Because the pairing is not calibrated to
  the maximum's null distribution (§2.4 quantifies an upward uplift of order 0.009 against a
  0.02 threshold), G2† can produce **false inadequacy findings** — it can fail an `S` that is
  in fact adequate. It errs toward not starting Stage 2, and it makes no family-wise error
  rate claim of any kind.
- **Why it remains the right scale after the change of functional.** Section H of the prior
  review already retired the original "one S=50 quantisation step" rationale (one step is
  0.002 at S=500 and 0.001 at S=1000) and re-justified 0.02 as a **decision-relevance**
  threshold rather than a resolution threshold. On the decision-relevance reading, 0.02 is
  20% of the entire distance from `α = 0.90` to certainty, and it is twice the half-width of
  the pre-registered KNIFE band (`|p̄ − 0.90| < 0.01`) — i.e. a systematic shift of 0.02 is
  large enough to carry a candidate clean across the boundary region that defines the strata.
  That reading is functional-independent, so it transfers to G2† without modification.
- **Why not a different number.** Introducing a *new* constant for G2† would be a free
  parameter chosen after the failure mode was identified, and it would put the two gating
  probability criteria on two different materiality scales with no principled ratio between
  them. Reusing the single pre-declared probability-materiality constant keeps Stage 1B with
  exactly the two numeric thresholds Stage 1 already declared: **0.02 on probability
  differences, 0.10 on flip rates.** The superseded G2 was the only criterion that had
  introduced a third constant (0.05, on a different scale — a rate, not a probability
  difference); removing it **reduces** the number of free parameters in the gate.

**Non-selection statement.** Stage 1B has not been implemented or run; no Stage-1B
ScenarioSet, evaluation, or result artifact exists at the time of writing, and none was
inspected. No threshold in this amendment — 0.02, 0.01, 0.10, or `α = 0.90` — was or could
have been selected from a Stage-1B outcome. The only empirical numbers consulted are Stage-1
results (`a_lite_stage1_results/`), which are out-of-sample with respect to Stage 1B's new
disjoint master seeds, and they were consulted **after** the functional was fixed, for the
disclosure in §2.4.

---

## 4. The 0.01 non-gating integrity tripwire (Step 4)

### 4.1 Exact formula

For a comparison `(S vs S_ref)`, over all `N_c × N_s = 100` candidate × seed cells:

    T(S, S_ref)  =  (1 / (N_c · N_s)) · Σ_{c ∈ C} Σ_{s ∈ Σ}  ( p_hat_{c,s,S}  −  p_hat_{c,s,S_ref} )

Signed; denominator exactly **100** (fixed, never a surviving-row count — the same
fail-closed validity pre-check of §2.2 item 10 applies before `T` is formed). This is the
quantity Codex previously proposed, retained unchanged in definition.

### 4.2 Sign meaning

`T > 0` means the estimator is **on aggregate optimistic at the smaller `S`**: across the
whole panel, running at `S` instead of `S_ref` inflates the accept statistic. This is the
panel-level signature of the F2 quantisation bias. `T < 0` is aggregate pessimism at
smaller `S`.

### 4.3 Threshold semantics

**Inclusive**, and two-sided with an asymmetric interpretation:

- **TRIPPED (optimism):** `T ≥ +0.01`. This is the primary condition and the one carrying
  the integrity concern.
- **TRIPPED (pessimism):** `T ≤ −0.01`. Recorded, with the same reporting obligation, and
  labelled by direction.
- **NOT TRIPPED:** `|T| < 0.01`.

`T = +0.01` exactly is TRIPPED (inclusive `≥`, matching the pre-declared "thresholds are
inclusive" convention for declaring an effect material). Note this is the opposite bracket
convention from the *gates*, which pass on strict `<` — both follow the same underlying rule
that a statistic **at** its threshold is treated as material.

### 4.4 Operational consequence — explicitly **option B: investigation and reporting only**

Answering the question as posed: `T ≥ 0.01` causes **B**, not **A**. It is **not** an
automatic scientific failure, it does **not** flip any gate, it does **not** by itself block
Stage 2, and it may **never** be used to overturn a G1/G2†/G3 PASS or to rescue a FAIL. The
Stage-2 gate is `G1 ∧ G2† ∧ G3` and nothing else (§7).

"Non-gating" must not degrade into "ignored", so the consequence is made concrete and
mandatory:

1. `T` and its sign **must** be computed and reported for **every** comparison — including
   `S200 vs S500` — **regardless of the gate outcome**. It may not be omitted on the grounds
   that the gate passed. Omission is a reporting failure and invalidates the run.
2. If TRIPPED, the Stage-1B report **must** contain a dedicated, clearly headed section that:
   (a) states `T`, its sign, and which comparison tripped;
   (b) decomposes `T` by candidate (the full `D_c` table of §5) **and** by seed;
   (c) states whether the trip is concentrated in a subset of candidates/seeds or is
   uniform across the panel;
   (d) states whether it is concentrated in the KNIFE or OFF stratum.
3. **The combination that must be escalated.** If a comparison is **TRIPPED while its gate
   PASSES**, that specific combination must be stated explicitly in the Stage-1B summary /
   abstract — not buried in an appendix — and carried forward verbatim as a **stated
   limitation** in any Stage-2 document that relies on the accepted `S`. It remains
   non-blocking; it becomes a permanent disclosure obligation.
4. If TRIPPED, the report must state that a directional bias of this size was present at the
   accepted operating point and that the Stage-2 CCP labels inherit it.

### 4.5 Relation to G2† — non-redundant by construction

Since `max_c D_c ≥ T` always (§2.3), G2† PASS implies `T < 0.02`. Because `0.01 < 0.02`,
the tripwire can and does fire inside a passing gate: the live window is `T ∈ [0.01, 0.02)`.
The two statistics are therefore not redundant — the tripwire monitors panel-wide directional
drift at a finer scale than the gate adjudicates per-candidate worst case. Conversely a
tripped tripwire is never *sufficient* for a G2† failure.

---

## 5. Mandatory per-candidate signed diagnostics (Step 5)

For **every** candidate `c ∈ C` and **every** comparison
`(S, S_ref) ∈ {(200, 1000), (500, 1000), (200, 500)}`, Stage 1B **must** report:

| statistic | definition | role |
|---|---|---|
| `D_c` | `(1/N_s) Σ_s ( p_hat_{c,s,S} − p_hat_{c,s,S_ref} )` | **gating input** (via `max_c`, S_ref=1000 only) |
| `A_c` | `(1/N_s) Σ_s \| p_hat_{c,s,S} − p_hat_{c,s,S_ref} \|` | mandatory, non-gating |
| `Dmax_c` | `max_s ( p_hat_{c,s,S} − p_hat_{c,s,S_ref} )` | mandatory, non-gating (worst single seed) |
| `Dmin_c` | `min_s ( p_hat_{c,s,S} − p_hat_{c,s,S_ref} )` | mandatory, non-gating |
| `nflip_c` | `#{ s : Pass_{c,s,S} ≠ Pass_{c,s,S_ref} }` | mandatory, non-gating |
| `netflip_c` | `#{s : pass@S ∧ fail@ref} − #{s : fail@S ∧ pass@ref}` | mandatory, non-gating |
| `nboundary_c(S)` | `#{ s : \| p_hat_{c,s,S} − α \| ≤ 1e-12 }`, reported separately for `S` **and** for `S_ref` | mandatory, non-gating |
| stratum | `KNIFE` or `OFF`, from the pre-registered Stage-1 split | label |

Panel-level aggregates that **must** accompany the table:
`max_c D_c` **and the argmax candidate id**; `min_c D_c` and its argmin; `max_c A_c`;
`T` (§4); and the G1 / G3 values. Relation to G1 for cross-checking:
`G1 = (1/N_c) Σ_c A_c`, and `D_c ≤ A_c` for every `c`.

**These diagnostics cannot be hidden by the global mean.** Binding requirements:

1. The per-candidate table is printed **in full — all 10 rows** — for every comparison. No
   truncation, no "top 3", no "worst candidate only", no summarising the table by its mean.
2. It is written to a machine-readable artifact
   (`per_candidate_diagnostics.json`) alongside the human-readable report. Absence of that
   artifact makes the run **invalid**, not merely incompletely documented.
3. Any statement of a Stage-1B verdict — in the summary, the manifest, or downstream
   Stage-2 documents — **must quote `max_c D_c` together with its argmax candidate id**,
   alongside G1 and G3. A verdict reported as "G1 = x, flip rate = y, PASS" without the
   per-candidate worst case is non-compliant with this methodology.
4. The report must state, per comparison, whether the `max_c D_c` candidate is in the KNIFE
   or OFF stratum, since a KNIFE-stratum argmax and an OFF-stratum argmax carry different
   interpretations under F1.
5. The seed-cluster bootstrap and the LOCO sensitivity analysis are applied to `D_c` and to
   `max_c D_c` and reported with them, **as descriptive robustness analyses only** — they do
   not create an INDETERMINATE category and never change the G2† verdict (§7.1, §7.3).

---

## 6. The three pre-run preconditions (Step 6)

These are **pre-run conditions on the Stage-1B implementation**. Each is stated as a
requirement, followed by its verification status against the current codebase (read-only
inspection; nothing was modified or executed).

### A. MASTER SLICING — **required, and satisfied by the existing helpers**

**Requirement.** For every new master seed `s`, **exactly one** ScenarioSet is generated:

    master_s = build_scenario_set(arcs, border_delay_map, size=1000, seed=s, stochastic=True)

and it must pass the existing S1000 validation (border-event configuration match and
ScenarioSet metadata check). The smaller scenario counts are obtained **only** as literal
array prefixes of that same master:

    S200 = master_s[:200],   S500 = master_s[:500],   S1000 = master_s

taken elementwise on **both** `travel_multiplier[k]` (every arc key `k`) and
`border_delay_h[e]` (every border event `e`), with `arc_border_event`, `border_event_mean_h`,
`seed`, and `stochastic` carried over identically — i.e. exactly the semantics of
`make_nested_subset` (`scenario_count_sensitivity.py:112-124`).

**Prohibited.** Independent `build_scenario_set(seed, 200)` and `build_scenario_set(seed, 500)`
calls are forbidden anywhere in Stage 1B, for any purpose, including sanity checks.

**Why, mechanically (verified, not assumed).** `build_scenario_set` consumes a single
`np.random.default_rng(seed)` stream in **contiguous blocks of length `size`**, one block per
sorted arc key and then one per border-event key
(`baseline_uncertainty.py:1476-1541`). Changing `size` shifts the stream offset of every
block after the first, so an independently generated `size=200` set is **provably not** a
prefix of an independently generated `size=1000` set for any arc key beyond the first. This
is the same fact already recorded in the `c312e49` manifest and in the Stage-1 driver's
header comment (`a_lite_stage1_sensitivity.py:22`).

**Mandatory verification.** `verify_prefix_subset` (`scenario_count_sensitivity.py:127`) must
be run and must return `is_prefix_subset = True` for all three nestings
`(200 ⊂ 500)`, `(500 ⊂ 1000)`, `(200 ⊂ 1000)`, on **every** master seed, with the report
persisted to `nested_prefix_verification.json`. Any failure is a **hard abort**, not a
warning, and no results may be written.

**Common random numbers.** For a given `(s, S)` the *same* derived ScenarioSet is installed
once and shared by all 10 candidates. This is required for the paired within-realisation
design, and it is the source of the crossed candidate-cluster dependence recorded in F3.
It is unchanged from Stage 1.

**Relation to open item O2.** This precondition constrains **ScenarioSet construction only**.
It is deliberately agnostic to O2 (whether Stage 1B performs three evaluations per
(candidate, seed) or one S=1000 evaluation with exact prefix statistics): it holds
identically under either construction. O2 remains open and is not resolved here.

### B. TIE RULE — **required, and is exactly the production rule**

**Requirement, pre-declared.** `α = 0.90` exactly. The CCP empirical decision rule is
**inclusive**:

    candidate c passes the CCP probability test at (s, S)  ⟺  p_hat_{c,s,S}  ≥  0.90

Therefore `p_hat = 0.90` exactly counts as a **CCP probability PASS**. Every flip count,
every `Pass_{c,s,S}` indicator, and every classification statistic in Stage 1B uses this
convention and no other.

**Exact implementation form.** The indicator is taken to be the **production quantity**

    Pass_{c,s,S} = 1[ chance_vio_total_{c,s,S} ≤ TOL_HARD ],  TOL_HARD = 1e-12

(`a_lite_stage1_sensitivity.py:131, 480`; `CONFIDENCE_ONTIME = 0.90`,
`baseline_uncertainty.py:178`), where `chance_vio_total = Σ_b max(0, α − p_b)`.

**Equivalence, stated exactly rather than assumed.** The two forms differ exactly on the set
where the total shortfall `Σ_b max(0, α − p_b)` is strictly positive but no greater than
`1e-12` — which a **single** batch short of `α` by `≤ 1e-12` already achieves; two
simultaneous shortfalls are *not* required. (An earlier draft of this section stated the
condition as requiring two or more; that was wrong, and the corrected condition is the weaker
one stated here.) The conclusion is unchanged, because that set is unattainable on this grid:
every `p_b` is a multiple of `1/S` with `S ≤ 1000`, so **every attainable positive shortfall
is at least `1e-3`**, nine orders of magnitude above the tolerance. Within the Stage-1B grid
the two forms are therefore identical, and `TOL_HARD` is a floating-point guard, not a
materiality parameter. `TOL_HARD` is **not** retuned.

**The tie rule is live, not vacuous.** `α · S` is an integer at every grid point
(`180`, `450`, `900`), so `p_hat = 0.90` is exactly attainable at S = 200, 500 and 1000, and
Stage-1 observed 5–17 such cells per S. The boundary mass must be reported per §5
(`nboundary_c`).

### C. GENERATOR ASSUMPTION — **required; verified to hold. No gating problem found.**

**What is assumed i.i.d. — across scenario replications only.** Define the scenario
primitive vector for replication `j` of master seed `s`:

    Ξ_{s,j}  =  ( { M_k[j] }_{k ∈ K} ,  { B_e[j] }_{e ∈ E} )

where `K` is the set of unique arc keys `(from, to, mode)` and `E` the set of border-event
keys. The assumption required for the statistical interpretation is:

> **(i)** For each master seed `s`, the vectors `Ξ_{s,1}, …, Ξ_{s,S}` are **independent and
> identically distributed** draws from the scenario law; and
> **(ii)** across master seeds, the ScenarioSets are mutually independent.

**Verification of (i) against the generator.** Each `M_k` is
`sample_capped_mean_one_lognormal(size=S, cv, cap, rng)`, which is
`rng.lognormal(mean=μ, sigma=σ, size=S)` followed by an **elementwise**
`np.minimum(raw, cap_factor)` (`baseline_uncertainty.py:337-351`); each `B_e` is
`base_h_e ×` the same construction. The location parameter
`μ = calibrated_capped_lognormal_mu(cv, cap_factor)` is obtained by bisection on
`(cv, cap_factor)` **only** (`baseline_uncertainty.py:318-334`) — it never reads the drawn
sample. Consequences, all load-bearing:

- Entries within each `M_k` / `B_e` array are i.i.d.; the cap is a deterministic elementwise
  transform and preserves i.i.d.
- **There is no empirical re-normalisation** (no division by a sample mean, no
  post-hoc rescaling to force mean one, no variance reduction, no antithetic or stratified
  coupling). Had the mean-one property been enforced *empirically*, entries would have been
  exchangeable-but-dependent and, critically, a prefix would **not** have been a valid sample
  of its own size — which would have made precondition A unsound. It is enforced
  *analytically*, so this failure mode is absent.
- Therefore **any contiguous prefix of length `S' < S` is itself a valid i.i.d. sample of
  size `S'` with the identical marginal law.** This is precisely what licenses master slicing
  in precondition A, and it is why `S200 = master[:200]` is statistically legitimate rather
  than merely convenient.

**Verification of (ii).** Distinct integer seeds are expanded through NumPy's `SeedSequence`
entropy mixer into `PCG64` streams. Independence across seeds is the standard
pseudo-randomness assumption for this generator family; it is an assumption about the PRNG,
**not a proved probabilistic independence**, and it is stated as such. The 10 master seeds
supply the **only** independent replication in Stage 1B.

**What is explicitly NOT assumed independent — dependence within a scenario, preserved by
design.** The following structural dependencies exist and are *required* for the model to be
meaningful. No statistical claim in Stage 1B may treat these as independent:

1. **Across batches within one scenario.** Batch arrival times share arc multipliers: any two
   batches whose routes traverse a common arc key `k` both use the *same* draw `M_k[j]`, and
   any two batches crossing the same border event both use the same `B_e[j]`. Hence the 20
   per-batch on-time indicators `1[A_{c,s,b,j} ≤ LT_b]` are **dependent across `b` within a
   scenario**. The generator preserves this by drawing **one value per arc key per scenario**
   and reusing it for every batch traversing that arc — this is a modelling requirement, not
   an artifact.
2. **`p_hat` is not a binomial proportion.** It is a **minimum over 20 dependent batch-level
   proportions**. It has no Wilson-interval interpretation as a gating quantity (Wilson
   intervals remain descriptive only, unchanged from the prior review), and it is
   quantisation-biased at small `S` (F2).
3. **Across `S` within a seed.** `p_hat_{c,s,200}`, `p_hat_{c,s,500}` and `p_hat_{c,s,1000}`
   are computed on nested prefixes and are therefore **positively correlated by construction**
   (`corr ≈ √(S/S_ref)`). They are not independent samples. `D_c` is a **paired** statistic
   and is well defined under this dependence; but the paired spread understates the spread
   between two *independent* draws and must never be read as an absolute error bound.
4. **Across candidates within a seed.** All 10 candidates share the same installed
   ScenarioSet (common random numbers), so the 100 candidate × seed cells are **crossed-
   clustered**, with effective independent replication of order 10, not 100.

**Gating check required by Step 6C.** The required assumption is (i) + (ii): i.i.d. **across
scenario replications**, with all within-scenario dependence preserved. Read-only inspection
of `build_scenario_set`, `sample_capped_mean_one_lognormal`,
`calibrated_capped_lognormal_mu`, and `make_nested_subset` confirms the generator satisfies
it, including the prefix-validity property that precondition A depends on. **No gating
methodology problem is reported on this item.** The one honest qualification is that
across-seed independence (ii) rests on PRNG stream separation, which is an accepted
assumption rather than a theorem; it is unchanged from Stage 1 and is not newly introduced
here.

---

## 7. Final proposed Stage-1B gate (Step 7)

Pre-registered strata, unchanged, from Stage-1 pooled estimates (out-of-sample w.r.t.
Stage 1B's new seeds):

    KNIFE = { S1-01, S1-02, S1-03, S1-05, S1-06 }      (|pooled p̄ − 0.90| < 0.01)
    OFF   = { S1-04, S1-07, S1-08, S1-09, S1-10 }

For a comparison `(S vs S_ref = 1000)` over the 100 candidate × seed cells, **all three**
must hold:

    G1   (1/100) · Σ_{c,s} | p_hat_{c,s,S} − p_hat_{c,s,1000} |        < 0.02      [unchanged]
    G2†  max_{c ∈ C}  D_c(S, 1000)                                      < 0.02      [replaces G2]
    G3   gross flip rate among the 50 OFF-stratum cells                 < 0.10      [unchanged]

All three are **strict** `<`; a statistic exactly at its threshold FAILS.

G3's flip indicator is `1[ Pass_{c,s,S} ≠ Pass_{c,s,1000} ]` using the `Pass` indicator fixed
by precondition B (§6B) and no other convention; the gross flip rate over a cell set is the
unweighted mean of that indicator over the set (denominator 50 for the OFF stratum, 50 for
KNIFE, 100 for the all-pair diagnostic).

**Hierarchy (unchanged in shape):**

1. If `S200 vs S1000` passes `G1 ∧ G2† ∧ G3` → Stage 2 may proceed at **S = 200**.
2. Else if `S500 vs S1000` passes `G1 ∧ G2† ∧ G3` → Stage 2 may proceed at **S = 500**.
3. Else → scenario-count adequacy is unresolved and **Stage 2 does not start**. S = 1000 is
   **not** declared sufficient merely for being the largest tested.

`S200 vs S500` is computed and reported in full (including `D_c(200,500)` and `T(200,500)`)
but **remains diagnostic only and never gates**.

**`S1000` is an empirical reference, not ground truth.** No SAA claim and no theoretical
convergence claim is made for it. A PASS means the estimate and the label do not move
materially between `S` and `S1000` along a fixed realisation; it does **not** mean either
value is correct. (Unchanged from the prior review, §I.)

### 7.1 Superseded: the bootstrap indeterminacy band is REMOVED as a gate

The prior review (§G, §K clause 3) made an 80% seed-cluster bootstrap interval **gating**: a
statistic falling inside the band around its threshold was reported `INDETERMINATE` — neither
PASS nor FAIL — and Stage 2 did not start on it. **That gating role is removed.**

- There is **no** `INDETERMINATE` acceptance category. Every valid comparison resolves to
  exactly **PASS** or **FAIL**, determined solely by `G1 ∧ G2† ∧ G3`.
- The seed-cluster bootstrap and the leave-one-candidate-out (LOCO) sensitivity analysis are
  **DESCRIPTIVE ROBUSTNESS ANALYSES ONLY**. They **must not**: create an INDETERMINATE
  category; override G1/G2†/G3; turn a PASS into a FAIL; turn a FAIL into a PASS; or block
  Stage 2 independently.
- Consequently clause 3 of the prior review's Stage-2 ladder ("any gating statistic
  INDETERMINATE → treat as not passed") is **vacuous** and is struck. The ladder is otherwise
  unchanged: it remains S200 → S500 → stop, exactly as in §7 above.

This is the only change to the prior review other than G2 → G2†. It does **not** alter G1,
G2†, G3, the strata, the S grid, the seeds, or the ladder's shape.

### 7.2 Also carried over unchanged from the prior review

- **Fail-closed validity (§2.2 item 10)** precedes all three gates. This is a *validity*
  precondition, not a statistical gate: an invalid run yields no verdict at all.
- **Stage-2 gate clause 5** (mandatory Stage-2 redesign branch if F1 is confirmed) is
  unchanged and remains open as O3.
- **No multiplicity adjustment** may be applied across the three comparisons as if they were
  independent tests.
- **Wilson intervals** remain descriptive only.

### 7.3 Mandatory non-gating diagnostics

All of the following **must** be computed and reported for every comparison, including
`S200 vs S500`, and **regardless of gate outcome**. **Except for item 3's existing role as
G3, these diagnostics are non-gating. Listing G3 here for reporting completeness creates no
additional gate.** Omission of any of them is a reporting failure that invalidates the run;
apart from item 3 in its capacity as G3, none of them can change a verdict.

| # | diagnostic | definition / note |
|---|---|---|
| 1 | all-pair gross flip rate | over all 100 cells |
| 2 | KNIFE-stratum gross flip rate | over the 50 KNIFE cells |
| 3 | OFF-stratum gross flip rate | over the 50 OFF cells — **the statistic used by G3, duplicated here for reporting completeness** |
| 4 | **false-feasible count** | `#{ (c,s) : Pass_{c,s,S} = 1 ∧ Pass_{c,s,1000} = 0 }` — accepted at `S`, rejected at the reference (the over-acceptance direction) |
| 5 | **false-infeasible count** | `#{ (c,s) : Pass_{c,s,S} = 0 ∧ Pass_{c,s,1000} = 1 }` |
| 6 | **signed net classification imbalance** | `(false-feasible − false-infeasible)`, reported as a count and as a fraction of 100 |
| 7 | global mean signed probability difference | `T` (§4.1) |
| 8 | 0.01 optimism tripwire | `T` vs ±0.01, §4 — non-gating, reporting only |
| 9 | per-candidate signed `D_c` | full 10-row table, §5 |
| 10 | per-candidate absolute differences `A_c` | full 10-row table, §5 |
| 11 | exact-boundary counts | `nboundary_c(S)` and `nboundary_c(1000)`, §5 |
| 12 | **seed-cluster bootstrap** | resampling the 10 master seeds; reported as intervals around G1, `max_c D_c`, and the flip rates — **DESCRIPTIVE ONLY** |
| 13 | **leave-one-candidate-out (LOCO) sensitivity** | recompute G1, `max_c D_c` and the flip rates with each candidate omitted in turn (10 refits), to show how far each statistic depends on a single candidate — **DESCRIPTIVE ONLY** |
| 14 | per-seed breakdown | all statistics by master seed |
| 15 | √-rate check on mean |Δp| | convergence-behaviour check |
| 16 | ratio-matched scale-invariance comparison | `S500 vs S1000` against Stage-1's `S100 vs S200` |

**Items 12 and 13 are DESCRIPTIVE ONLY.** They characterise how fragile a reported statistic
is; they never determine PASS or FAIL. If a bootstrap interval straddles a threshold, or if
LOCO shows a verdict hinges on one candidate, the gate verdict **still stands as computed by
`G1 ∧ G2† ∧ G3`** and the fragility is **reported as a stated limitation** — the same
non-gating consequence class as the 0.01 tripwire (§4.4, option B).

**No contradiction check.** G2† does not contradict G1 or G3. G1 and G2† share units, scale
and strictness convention; `G1 = (1/N_c) Σ_c A_c` and `D_c ≤ A_c`, so the two are consistent
measurements of the same probability-difference object under different aggregations, and
neither implies the other (§2.3). G3 operates on classification flips, a different object
entirely, and is untouched. **G1 and G3 are therefore retained exactly as written in the
prior review.**

---

## 8. What this amendment changes and what it does not

**Changed (exactly two items):**

| | prior review | this amendment |
|---|---|---|
| G2 | `\|net classification bias\| / 100 < 0.05` | **superseded** |
| G2† | — | `max_c D_c < 0.02`, with `D_c = mean_s(p_hat_{c,s,S} − p_hat_{c,s,1000})` |
| 80% seed-cluster bootstrap band | **gating**: produced an `INDETERMINATE` verdict and blocked Stage 2 | **descriptive only**; no `INDETERMINATE` category exists (§7.1) |
| Stage-2 ladder clause 3 | "any gating statistic INDETERMINATE → treat as not passed" | **vacuous, struck** (ladder otherwise unchanged) |

**Added (previously unwritten, now explicit):** the operational definition of `D_c`; the 0.01
non-gating tripwire `T` and its concrete reporting consequence; the mandatory per-candidate
signed diagnostic table and its anti-concealment requirements; the three pre-run
preconditions (master slicing, `p_hat ≥ 0.90` tie rule, i.i.d. generator assumption); the
fail-closed validity pre-check; the full mandatory non-gating diagnostic list (§7.3),
including false-feasible / false-infeasible counts, signed net classification imbalance, and
LOCO sensitivity.

**Unchanged (not reopened):** Stage-1B necessity; S grid `{200, 500, 1000}`; all 10 frozen
candidates; 10 new pre-declared disjoint master seeds; nested-prefix construction; the
comparison set and the diagnostic-only status of `S200 vs S500`; the pre-registered
KNIFE/OFF strata; **G1**; **G3**; the 0.02 and 0.10 materiality values; Wilson intervals
descriptive only; `S1000` as an empirical reference and not ground truth; held-out validation
reserved for Stage 2; the Stage-2 gate ladder's shape (S200 → S500 → stop) including
clause 5; evaluation scope; all seven stated limitations. Open items **O2** (compute
construction) and **O3** (Stage-2 redesign branch) remain open and non-gating; this
amendment resolves **O1 only**.

**Traceability note on the bootstrap change.** The gating indeterminacy band is present in
the saved review file at lines 210–212 and 317, and this amendment originally carried it over
verbatim for that reason. Its removal was agreed in conversation but was **not** recorded in
either saved document — the same class of traceability defect that motivated this amendment
in the first place. §7.1 now records it durably.

**Process state.** Nothing was implemented, generated, run, or committed. No Stage-1B
artifact exists. The historical review file was not modified.
