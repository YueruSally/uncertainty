# A-lite Stage 1B — Independent Claude Methodology Review

Status: **METHODOLOGY NOT YET CONVERGED** — one gating disagreement (acceptance rule), two
non-gating disagreements. No code written, nothing run, nothing committed.

Reviewed against verified artifacts:
- candidate provenance commit `feb4cf2`
- audited driver commit `c312e49`
- Stage-1 results `expanda/a_lite_stage1_results/` (raw_rows.csv, per_S_summary.json, manifest.json)

All numbers below were recomputed by Claude from `raw_rows.csv`, not taken from the brief.

---

## 0. Three findings that change the review

### F1 — The gross classification flip rate is non-convergent, so Codex's rule cannot terminate

Decomposing the 100 candidate x seed pairs by each candidate's cross-seed pooled estimate
`p_bar` (10 seeds x S=200, pooled SE = 0.0067):

| candidate | pooled p_bar | dist from 0.90 | stratum |
|---|---|---|---|
| S1-01 | 0.8950 | 0.0050 | KNIFE-EDGE |
| S1-02 | 0.8950 | 0.0050 | KNIFE-EDGE |
| S1-03 | 0.8960 | 0.0040 | KNIFE-EDGE |
| S1-05 | 0.8995 | 0.0005 | KNIFE-EDGE |
| S1-06 | 0.8995 | 0.0005 | KNIFE-EDGE |
| S1-04 | 0.9250 | 0.0250 | off-boundary |
| S1-07 | 0.9190 | 0.0190 | off-boundary |
| S1-08 | 0.9410 | 0.0410 | off-boundary |
| S1-09 | 0.9475 | 0.0475 | off-boundary |
| S1-10 | 0.9655 | 0.0655 | off-boundary |

Stratified flip rates versus S=200:

| S | gross flip (all 100) | KNIFE flip (50) | off-boundary flip (50) | net bias |
|---|---|---|---|---|
| 50  | 0.18 | 0.30 | 0.06 | +0.04 |
| 100 | 0.20 | 0.34 | 0.06 | +0.02 |

The entire "material classification instability" verdict is produced by 5 of 10 candidates whose
true on-time probability is statistically indistinguishable from alpha = 0.90. The off-boundary
stratum already satisfies the 0.10 criterion at S = 50, and does not improve from 50 to 100.
The knife-edge stratum gets *worse* from 50 to 100.

Theory for why this cannot be fixed by more scenarios. For nested prefixes,
`corr(p_S, p_Sref) = sqrt(S/Sref)`, so for a candidate whose true p equals alpha exactly the
probability that the two estimates land on opposite sides of alpha is

    P(flip | p = alpha) = arccos( sqrt(S/Sref) ) / pi

which depends **only on the ratio S/Sref** and never on absolute S. Calibration against Stage-1:

| comparison | ratio | model P(flip \| knife) | observed KNIFE flip |
|---|---|---|---|
| S50 vs S200  | 0.25 | 0.333 | 0.30 |
| S100 vs S200 | 0.50 | 0.250 | 0.34 |

Applied forward to Codex's proposed grid, with 50 knife-edge pairs out of 100:

| comparison | ratio | knife contribution to gross flip rate | + off-boundary (~0.06) | predicted gross |
|---|---|---|---|---|
| S200 vs S1000 | 0.20 | 0.176 | +0.03 | **~0.19 - 0.21** |
| S500 vs S1000 | 0.50 | 0.125 | +0.03 | **~0.15** |
| S200 vs S500  | 0.40 | 0.141 | +0.03 | **~0.17** |
| S1000 vs S5000| 0.20 | 0.176 | +0.03 | ~0.19 (unchanged) |

**Prediction, pre-registered before Stage 1B runs: under Codex's rule, S200 and S500 will both
be rejected, Stage 2 will be blocked, and the identical outcome will recur at every larger S.**
The rule has no termination condition. That is not conservatism; it is a non-terminating gate.

### F2 — p_hat drifts monotonically downward in S; small S systematically over-accepts

| S | mean p_hat over 100 pairs | mean # binding batches | pairs with p_hat exactly 0.900 |
|---|---|---|---|
| 50  | 0.9270 | 1.67 | 10/100 |
| 100 | 0.9218 | 1.23 | 17/100 |
| 200 | 0.9183 | 1.06 | 5/100 |

`min_on_time_prob` is a minimum over 20 batch-level empirical probabilities. Coarse quantisation
at small S rounds that minimum up and creates ties (1.67 tied binding batches at S=50 vs 1.06 at
S=200), so the statistic is **optimistically biased at small S and drifts down as S grows**
(-0.0087 from S50 to S200; -0.0035 from S100 to S200). The drift has not plateaued.

Consequences Codex's design does not capture:
- the persistent asymmetry (11 pass->fail vs 7 and 9 fail->pass) is a *systematic* effect, not
  symmetric noise;
- S=1000 is not a neutral reference — it is deterministically stricter than S=200;
- the decision-relevant statistic for a one-sided `p >= alpha` test is the **signed / net**
  classification bias, not the gross flip rate;
- 17/100 pairs sit exactly on 0.900 at S=100 and pass only by the inclusive `>=` convention;
  alpha*S is an integer at every S in every proposed grid, so this boundary mass persists.

### F3 — The gate has no stated sampling uncertainty

The 100 pairs are crossed-clustered: 10 candidate clusters and 10 seed clusters. Effective
independent replication is of order 10, not 100. A flip rate of 0.20 carries a cluster-robust
standard error of roughly 0.04 - 0.06. Therefore an observed statistic of 0.09 or 0.12 cannot be
distinguished from the 0.10 threshold, and a hard PASS/FAIL at that point is adjudicated by noise.
Codex's rule states no indeterminacy band, so a borderline Stage-1B result would be over-read.

---

## A. Stage-1B necessary: **YES**

But for a partly different reason than Codex gives.

S=200 cannot be defensibly selected from Stage 1 alone: it was the largest member of its own grid,
it was also the scan configuration used to *select* the panel, and nothing has ever been compared
against anything larger. Accepting it now would be accepting the reference because it is the
reference. So a larger-reference stage is required.

The additional reason: Stage 1B is now a genuine out-of-sample test of the F1/F2 structural
diagnosis. F1 predicts specific numbers (knife stratum ~0.35, off-boundary ~0.06, gross ~0.20 at
ratio 0.2, mean |dp| ~0.013 - 0.015). If those land, scenario-count adequacy is settled as the wrong
frame for the residual instability and the project moves to the right question. If instead the
knife stratum collapses toward 0.10 at S500 vs S1000, the sample-size story survives and Codex's
rule is vindicated. Either outcome is decisive, which is what makes the stage worth running.

## B. Recommended scenario grid: **S = {200, 500, 1000}** (agree with Codex)

- `{200, 1000}` — rejected: no intermediate operating point, and no within-stage ratio replicate,
  so no internal scale-invariance diagnostic.
- `{200, 400, 800}` — rejected: cleanest ratio ladder (0.5, 0.5, 0.25) but neither 400 nor 800 is a
  plausible Stage-2 operating point, and reach is 4x rather than 5x.
- `{200, 500, 2000}` — rejected: +59% scenario cost for reference precision that is already
  sufficient once pooled across seeds (pooled SE 0.0021 vs 0.0030), and it drops the primary ratio
  to 0.10, which under F1 inflates the gross flip rate without adding decision value.
- `{200, 500, 1000}` — chosen: 5x reach; 500 is the realistic Stage-2 fallback budget; pooled
  reference SE 0.0030 resolves a 0.01 knife-edge band; and `500 vs 1000` is a **ratio-matched
  replicate of Stage-1's `100 vs 200`**, which is the sharpest available test of F1.

Added requirement (not in Codex's design): report `S500 vs S1000` (ratio 0.5) directly against
Stage-1's `S100 vs S200` (ratio 0.5) as a first-class scale-invariance diagnostic.

## C. Candidate policy: keep all 10 frozen candidates unchanged (agree with Codex)

Restricting to Stage-1 flippers would be post-hoc selection on the outcome variable: it conditions
on inspected realisations, mechanically inflates the flip rate, and destroys the meaning of the
denominator. Reject that.

Added requirement: report every comparison **stratified** by the pre-registered KNIFE / off-boundary
split, because F1 shows the aggregate is almost entirely a property of panel composition.

Limitation to state in any write-up: the panel was selected inside a 0.86 - 0.96 band at S=200,
seed 42, so it is deliberately enriched in near-threshold candidates and, by regression to the mean,
also in candidates that merely drew near-threshold at that one realisation. The aggregate flip rate
is therefore conditional on a boundary-enriched panel and does not generalise to a random candidate
population. It does not bias the within-seed paired contrasts, because Stage-1 seeds (700001-700010)
are disjoint from the selection seed.

## D. Seed policy: 10 completely new pre-declared master seeds (agree with Codex)

Reusing Stage-1 seed identities with larger masters buys nothing. The manifest for `c312e49` records
that independent `build_scenario_set(seed, S)` calls do **not** form prefixes, so a reused seed at
S=1000 would not reproduce Stage-1's S=200 realisations anyway — the "continuity" is illusory.
Meanwhile reuse would condition the test on realisations already inspected, and the knife-edge
stratification was derived from exactly those realisations.

New seeds make Stage 1B a clean independent replication and a true out-of-sample test of F1.
Seeds must be declared before any Stage-1B evaluation and be disjoint from
`{0, 42, 43, 44, 45, 46, 1000, 1000003, 700001..700010}`.

## E. Nested construction: exact prefixes required (agree with Codex, with a stronger method)

One master ScenarioSet per seed; `S200 = master[:200]`, `S500 = master[:500]`,
`S1000 = master[:1000]`; never independently generated. Reasons: only nesting makes the contrast a
paired within-realisation measurement that isolates the marginal effect of adding scenarios;
independent redraws would confound sample size with realisation variance; and the generator is not
prefix-stable across independent calls.

Stronger method available. Because the candidate decision is frozen, evaluation is a pure forward
simulation and is **separable per scenario**. I verified this empirically: `max_late_excess_h` is
monotone nondecreasing along the nested prefix in 100/100 Stage-1 pairs. Therefore evaluate **once
at S=1000 per (candidate, seed)** and derive S=200 and S=500 as exact prefix statistics (prefix mean
per batch for the CCP probability, prefix max for lateness). This makes nesting true by construction
rather than by post-hoc verification, removes an entire class of nesting bugs, and cuts scenario
cost from 1700 to 1000 per (candidate, seed). Keep the prefix-verification hash check as a tripwire.

Caveat to state: nesting makes `p_200` and `p_1000` positively correlated, so the paired `|dp|`
understates the spread between two *independent* S=200 and S=1000 draws. That is the right choice
for an adequacy question, but `|dp|` must not be read as an absolute error bound.

## F. Paired comparisons (agree with Codex on the set, disagree on the role of one)

Primary: `S200 vs S1000`, `S500 vs S1000`.
Diagnostic: `S200 vs S500`.
Added diagnostic: `S500 vs S1000` against Stage-1's `S100 vs S200` (ratio-matched, per F1).
All comparisons reported stratified by KNIFE / off-boundary, with signed as well as absolute deltas.

`S200 vs S500` must remain **diagnostic only** and must not gate — see G. The three comparisons are
mutually dependent (shared prefixes, shared seeds), so no multiplicity adjustment may be applied as
if they were independent tests.

## G. Exact pre-declared acceptance rule (**DISAGREE with Codex — gating**)

Pre-register the strata now, from Stage-1 pooled estimates, which are fully out-of-sample with
respect to Stage 1B:

    KNIFE = { S1-01, S1-02, S1-03, S1-05, S1-06 }    (|pooled p_bar - 0.90| < 0.01)
    OFF   = { S1-04, S1-07, S1-08, S1-09, S1-10 }

For a comparison (S vs S_ref) over 100 candidate x seed pairs, **all three** must hold:

    G1  mean |p_S - p_Sref| over all 100 pairs                 < 0.02
    G2  | #(pass@S & fail@ref) - #(fail@S & pass@ref) | / 100  < 0.05   [net classification bias]
    G3  gross flip rate among the 50 OFF pairs                 < 0.10

Indeterminacy band (addresses F3): if any gating statistic falls inside an 80% seed-cluster
bootstrap interval around its threshold, the comparison is reported **INDETERMINATE**, not PASS and
not FAIL, and Stage 2 does not start on it.

Reported descriptively, never gating: gross flip rate over all 100 pairs; KNIFE-stratum flip rate;
per-candidate and per-seed breakdowns; mean signed delta; count of pairs sitting exactly on 0.900;
the sqrt-rate check on mean |dp|; the ratio-matched scale-invariance comparison.

Hierarchy, unchanged in shape from Codex: accept S=200 if `S200 vs S1000` passes G1-G3; else accept
S=500 if `S500 vs S1000` passes G1-G3; else adequacy is unresolved and Stage 2 does not start.
`S200 vs S500` is reported but does not gate.

Why I reject Codex's rule, point by point:

1. **The gross flip rate is not a valid adequacy criterion.** Per F1 it is scale-invariant for
   knife-edge candidates and cannot be driven below 0.10 by any S. A criterion that no experiment
   can satisfy does not measure adequacy; it measures panel composition.
2. **Requiring `S200 vs S500` in addition to `S200 vs S1000` adds almost no strictness and a real
   false-rejection risk.** Under F1, ratio 0.4 is a *weaker* perturbation than ratio 0.2, so the
   extra condition is nearly implied — but "nearly" is not "always", because empirical threshold
   classifications are non-monotone, so it functions mainly as a second chance to fail on noise
   across 100 correlated pairs. Codex's stated justification is conservatism, which is not a
   scientific justification for a redundant gate.
3. **Codex's rule never checks that the reference is stabilising.** It correctly denies that S=1000
   is ground truth, then makes acceptance depend entirely on agreement with it, with no internal
   evidence that the estimator is behaving like a converging Monte-Carlo estimator. G's descriptive
   sqrt-rate check and the ratio-matched diagnostic supply that evidence.
4. **F2 shows the decision-relevant error is one-sided.** Small S systematically over-accepts. A
   two-sided gross flip count mixes that real bias together with symmetric knife-edge noise and
   dilutes exactly the signal that matters. G2 isolates it.

**Honest disclosure — this is why the user must scrutinise my proposal rather than accept it.**
My rule is more permissive than Codex's and it retroactively changes a Stage-1 conclusion:

| | G1 | G2 | G3 | my verdict | Codex verdict |
|---|---|---|---|---|---|
| S=50 vs S200  | 0.0240 FAIL | +0.04 pass | 0.06 pass | REJECT | REJECT |
| S=100 vs S200 | 0.0155 pass | +0.02 pass | 0.06 pass | **ACCEPT** | **REJECT** |

Under my rule Stage 1 would have concluded that S=100 is acceptable relative to S=200. (It would
still not have established that S=200 itself is adequate, so Stage 1B remains necessary either way.)
I did not choose these thresholds to secure a pass: S=50 still fails on the unchanged G1, and S=50's
net bias of 0.04 is close to breaching G2.

**The strongest argument for Codex's position, stated fairly.** NSGA-II under a CCP constraint is
attracted *to* the constraint boundary, because cost falls as reliability falls. So the population
the optimiser actually visits during search will be far more knife-edge-concentrated than this
panel, not less. On Codex's reading, that means the gross flip rate is precisely the decision-relevant
quantity — a ~30% mislabelling rate exactly where the search spends its time — and my G3
stratification discards the only stratum that matters. That argument is serious and I cannot refute
it from the data in hand.

What I would say against it, without claiming to settle it: even granting the premise, the gross
flip rate still is not a *scenario-count* criterion, because F1 shows no S removes it. If boundary
mislabelling is the real worry, the remedy is a different mechanism — optimise at a margin
(alpha' > alpha) and verify at alpha; hold the ScenarioSet fixed during search so labels are at
least self-consistent; and re-evaluate the final Pareto set independently at high S. Those are
Stage-2 design changes, not a larger S. So the knife-edge finding reframes Stage 2 more than it
reframes the choice of S.

This is a genuine disagreement about which statistic measures scenario-count adequacy, it changes
what Stage 1B concludes and whether Stage 2 starts, and **I am not resolving it by assumption.**

## H. Materiality thresholds: keep 0.02 and 0.10 numerically unchanged (agree with Codex)

Neither number is loosened. Three notes:

- The original rationale for 0.02 ("one S=50 quantisation step") does not transfer to the new grid,
  where one quantisation step is 0.002 at S=500 and 0.001 at S=1000. Keep the value, but re-justify
  it as a decision-relevance threshold rather than a resolution threshold.
- 0.02 has little headroom at the new design point. Calibrating the nested half-normal model on
  Stage-1 (factor 0.867) predicts mean |dp| of 0.0131 for S200 vs S1000 and 0.0066 for S500 vs S1000.
  So G1 will probably pass at both, and G1 will not be the binding criterion. Expect the verdict to
  turn on the classification criterion, whichever form it takes.
- The 0.10 value is carried over unchanged into G3; only the denominator changes, and that change is
  justified structurally (F1), not by inspecting a result.

## I. Statistical interpretation (agree with all four Codex points, plus two additions)

Agreed: master seeds are the only independent replication; nested S estimates are positively
correlated by construction; the 100 candidate x seed comparisons are **not** 100 independent
Bernoulli trials; S=1000 is not truth and no SAA or theoretical convergence claim is made.

Additions:
- The dependence is **crossed**, not merely clustered: 10 candidate clusters and 10 seed clusters.
  Effective replication is of order 10. Any statistic must carry a seed-cluster (and ideally
  candidate-cluster) bootstrap interval — hence the F3 indeterminacy band in G.
- `p_hat` is a **minimum over 20 batch-level probabilities**, not a single binomial proportion. It is
  quantisation-biased at small S with a monotone downward drift in S (F2). The paired design partly
  conceals this, so mean *signed* delta and the exact-0.900 boundary mass must both be reported
  alongside the absolute delta.

## J. Held-out validation: keep it outside Stage 1B (agree with Codex)

Stage 1B's question is internal: how much does the estimate and the label move as S grows along a
fixed realisation? A held-out set answers a different question — does the label survive on scenarios
never used — which is the Stage-2 search-time selection-bias question. Mixing them would blur both.

Note that the knife-edge conditioning objection that would normally motivate a held-out set is
already dissolved: the KNIFE / OFF strata are pre-registered in G from Stage-1 data, which is
out-of-sample with respect to Stage 1B's new seeds. Recomputing the strata from Stage-1B's own
pooled reference should be reported as a sensitivity check only, never as the gating definition.

## K. Exact Stage-2 gate

1. `S200 vs S1000` PASSES G1-G3  -> Stage 2 may proceed at S = 200.
2. Else `S500 vs S1000` PASSES G1-G3 -> Stage 2 may proceed at S = 500.
3. Any gating statistic INDETERMINATE -> treat as not passed; Stage 2 does not start on it.
4. Neither passes -> scenario-count adequacy unresolved; S=1000 is **not** declared sufficient merely
   for being the largest tested; Stage 2 does not start.
5. Added: if Stage 1B confirms F1 (KNIFE flip rate approximately at the arccos prediction and flat in
   absolute S, off-boundary flip rate below 0.10), then scenario count is **not** the binding issue
   and Stage 2 must be redesigned around boundary handling — search margin, fixed common random
   numbers during search, independent high-S re-evaluation of the final Pareto set — before it
   starts, regardless of which S is chosen.

Codex's clauses 1-4 are scientifically defensible in shape. Clause 5 is mine and is the part Codex's
gate lacks: Codex's gate can only ever say "S too small", so it has no path to the conclusion that
the residual instability is not about S at all.

## L. Evaluation count

Scope agreed: 10 candidates x 10 new seeds x 3 S values = 300 fixed-candidate evaluations, no
optimisation. Two corrections to the framing:

- Cost is not comparable to Stage 1 unit-for-unit. Stage-1 was 350 scenario-evaluations per
  (candidate, seed); Stage-1B as Codex specifies it is 1700, i.e. ~4.9x, or ~1.7M scenario
  evaluations in total.
- Under E's prefix construction the real work is **100 forward evaluations at S=1000** (1000 per
  candidate-seed, ~1.0M total), with the 300 comparisons derived exactly from prefixes. Same
  science, ~41% less compute, and nesting correct by construction.

## M. Main limitations

1. Panel composition, not the model, determines the aggregate flip rate: 5 of 10 candidates are
   statistically indistinguishable from alpha.
2. Panel selected at S=200 / seed 42 inside a 0.86 - 0.96 band, so it is boundary-enriched and subject
   to regression to the mean; aggregate rates do not generalise to a random candidate population.
3. Binding-batch degeneracy carried over from `feb4cf2`: below-threshold candidates all bind on batch
   17, p_scan = 0.900 candidates all bind on batch 2. The panel is not representative across all 20
   batches, and the knife-edge stratum is therefore also largely a batch-17 stratum. This confound
   cannot be separated within the frozen panel.
4. S=1000 is a larger empirical reference only, and F2 shows it is deterministically stricter than
   S=200 rather than neutral.
5. Effective replication is of order 10, so gating statistics near their thresholds are not resolvable.
6. Stage 1B says nothing about search-time selection bias; the boundary-attraction argument in G is
   unresolved and is a Stage-2 question.
7. `p_hat` quantisation interacts with the inclusive `>=` test at every S in the grid, since alpha*S
   is an integer throughout.

## N. Claude-vs-Codex agreement matrix

| Issue | Codex position | Claude position | Verdict |
|---|---|---|---|
| Stage-1B necessity | Necessary | Necessary (plus: as a pre-registered out-of-sample test of F1) | **AGREE** |
| S grid | {200, 500, 1000} | {200, 500, 1000} | **AGREE** |
| Candidate panel | All 10 frozen, unchanged | All 10 frozen, unchanged, reported stratified KNIFE/OFF | **AGREE** |
| Seed policy | 10 new pre-declared disjoint seeds | 10 new pre-declared disjoint seeds | **AGREE** |
| Nesting | One S=1000 master per seed, exact prefixes | Same, computed as prefix statistics of one S=1000 forward evaluation | **AGREE** (method strengthened) |
| Paired comparisons | S200vS1000, S500vS1000 primary; S200vS500 diagnostic | Same set, plus ratio-matched S500vS1000 vs Stage-1 S100vS200; S200vS500 strictly non-gating | **AGREE** |
| S200 acceptance rule | Non-material vs BOTH S500 and S1000, on mean\|dp\| and gross flip rate | Non-material vs S1000 only, on G1 mean\|dp\| + G2 net bias + G3 off-boundary flip rate, with an indeterminacy band | **DISAGREE — GATING** |
| Materiality thresholds | 0.02 and 0.10 unchanged | 0.02 and 0.10 unchanged, plus new G2 at 0.05; 0.02 re-justified | **AGREE on values**; the disagreement is over which statistics they apply to (same issue as above) |
| Statistical interpretation | Seeds independent; nested correlated; not 100 Bernoulli; S1000 not truth | All of that, plus crossed clustering with effective n ~ 10, and p_hat is a biased min-statistic drifting downward in S | **AGREE** (extended) |
| Wilson intervals | Descriptive only, not the gate | Descriptive only; gating uncertainty via seed-cluster bootstrap instead | **AGREE** |
| Held-out validation | Not in Stage 1B; reserve for Stage 2 | Not in Stage 1B; reserve for Stage 2 | **AGREE** |
| Stage-2 gate | S200 -> S500 -> stop | Same ladder, plus INDETERMINATE handling and a mandatory redesign branch if F1 confirms | **DISAGREE — non-gating** (inherits the rule disagreement; the added branch is additive) |
| Computational scope | 300 evaluations | 300 derived comparisons from 100 S=1000 forward evaluations | **DISAGREE — non-gating** (efficiency and correctness, not science) |

## O. Unresolved methodology disagreements

**O1 (GATING) — which statistic measures scenario-count adequacy.**
Codex gates on the gross classification flip rate over all 100 pairs, and requires S=200 to clear it
against both S=500 and S=1000. I hold that the gross flip rate is scale-invariant for knife-edge
candidates (F1, calibrated on Stage-1 to within a few points), so it cannot fall below 0.10 at any S,
making Codex's gate non-terminating; and that the decision-relevant error is one-sided (F2), so the
net bias plus an off-boundary flip rate is the correct pair of criteria. Codex's counter-position —
that NSGA-II is attracted to the boundary, so the knife-edge stratum is exactly what matters — is
serious and cannot be refuted from present data. The two rules give different verdicts on Stage-1's
S=100 and, by F1's prediction, on Stage-1B's S=200 and S=500. **Not resolved. Requires a decision
before Stage 1B is implemented.**

**O2 (non-gating) — compute construction.**
Whether Stage 1B performs three separate evaluations per (candidate, seed) or one S=1000 evaluation
with exact prefix statistics. Separability is verified (100/100 monotone). This does not affect any
scientific conclusion.

**O3 (non-gating) — Stage-2 redesign branch.**
Whether the Stage-2 gate should include a mandatory redesign path for the case where F1 is confirmed
and scenario count is shown not to be the binding issue. Additive to Codex's ladder.

## P. Final verdict

**METHODOLOGY NOT YET CONVERGED**

**READY FOR STAGE-1B IMPLEMENTATION DESIGN: NO**

Blocked on O1. Nothing was implemented, generated, run, modified, or committed.
