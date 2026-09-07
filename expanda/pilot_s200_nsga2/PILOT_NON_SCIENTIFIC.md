# NSGA-II PILOT — NON-SCIENTIFIC. DO NOT USE AS A RESULT.

This directory holds ONE exploratory pilot run.

- It is **NOT** one of the formal 30 independent runs.
- It **MUST NEVER** be pooled into, compared against, or reported alongside
  the formal 30-run statistical comparison.
- Its algorithm seed (990001) is a dedicated PILOT seed and is deliberately
  outside the formal run-seed block, so it can never be mistaken for, or
  silently reused as, one of the 30.

Purpose: exercise the real downstream NSGA-II + CCP configuration end to end
at the scenario size selected by Stage-1B (S = 200), before any formal run.

Provenance of S = 200
- Stage-1B full study, 300 evaluations, commit 032d23e.
- Frozen ladder: S200 vs S1000 passed G1 = 0.01589, G2dagger = +0.0148,
  G3 = 0.08, all strict '<'. selected_S = 200.
- Carried limitation: the DESCRIPTIVE bootstrap/LOCO diagnostics show
  fragility around the S200 thresholds. Under the pre-frozen ladder this does
  not change the selection, and S was NOT switched to 500 post hoc.

Nothing here may be written into a_lite_stage1b_results/, outputs/, or any
formal 30-run directory.
