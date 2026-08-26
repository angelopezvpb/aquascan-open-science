# Baseline v1 Leakage — Postmortem

This document exists so the incident is never repeated and never hidden.

## What happened
1. **Baseline v1** initially produced implausibly strong performance
   (depth-to-water MAE ≈ 0.21 m in-domain).
2. A **forensic audit** (Gate 8.1) found that `_target_*` columns were
   permitted as input features, i.e. the target leaked into the predictors.
3. The original Gate 8 benchmark was marked
   `INVALID_FOR_SCIENTIFIC_COMPARISON` (`reports/gate8_v1_supersession.json`).
4. A **fail-closed feature policy** was introduced: any column matching
   `_target_*`, `target_*`, `gt_*`, `readiness_*`, `phys_val_*` is forbidden as
   an input; unknown/unclassified columns are excluded by default.
5. **Baseline v2** was re-frozen on the leakage-free pipeline. Honest
   in-domain RF MAE ≈ 10 m (Dummy ≈ 21.5 m).

## Rule
**The leaked performance (≈0.21 m MAE) must NEVER be presented as AquaScan
accuracy.** It appears here only as the documented failure mode.

## Tests that guard against regression
- `_target_*` can never enter predictors (fail-closed tests).
- Unknown columns fail closed.
- Phase 0 firewall tests.
- Target-shuffle collapse tests.
