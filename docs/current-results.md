# Current Canonical Results (v0.2)

Only leakage-free, canonical results are reported here. The leaked Gate 8 v1
numbers appear **only** in the postmortem that explains the failure mode.

## TARGET_DEPTH_TO_WATER — open-source context benchmark

| Evaluation regime | Baseline v2 (RF) MAE | GeoAI Context v0 (RF) MAE |
|---|---|---|
| IN_DOMAIN_CONTEXT_BASELINE | ~10.00 m | ~9.94 m |
| SPATIAL_HOLDOUT_CONTEXT_BASELINE | ~9.45 m | ~9.54 m |
| CROSS_DOMAIN_CONTEXT_BASELINE | ~15.67 m | ~15.39 m |

- Modeling population: **563** GT-A/B depth records (10 Phase 0 sites
  firewalled).
- Dummy (median) in-domain MAE: ~21.5–24.4 m depending on fold.
- **GeoAI Context v0 does NOT meaningfully outperform Baseline v2.** The
  documentary/context signal has limited headroom.

## Known generalization failure
`KNOWN_OOD_DOMAIN = Región de los arenales` — cross-domain evaluation fails
catastrophically there (MAE far worse than Dummy). This is documented, not
averaged away.

## What is NOT reported
- **No ERT/seismic groundwater accuracy exists.** Physical acquisition has
  not been performed; Gate 11 hardware acceptance is NOT READY.
- **No dry-well detection accuracy exists.** Confirmed negatives = 0.
- No Phase 0 sealed Ground Truth has been revealed.

See also: `reports/` (curated aggregate results).
