# GeoAI Context v0

Open-source scikit-learn model that predicts `TARGET_DEPTH_TO_WATER` from
documentary/contextual features only. It is a **research baseline**, not a
groundwater detector.

## Model

- Family: RandomForest (scikit-learn).
- Target: `TARGET_DEPTH_TO_WATER` (primary); hydraulic head and productivity are
  exploratory.
- Training population: 563 GT-A/B depth records; 10 Phase 0 sites firewalled.
- Feature policy: fail-closed; `_target_*`, `target_*`, `gt_*`, `readiness_*`
  and `phys_val_*` are forbidden as inputs.

## Evaluation (leakage-free)

| Regime | Baseline v2 (RF) MAE | GeoAI Context v0 (RF) MAE |
|---|---:|---:|
| In-domain | ~10.00 m | ~9.94 m |
| Spatial holdout | ~9.45 m | ~9.54 m |
| Cross-domain | ~15.67 m | ~15.39 m |

Dummy (median) in-domain MAE: ~21.5–24.4 m.

**GeoAI Context v0 does not meaningfully outperform Baseline v2.** The
documentary signal has limited headroom.

## Limitations

- No ERT/seismic field validation.
- No confirmed dry-well Ground Truth.
- Historical Ground Truth (median of dated observations; not current level).
- Known OOD failure: `Región de los arenales`.
- Not a substitute for a hydrogeological study.

## Weights

The `.joblib` weights are **not distributed in v0.2**. The model card,
feature policy, confidence policy and OOD policy are public so the evaluation
protocol can be audited.

## Files

- `feature_manifest.json` — permitted feature set.
- `confidence_policy.json` — confidence bands.
- `uncertainty_calibrator.json` — calibration artifact.
- `ood_policy_reference.json` — out-of-distribution policy.
- Model card: [`reports/gate9_model_card_v0.json`](../../reports/gate9_model_card_v0.json).
- Training/evaluation entry point: [`scripts/gate9_geoai_context.py`](../../scripts/gate9_geoai_context.py)
  (requires the validated dataset, which is not distributed).

## License

Apache-2.0.
