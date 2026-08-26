# AquaScan Experiments

This directory tracks reproducible experiments.

## Structure

Each experiment should be a subdirectory with:
- `config.yaml` — experiment configuration
- `README.md` — description and results
- `results/` — metrics, figures, output

## Running an Experiment

```bash
python scripts/run_baseline.py \
  --input data/processed/dataset.parquet \
  --baseline logistic \
  --output experiments/exp_001/
```

## Fingerprint

```bash
aquascan experiment fingerprint --config experiments/exp_001/config.yaml
```