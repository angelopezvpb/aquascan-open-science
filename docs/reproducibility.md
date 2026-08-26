# Reproducibility

## Reproducible Experiments

Every AquaScan experiment must be reproducible by an independent researcher.

### Experiment Record

Each experiment records:
```yaml
experiment_id: "exp-2025-001"
git_commit: "abc123"
dataset_version: "aquascan-igme-v1"
schema_version: "1.0.0"
random_seed: 42
model: "logistic_regression"
hyperparameters:
  C: 1.0
  class_weight: "balanced"
configuration: { ... }
metrics: { ... }
timestamp: "2025-01-15T10:30:00Z"
```

### Fingerprint

```bash
aquascan experiment fingerprint --config configs/experiment.yaml
```

Generates a SHA-256 based fingerprint from:
- Experiment configuration
- Dataset version
- Git commit (if available)

This fingerprint uniquely identifies the experiment setup and can be
cited in papers and reports.

### Reproducibility Checklist

- [ ] Dataset version documented
- [ ] Random seed specified
- [ ] Split strategy documented
- [ ] Model and hyperparameters recorded
- [ ] Git commit recorded
- [ ] Configuration file preserved
- [ ] Metrics saved in machine-readable format

## Environment

Reproducibility requires:
1. Same code version (git commit)
2. Same dataset version
3. Same random seed
4. Same hyperparameters
5. Same evaluation protocol

Use `pyproject.toml` dependencies with version ranges to manage
the Python environment.

## Limitations

Full bitwise reproducibility may not be achievable due to:
- Floating-point differences across hardware/OS
- Non-deterministic operations in some libraries
- External data access timing

The goal is *scientific* reproducibility — same methodology, same
conclusions — not necessarily identical floating-point results.