# Intellectual Property Boundary

## Public Domain (This Repository)

The following components and data are intentionally public:

### Data & Schemas
- AquaScan canonical dataset specification
- JSON Schema definitions
- Public source data extractors (IGME, etc.)
- Data normalization and validation tools
- Provenance tracking infrastructure

### Processing & Analysis
- Deterministic ERT processing (geometric factors, apparent resistivity)
- Deterministic seismic processing (first-break picking, Vp estimation)
- QA/QC scoring algorithms (public domain methods)
- Ground Truth quality scoring

### Machine Learning
- Baseline models: rule-based, logistic regression, random forest
- Feature engineering pipelines
- Evaluation metrics and split strategies
- Abstention analysis framework

### Research
- Experimental protocols
- Configuration templates
- Reproducibility infrastructure
- Simulation/synthetic data generators

### Documentation
- Architecture and design documents
- Scientific methodology
- Benchmark specifications

---

## Proprietary Layer (NOT in This Repository)

The following are intentionally excluded and kept in private infrastructure:

### GeoAI Production
- Production multimodal neural architectures
- Trained model weights and checkpoints
- Ensemble and stacking strategies
- Production inference pipelines

### Calibration AI (AutoCal)
- Advanced calibration algorithms
- Machine learning-based quality enhancement
- Proprietary noise reduction methods
- Adaptive acquisition optimization

### Private Data
- Client borehole logs and Ground Truth
- Private survey coordinates and results
- Regional proprietary models
- Commercial project data

### Operations
- Internal procedures and protocols
- Commercial infrastructure configuration
- Client-specific customizations
- Business-sensitive information

---

## Interface Placeholders

Where architectural clarity requires referencing proprietary components,
code uses **documented placeholders and interfaces** only:

```python
# Example: placeholder for proprietary AutoCal integration
autocal_version: Optional[str] = Field(
    default=None, description="Version of AutoCal used (placeholder for proprietary system)."
)
```

No actual proprietary implementation is simulated or exposed.

---

## Contribution Guidelines

All contributions to this repository must:
1. Use only public-domain or open-source methods
2. Not include trained weights or production algorithms
3. Not include client data or configurations
4. Respect the IP boundary defined here

See [CONTRIBUTING.md](../CONTRIBUTING.md) for full guidelines.