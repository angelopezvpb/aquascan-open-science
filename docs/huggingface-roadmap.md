# Hugging Face Publication Roadmap

## Current Status

Hugging Face is **not yet integrated** as a dependency. All data and models
currently live in this GitHub repository and local storage.

## Planned Integration

When datasets are ready for public release, the architecture will be:

```
GitHub (this repo)
├── Code (src/aquascan/)
├── Experiment protocols
├── Configuration
└── Documentation

Hugging Face
├── AquaScan-25 dataset
├── AquaScan-100 dataset
├── Public baseline model weights
└── Dataset cards and documentation
```

## Dataset Roadmap

### AquaScan-25 (v0.3-v0.4)
- 25 validated sites with complete Ground Truth
- IGME metadata enrichment
- ERT and seismic where available
- Published on Hugging Face Datasets
- Dataset card with:
  - Task description
  - Data collection methodology
  - Field distribution
  - Known limitations
  - Licensing

### AquaScan-100 (v0.6+)
- 100+ validated sites
- Balanced across geological domains
- Multiple geophysical modalities
- Benchmark-ready splits
- Community leaderboard

## Hugging Face Structure

```
aquascan-research/
├── aquascan-25          (Dataset)
│   ├── data/
│   ├── dataset_card.md
│   └── README.md
├── aquascan-100         (Dataset)
├── aquascan-baselines   (Model)
│   ├── logistic/
│   ├── random_forest/
│   └── model_card.md
└── aquascan-benchmark   (Space)
    ├── app.py
    └── leaderboard
```

## Dependencies

When Hugging Face integration is added:
```toml
[project.optional-dependencies]
hf = [
    "datasets>=2.14",
    "huggingface_hub>=0.19",
]
```

## Not Planned for Hugging Face

- Proprietary model weights
- Client-specific datasets
- Production inference endpoints
- AutoCal models

---

> **Note:** This is a forward-looking document. No Hugging Face datasets
> or models currently exist. Implementation timeline depends on data
> collection progress and validation.