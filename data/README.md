# AquaScan Data Directory

> **Important:** Large geophysical data files are NOT stored in Git.
> This directory contains only READMEs, small test fixtures, and directory
> structure for local development.

## Directory Structure

```
data/
├── raw/         # L0 — immutable original data
├── interim/     # L1/L2 — processed intermediate data
├── processed/   # L3/L4 — ML-ready datasets
└── external/    # Third-party reference data
```

## Not In Git

The following file types are excluded from Git:

- `*.parquet` — Parquet data files (exception: tiny test fixtures)
- `*.seg2`, `*.sg2`, `*.sgy`, `*.segy` — Seismic data
- `*.npy`, `*.npz` — NumPy arrays
- `*.h5`, `*.hdf5` — HDF5 files
- `*.csv` — CSV data (exception: tiny test fixtures)
- `*.fits` — FITS files

## Future Data Distribution

Public AquaScan datasets (AquaScan-25, AquaScan-100) will be published
on Hugging Face Datasets or similar object storage when field validation
and license review are complete. See [docs/roadmap.md](../docs/roadmap.md).

## Local Setup

```bash
# Create directory structure
mkdir -p data/{raw,interim,processed,external}
```