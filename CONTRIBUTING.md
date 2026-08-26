# Contributing to AquaScan Open Science

Thank you for your interest in contributing to AquaScan! This document outlines how to contribute effectively.

## Open Science Principles

AquaScan is an **Open Science** project. All contributions to this public repository must be:

- **Reproducible** — experiments must be rerunnable
- **Transparent** — methods and data provenance must be documented
- **Free of proprietary content** — no private data, weights, or algorithms
- **Scientifically rigorous** — claims require evidence

## What Can Be Contributed

- Bug fixes and improvements to existing code
- New public data source extractors
- Improved QA/QC methods (deterministic, public domain)
- Baseline model improvements
- Documentation and tutorials
- Test coverage
- Benchmark methodology improvements

## What Cannot Be Contributed

See [IP Boundary](docs/ip-boundary.md) for details. In short:

- No proprietary model weights
- No private Ground Truth data
- No production GeoAI algorithms
- No client-specific data or configurations

## Development Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
```

## Code Style

- Python 3.11+
- Formatted with `ruff format`
- Linted with `ruff check`
- Type hints where reasonable (checked with `mypy`)
- Tests with `pytest`

## Before Submitting

```bash
pytest
ruff check .
ruff format --check .
```

## Pull Request Process

1. Open an issue describing the change
2. Create a branch (`feature/description` or `fix/description`)
3. Write tests for new functionality
4. Update documentation as needed
5. Submit a PR with a clear description
6. Ensure CI passes

## Scientific Contributions

If your contribution involves scientific claims or methodology:

- Describe the method in the PR description
- Provide references to relevant literature
- Include reproducibility instructions
- Report metrics on public or synthetic data only
- Clearly label synthetic results

## Questions?

Open a [GitHub Discussion](https://github.com/aquascan-research/aquascan-open-science/discussions).