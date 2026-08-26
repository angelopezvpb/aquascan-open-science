"""AquaScan command-line interface."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click

from aquascan import __version__


@click.group()
@click.version_option(__version__)
def main() -> None:
    """AquaScan Open Science — hydrogeophysical groundwater prospecting research."""
    pass


# ---------------------------------------------------------------------------
# IGME commands
# ---------------------------------------------------------------------------


@main.group()
def igme() -> None:
    """IGME — Inventario de Puntos de Agua operations."""
    pass


@igme.command("download")
@click.option(
    "--output",
    "-o",
    default="data/raw/igme",
    help="Output directory for raw IGME data.",
)
@click.option(
    "--max-pages",
    type=int,
    default=None,
    help="Maximum number of pages to download (for testing).",
)
@click.option(
    "--page-size",
    type=int,
    default=1000,
    help="Records per page.",
)
def igme_download(output: str, max_pages: int | None, page_size: int) -> None:
    """Download IGME water point inventory (ArcGIS REST).

    Downloads raw JSON responses, computes SHA-256 checksums,
    and stores them with provenance metadata.
    """
    from aquascan.ingest.igme import IGMEIngester

    ingester = IGMEIngester(
        output_dir=Path(output),
        page_size=page_size,
    )
    click.echo(f"Downloading IGME data to {output}...")
    result = ingester.download(max_pages=max_pages)

    click.echo(f"Downloaded {result['records']} records in {result['pages']} pages.")
    click.echo(f"Expected total: {result['expected_total']}")
    click.echo(f"Complete: {result['complete']}")
    click.echo(f"Total size: {result['size_mb']:.1f} MB")
    click.echo(f"Checksum: {result['checksum']}")


@igme.command("normalize")
@click.option(
    "--input",
    "-i",
    default="data/raw/igme",
    help="Directory with raw IGME JSON responses.",
)
@click.option(
    "--output",
    "-o",
    default="data/interim/igme.parquet",
    help="Output path for normalized Parquet file.",
)
def igme_normalize(input: str, output: str) -> None:
    """Normalize raw IGME data to AquaScan canonical schema.

    Reads raw IGME JSON pages, normalizes every ArcGIS feature into a
    CampaignRecord, and exports to Parquet. One feature → one record.
    """
    import pandas as pd

    from aquascan.ingest.igme import IGMEIngester

    ingester = IGMEIngester(output_dir=Path("data/raw/igme"))
    click.echo(f"Normalizing data from {input}...")
    records = ingester.normalize(Path(input))
    click.echo(f"Normalized {len(records)} records.")

    # Convert CampaignRecord instances to dicts for DataFrame
    df = pd.DataFrame([r.model_dump() for r in records])
    df.to_parquet(output)
    click.echo(f"Exported {len(df)} rows to {output}")


@igme.command("candidates")
@click.option(
    "--input",
    "-i",
    default="data/interim/igme.parquet",
    help="Input Parquet file with normalized records.",
)
@click.option(
    "--output",
    "-o",
    default="data/processed/candidates.parquet",
    help="Output path for candidate table.",
)
@click.option(
    "--min-score",
    type=float,
    default=0.3,
    help="Minimum Candidate Priority Score threshold.",
)
def igme_candidates(input: str, output: str, min_score: float) -> None:
    """Build candidate table with priority scores.

    Computes Candidate Priority Score for each record and filters
    to those above the threshold.
    """
    import pandas as pd

    from aquascan.ground_truth.scoring import compute_candidate_priority_score

    df = pd.read_parquet(input)
    click.echo(f"Loaded {len(df)} records from {input}")

    scores = []
    for _, row in df.iterrows():
        result = compute_candidate_priority_score(row.to_dict())
        scores.append(result)

    df["candidate_priority_score"] = [s.score for s in scores]
    df["candidate_priority_reasons"] = [s.reasons for s in scores]

    # Filter by threshold
    candidates = df[df["candidate_priority_score"] >= min_score]
    click.echo(
        f"Selected {len(candidates)} candidates (score >= {min_score}) from {len(df)} records."
    )

    candidates.to_parquet(output)
    click.echo(f"Exported to {output}")


# ---------------------------------------------------------------------------
# Dataset commands
# ---------------------------------------------------------------------------


@main.group()
def dataset() -> None:
    """Dataset operations."""
    pass


@dataset.command("validate")
@click.option(
    "--input",
    "-i",
    required=True,
    help="Input file (Parquet or CSV) to validate.",
)
def dataset_validate(input: str) -> None:
    """Validate a dataset against the AquaScan schema."""
    import pandas as pd

    from aquascan.data.models import CampaignRecord
    from aquascan.data.validation import validate_dataset

    path = Path(input)
    if path.suffix == ".parquet":
        df = pd.read_parquet(input)
    else:
        df = pd.read_csv(input)

    records = []
    for _, row in df.iterrows():
        try:
            records.append(CampaignRecord(**row.to_dict()))
        except Exception as e:
            click.echo(f"Error parsing record: {e}", err=True)

    report = validate_dataset(records)
    click.echo(str(report))

    if not report.is_valid:
        sys.exit(1)


# ---------------------------------------------------------------------------
# Experiment commands
# ---------------------------------------------------------------------------


@main.command("fingerprint")
@click.option(
    "--config",
    "-c",
    required=True,
    help="Experiment configuration file (YAML/JSON).",
)
@click.option(
    "--dataset-version",
    default="unknown",
    help="Dataset version identifier.",
)
def experiment_fingerprint(config: str, dataset_version: str) -> None:
    """Generate a reproducible experiment fingerprint.

    Computes a hash from configuration + dataset version + (optional) git commit.
    """
    import hashlib

    import yaml

    with open(config) as f:
        if config.endswith(".yaml") or config.endswith(".yml"):
            cfg = yaml.safe_load(f)
        else:
            cfg = json.load(f)

    # Canonical JSON for hashing
    canonical = json.dumps(cfg, sort_keys=True) + dataset_version

    # Try to get git commit
    import subprocess

    try:
        git_commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
        canonical += git_commit
    except (subprocess.CalledProcessError, FileNotFoundError):
        git_commit = "unknown"

    fingerprint = hashlib.sha256(canonical.encode()).hexdigest()[:12]
    click.echo(f"Experiment fingerprint: {fingerprint}")
    click.echo(f"  dataset_version: {dataset_version}")
    click.echo(f"  git_commit: {git_commit}")


# ---------------------------------------------------------------------------
# Schema export
# ---------------------------------------------------------------------------


@main.command("export-schemas")
@click.option(
    "--output",
    "-o",
    default="schemas/",
    help="Output directory for JSON Schema files.",
)
def export_schemas(output: str) -> None:
    """Export JSON Schema files from Pydantic models."""
    from aquascan.data.models import CampaignRecord

    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)

    schema = CampaignRecord.model_json_schema()
    with open(out / "aquascan_dataset.schema.json", "w") as f:
        json.dump(schema, f, indent=2)
    click.echo(f"Exported schema to {out / 'aquascan_dataset.schema.json'}")


if __name__ == "__main__":
    main()
