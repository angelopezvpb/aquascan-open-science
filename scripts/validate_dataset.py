#!/usr/bin/env python3
"""Validate a dataset against the AquaScan schema.

Usage:
    python scripts/validate_dataset.py --input data/interim/igme.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from aquascan.data.models import CampaignRecord
from aquascan.data.validation import validate_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate dataset against AquaScan schema")
    parser.add_argument("--input", "-i", required=True, help="Input file (Parquet or CSV).")
    args = parser.parse_args()

    path = Path(args.input)
    if path.suffix == ".parquet":
        df = pd.read_parquet(args.input)
    else:
        df = pd.read_csv(args.input)

    records = []
    for _, row in df.iterrows():
        try:
            records.append(CampaignRecord(**row.to_dict()))
        except Exception as e:
            print(f"ERROR parsing record: {e}", file=sys.stderr)

    report = validate_dataset(records)
    print(report)

    if not report.is_valid:
        sys.exit(1)


if __name__ == "__main__":
    main()
