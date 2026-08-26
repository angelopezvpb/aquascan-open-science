#!/usr/bin/env python3
"""Build an ML-ready dataset from normalized records.

Usage:
    python scripts/build_dataset.py \
        --input data/interim/igme.parquet \
        --output data/processed/dataset.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from aquascan.data.models import CampaignRecord
from aquascan.features.build import build_features


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ML-ready dataset from AquaScan records")
    parser.add_argument(
        "--input", "-i", required=True, help="Input Parquet file with normalized records."
    )
    parser.add_argument(
        "--output", "-o", default="data/processed/dataset.parquet", help="Output path."
    )
    args = parser.parse_args()

    df_raw = pd.read_parquet(args.input)
    print(f"Loaded {len(df_raw)} records")

    # Parse into CampaignRecords
    records = []
    for _, row in df_raw.iterrows():
        try:
            records.append(CampaignRecord(**row.to_dict()))
        except Exception as e:
            print(f"Warning: skipped record due to: {e}")

    print(f"Parsed {len(records)} valid records")

    # Build features
    features_df = build_features(records, include_ground_truth=True)
    print(f"Built {len(features_df.columns)} features")

    features_df.to_parquet(args.output)
    print(f"Exported to {args.output}")


if __name__ == "__main__":
    main()
