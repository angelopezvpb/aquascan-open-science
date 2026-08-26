#!/usr/bin/env python3
"""Build candidate table from normalized IGME records.

Usage:
    python scripts/build_candidate_table.py \
        --input data/interim/igme.parquet \
        --output data/processed/candidates.parquet \
        --min-score 0.3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from aquascan.ground_truth.scoring import compute_candidate_priority_score


def main() -> None:
    parser = argparse.ArgumentParser(description="Build candidate table with priority scores")
    parser.add_argument(
        "--input", "-i", required=True, help="Input Parquet file with normalized records."
    )
    parser.add_argument(
        "--output", "-o", default="data/processed/candidates.parquet", help="Output path."
    )
    parser.add_argument(
        "--min-score", type=float, default=0.3, help="Minimum Candidate Priority Score."
    )
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    print(f"Loaded {len(df)} records from {args.input}")

    scores = []
    for _, row in df.iterrows():
        result = compute_candidate_priority_score(row.to_dict())
        scores.append(result)

    df["candidate_priority_score"] = [s.score for s in scores]
    df["candidate_priority_reasons"] = [s.reasons for s in scores]

    candidates = df[df["candidate_priority_score"] >= args.min_score]
    print(f"Selected {len(candidates)} candidates (score >= {args.min_score})")

    candidates.to_parquet(args.output)
    print(f"Exported to {args.output}")


if __name__ == "__main__":
    main()
