#!/usr/bin/env python3
"""Run a baseline model on an AquaScan dataset.

Usage:
    python scripts/run_baseline.py \
        --input data/processed/dataset.parquet \
        --baseline random_forest \
        --output results/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from aquascan.baselines import LogisticBaseline, RandomForestBaseline, RuleBaseline
from aquascan.evaluation.geographic_split import (
    GeographicHoldoutSplit,
    check_geographic_leakage,
)
from aquascan.evaluation.metrics import compute_classification_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline model on AquaScan dataset")
    parser.add_argument(
        "--input", "-i", required=True, help="Input Parquet file with ML-ready features."
    )
    parser.add_argument(
        "--baseline",
        "-b",
        default="logistic",
        choices=["rule", "logistic", "random_forest"],
        help="Baseline model to run.",
    )
    parser.add_argument("--output", "-o", default="results/", help="Output directory for results.")
    parser.add_argument("--target", "-t", default="ground_truth_score", help="Target column name.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold to binarize target for classification.",
    )
    args = parser.parse_args()

    # Load data
    df = pd.read_parquet(args.input)
    print(f"Loaded {len(df)} records")

    # Binarize target
    if args.target in df.columns:
        y = (df[args.target] >= args.threshold).astype(int)
    else:
        print(f"Target column '{args.target}' not found.", file=sys.stderr)
        sys.exit(1)

    # Drop non-feature columns
    drop = [
        "campaign_id",
        "site_id",
        "latitude",
        "longitude",
        "ground_truth_score",
        "ground_truth_class",
    ]
    X = df.drop(columns=[c for c in drop if c in df.columns], errors="ignore")

    # Fill NaN
    X = X.fillna(X.median(numeric_only=True))
    for col in X.select_dtypes(include="object").columns:
        X[col] = X[col].fillna("unknown")

    # Geographic split
    splitter = GeographicHoldoutSplit(test_fraction=0.3, random_state=42)
    train_idx, test_idx = splitter.split(df)
    print(f"Train: {len(train_idx)}, Test: {len(test_idx)}")

    # Check leakage
    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]
    leak = check_geographic_leakage(train_df, test_df)
    if leak["has_leakage"]:
        print(f"WARNING: Geographic leakage detected: {leak['issues']}")

    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

    # Select and fit baseline
    if args.baseline == "rule":
        model = RuleBaseline(strategy="mean")
    elif args.baseline == "random_forest":
        model = RandomForestBaseline(random_state=42)
    else:
        model = LogisticBaseline(random_state=42)

    print(f"Fitting {args.baseline} baseline...")
    model.fit(X_train, y_train.values)

    # Predict
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
        y_pred = (y_score >= 0.5).astype(int)
    else:
        y_pred = model.predict(X_test)
        y_score = model.predict(X_test).astype(float)

    # Metrics
    metrics = compute_classification_metrics(y_test.values, y_pred, y_score)

    print(f"\nResults ({args.baseline}):")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # Save results
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    results = {
        "baseline": args.baseline,
        "metrics": {k: v for k, v in metrics.items() if isinstance(v, int | float)},
    }
    with open(out / f"{args.baseline}_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {out / f'{args.baseline}_results.json'}")


if __name__ == "__main__":
    main()
