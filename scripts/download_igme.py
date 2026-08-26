#!/usr/bin/env python3
"""Download IGME water points inventory (standalone script).

Usage:
    python scripts/download_igme.py --output data/raw/igme --max-pages 1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src to path for standalone execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from aquascan.ingest.igme import IGMEIngester


def main() -> None:
    parser = argparse.ArgumentParser(description="Download IGME water points inventory")
    parser.add_argument(
        "--output", "-o", default="data/raw/igme", help="Output directory for raw data."
    )
    parser.add_argument(
        "--max-pages", type=int, default=None, help="Maximum pages to download (for testing)."
    )
    parser.add_argument(
        "--max-records", type=int, default=None, help="Maximum records to download."
    )
    args = parser.parse_args()

    ingester = IGMEIngester(output_dir=Path(args.output))
    print(f"Downloading IGME data to {args.output}...")
    result = ingester.download(
        max_pages=args.max_pages,
        max_records=args.max_records,
    )

    print("\nDownload complete:")
    print(f"  Records:   {result['records']}")
    print(f"  Pages:     {result['pages']}")
    print(f"  Expected:  {result['expected_total']}")
    print(f"  Complete:  {result['complete']}")
    print(f"  Size:      {result['size_mb']} MB")
    print(f"  Hash:      {result['checksum']}")


if __name__ == "__main__":
    main()
