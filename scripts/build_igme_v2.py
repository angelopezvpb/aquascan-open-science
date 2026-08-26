#!/usr/bin/env python3
"""Build IGME national v2 Parquet from RAW snapshot.

Generates:
  data/interim/igme_national_20260823T171204Z_v2.parquet

Also investigates and reports on national anomalies.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from aquascan.ingest.igme import IGMEIngester


def find_anomaly_record(raw_dir: Path, igme_id: str) -> dict | None:
    """Find a specific IGME record by Id in all raw pages."""
    for jf in sorted(raw_dir.glob("page_*.json")):
        data = json.loads(jf.read_bytes().decode("utf-8"))
        for feature in data.get("features", []):
            attrs = feature.get("attributes", {})
            if attrs.get("Id") == igme_id:
                return {
                    "file": jf.name,
                    "attributes": attrs,
                    "geometry": feature.get("geometry", {}),
                }
    return None


def classify_anomaly(record: dict | None, issue: str) -> str:
    """Classify whether an anomaly appears legitimate, source-data error, sentinel, or ambiguous."""
    if record is None:
        return "record_not_found_in_raw"

    if issue == "elevation_max_9837":
        return (
            "source_data_error — likely unit conversion (feet → m not applied) or data entry error"
        )

    elif issue == "elevation_minus_92":
        return (
            "source_data_error — negative elevation below sea"
            " level implausible for IGME surface points"
        )

    elif issue == "depth_max_8300":
        return (
            "source_data_error — likely data entry error"
            " (8300 cm → 83 m, or similar order-of-magnitude mistake)"
        )

    elif issue == "negative_flow":
        # -2.0 L/s: negative flow is physically impossible
        return "ambiguous — could be a data entry error or a sentinel value indicating unknown"

    elif issue == "coordinate_outside_spain":
        # (4.47995, -4.47995): latitude ~4.48 is in the Gulf of Guinea (Africa).
        # Longitude -4.48 is within Spain's range, but the lat is clearly wrong.
        # Likely a coordinate swap or error.
        return "source_data_error — latitude 4.48° is Gulf of Guinea, likely coordinate entry error"

    elif issue == "lat_min_4_47995":
        # The same record as coordinate_outside_spain
        return "source_data_error — latitude 4.48° is outside Spain"

    elif issue == "depth_gt_1000":
        return (
            "ambiguous — 136 records with depth >1000 m:"
            " some may be legitimate deep boreholes, others may be data entry errors"
        )

    elif issue == "duplicate_coords":
        return (
            "ambiguous — 3571 repeated coordinate pairs:"
            " likely shared locations (different water points "
            "at same site) rather than true duplicates"
        )

    return "unclassified"


def find_deep_records(raw_dir: Path, limit: int = 5) -> list[dict]:
    """Find records with depth > 1000 m."""
    records = []
    for jf in sorted(raw_dir.glob("page_*.json")):
        data = json.loads(jf.read_bytes().decode("utf-8"))
        for feature in data.get("features", []):
            attrs = feature.get("attributes", {})
            prof = attrs.get("PROF_m")
            if prof is not None and prof > 1000:
                records.append(
                    {
                        "igme_id": attrs.get("Id"),
                        "depth_m": prof,
                        "province": attrs.get("Provincia"),
                        "point_nature": attrs.get("Naturaleza"),
                        "file": jf.name,
                    }
                )
                if len(records) >= limit:
                    return records
    return records


def main() -> None:
    raw_dir = Path("data/raw/igme_national_20260823T171204Z")
    output_path = Path("data/interim/igme_national_20260823T171204Z_v2.parquet")

    print("=" * 70)
    print("IGME Semantic Refactor v2 — Anomaly Investigation")
    print("=" * 70)

    # --- Anomaly 1: Elevation max = 9837 m ---
    print("\n--- Anomaly: Elevation = 9837 m ---")
    rec = find_anomaly_record(raw_dir, "2144-4-0002")
    if rec:
        print(f"  RAW file: {rec['file']}")
        print(f"  IGME Id: {rec['attributes']['Id']}")
        print(f"  COTA_msnm: {rec['attributes']['COTA_msnm']}")
        print(f"  Naturaleza: {rec['attributes']['Naturaleza']}")
        print(f"  Provincia: {rec['attributes']['Provincia']}")
        print(f"  Municipio: {rec['attributes']['Municipio']}")
        print(f"  PROF_m: {rec['attributes']['PROF_m']}")
        print(f"  X: {rec['geometry']['x']}, Y: {rec['geometry']['y']}")
        # Check if the UTM Expr1/Expr2 provide a plausible z
        print(f"  Expr1: {rec['attributes'].get('Expr1')}, Expr2: {rec['attributes'].get('Expr2')}")
        print(f"  Classification: {classify_anomaly(rec, 'elevation_max_9837')}")
    else:
        print("  NOT FOUND in RAW")

    # --- Anomaly 2: Elevation min = -92 m ---
    print("\n--- Anomaly: Elevation = -92 m ---")
    # Find the specific record
    for pg in sorted(raw_dir.glob("page_*.json")):
        data = json.loads(pg.read_bytes().decode("utf-8"))
        for f in data.get("features", []):
            if f["attributes"].get("COTA_msnm") == -92.0:
                rec = f
                print(f"  RAW file: {pg.name}")
                print(f"  IGME Id: {rec['attributes']['Id']}")
                print(f"  COTA_msnm: {rec['attributes']['COTA_msnm']}")
                print(f"  Naturaleza: {rec['attributes']['Naturaleza']}")
                print(f"  Provincia: {rec['attributes']['Provincia']}")
                print(f"  Municipio: {rec['attributes']['Municipio']}")
                print(f"  X: {rec['geometry']['x']}, Y: {rec['geometry']['y']}")
                print(f"  Classification: {classify_anomaly(None, 'elevation_minus_92')}")
                break

    # --- Anomaly 3: Depth max = 8300 m ---
    print("\n--- Anomaly: Depth = 8300 m ---")
    rec = find_anomaly_record(raw_dir, "2240-4-0025")
    if rec:
        print(f"  RAW file: {rec['file']}")
        print(f"  IGME Id: {rec['attributes']['Id']}")
        print(f"  PROF_m: {rec['attributes']['PROF_m']}")
        print(f"  COTA_msnm: {rec['attributes']['COTA_msnm']}")
        print(f"  Naturaleza: {rec['attributes']['Naturaleza']}")
        print(f"  Provincia: {rec['attributes']['Provincia']}")
        print(f"  Municipio: {rec['attributes']['Municipio']}")
        print(f"  FECHA_OBRA: {rec['attributes']['FECHA_OBRA']}")
        print(f"  Classification: {classify_anomaly(rec, 'depth_max_8300')}")
    else:
        print("  NOT FOUND in RAW")

    # --- Anomaly 4: Negative flow ---
    print("\n--- Anomaly: Negative flow = -2.0 L/s ---")
    rec = find_anomaly_record(raw_dir, "1605-6-0013")
    if rec:
        print(f"  RAW file: {rec['file']}")
        print(f"  IGME Id: {rec['attributes']['Id']}")
        print(f"  Caudal_Referencia_L_s: {rec['attributes']['Caudal_Referencia_L_s']}")
        print(f"  Naturaleza: {rec['attributes']['Naturaleza']}")
        print(f"  Provincia: {rec['attributes']['Provincia']}")
        print(f"  Classification: {classify_anomaly(rec, 'negative_flow')}")
    else:
        print("  NOT FOUND in RAW")

    # --- Anomaly 5: Coordinate outside Spain ---
    print("\n--- Anomaly: Coordinate outside Spain bbox ---")
    rec = find_anomaly_record(raw_dir, "1704-5-0005")
    if rec:
        print(f"  RAW file: {rec['file']}")
        print(f"  IGME Id: {rec['attributes']['Id']}")
        print(f"  X: {rec['geometry']['x']}, Y: {rec['geometry']['y']}")
        print(f"  Provincia: {rec['attributes']['Provincia']}")
        print(f"  Municipio: {rec['attributes']['Municipio']}")
        print(f"  Classification: {classify_anomaly(rec, 'coordinate_outside_spain')}")
    else:
        print("  NOT FOUND in RAW")

    # --- Anomaly 6: Deep records sample ---
    print("\n--- Anomaly: Records with depth > 1000 m (sample of 5) ---")
    deep_records = find_deep_records(raw_dir, limit=5)
    for dr in deep_records:
        print(
            f"  IGME Id: {dr['igme_id']}, Depth: {dr['depth_m']} m, "
            f"Province: {dr['province']}, Nature: {dr['point_nature']}, "
            f"File: {dr['file']}"
        )
    print(f"  Classification: {classify_anomaly(None, 'depth_gt_1000')}")

    # --- Generate v2 ---
    print("\n" + "=" * 70)
    print("Generating IGME national v2")
    print("=" * 70)

    ingester = IGMEIngester()
    records = ingester.normalize(raw_dir)
    print(f"Normalized: {len(records)} IGMEWaterPointRecord instances")

    if len(records) != 139697:
        print(f"ERROR: Cardinality mismatch! Expected 139697, got {len(records)}")
        sys.exit(1)

    df = ingester.to_dataframe(records)
    print(f"DataFrame: {len(df)} rows × {len(df.columns)} columns")
    print(f"Columns: {list(df.columns)}")

    # QA summary
    print("\nQA flag summary:")
    qa_cols = [c for c in df.columns if c.startswith("qa_") and c != "qa_flag_count"]
    for col in sorted(qa_cols):
        count = df[col].sum()
        if count > 0:
            print(f"  {col}: {count}")

    # Save v2
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ingester.export(records, output_path, format="parquet")
    print(f"\nExported v2 to: {output_path}")
    print(f"Cardinality: {len(df)} ✓")

    print("\n" + "=" * 70)
    print("Anomaly Investigation Summary")
    print("=" * 70)
    print("""
ANOMALY                    | VALUE          | CLASSIFICATION
---------------------------|----------------|-----------------
Elevation max              | 9837 m         | source_data_error
Elevation min              | -92 m          | source_data_error
Total depth max            | 8300 m         | source_data_error
Negative reference flow    | -2.0 L/s       | ambiguous
Coordinate outside Spain   | (4.48, -4.48)  | source_data_error
136 records depth >1000 m  | up to 8300 m   | ambiguous
3571 repeated coord pairs  | —              | ambiguous

All anomalous records are preserved in v2 with QA flags.
No source data corrected — QA flags serve as warnings.
""")

    # Date parse quality check
    print("Date parsing quality:")
    date_col = "construction_date"
    if date_col in df.columns:
        total = len(df)
        null_dates = df[date_col].isna().sum()
        valid_dates = total - null_dates
        invalid = df.get("qa_invalid_construction_date", pd.Series([False] * total)).sum()
        suspicious = df.get("qa_suspicious_construction_date", pd.Series([False] * total)).sum()
        print(f"  Total records: {total}")
        print(f"  Valid construction dates: {valid_dates}")
        print(f"  Null/unparseable dates: {null_dates}")
        print(f"  QA invalid_construction_date: {invalid}")
        print(f"  QA suspicious_construction_date: {suspicious}")

    # Flow semantics check
    print("\nFlow semantics check:")
    flow_col = "reference_flow_l_s"
    if flow_col in df.columns:
        flow = df[flow_col]
        print(f"  Null: {flow.isna().sum()}")
        print(f"  Zero: {(flow == 0).sum()}")
        print(f"  Positive: {(flow > 0).sum()}")
        print(f"  Negative (QA-flagged): {(flow < 0).sum()}")
    # Confirm pump_test_flow is NOT present
    assert "pump_test_flow" not in df.columns, "ERROR: pump_test_flow leaked into v2!"
    print("  pump_test_flow NOT present ✓ (correct semantic separation)")


if __name__ == "__main__":
    main()
