"""Candidate Priority Engine — deterministic scoring for IGME enrichment candidates.

Candidate Priority Score answers only:
  "How useful is this IGME record for further Ground Truth enrichment
   and potential AquaScan validation-site selection?"

It MUST NOT represent:
- probability of finding groundwater
- expected well productivity
- probability of drilling success
- Ground Truth quality
- AquaScan prediction

See Gate 2 specification for full requirements.
"""

from __future__ import annotations

import pandas as pd

# ---------------------------------------------------------------------------
# Scoring constants
# ---------------------------------------------------------------------------

# Base component weights (sum = 100)
COMPONENT_WEIGHTS = {
    "coordinates": 15,
    "total_depth": 20,
    "hydrogeological_unit": 15,
    "aquifer_system": 10,
    "detail_url": 10,
    "point_nature": 15,
    "drilling_method": 5,
    "construction_date": 5,
    "reference_flow": 5,
}

# QA penalty points (subtracted from base; max total penalty = 20)
QA_PENALTIES = {
    "coordinate_outlier": -5,
    "elevation_outlier": -3,
    "depth_outlier": -3,
    "negative_reference_flow": -2,
    "duplicate_coordinate": -1,
    "missing_hydrogeological_unit": -2,
    "invalid_construction_date": -2,
    "suspicious_construction_date": -1,
}

# Point nature → score mapping
POINT_NATURE_SCORES = {
    "Sondeo": 15,
    "Pozo con sondeo": 15,
    "Pozo con galería y sondeo": 14,
    "Sondeo de pequeño diámetro, piezómetro": 13,
    "Pozo": 12,
    "Galería": 10,
    "Pozo con galería o taladro horizontal": 8,
    "Manantial": 5,
    "Cauce superficial": 3,
    "Zanjas y Socavones": 5,
    "Calicatas": 5,
}
POINT_NATURE_DEFAULT = 5

# Drilling methods that count as "known"
UNKNOWN_DRILLING_PATTERNS = ["desconocido", "no especificado", "sin especificar"]

# Date quality sub-scores within the construction_date component (max 5)
DATE_CLEAN = 5
DATE_SUSPICIOUS = 3
DATE_INVALID = 0

# Flow availability sub-scores within the reference_flow component (max 5)
FLOW_AVAILABLE = 5  # any value including 0.0
FLOW_NEGATIVE = 3  # present but QA-flagged
FLOW_MISSING = 0

# Scoring version
SCORING_VERSION = "1.0.0"

# Candidate tiers (percentile-based from score distribution)
# Thresholds determined from actual distribution after scoring
TIER_LABELS = ["P1", "P2", "P3", "P4"]


# ---------------------------------------------------------------------------
# Scoring logic
# ---------------------------------------------------------------------------


def _score_coordinates(row: pd.Series) -> tuple[float, str]:
    """Score coordinate availability. All records have coords, but QA flags matter."""
    if row.get("qa_coordinate_outlier", False):
        return 5.0, "coordinates valid but QA coordinate_outlier flag set"
    return 15.0, "coordinates valid"


def _score_total_depth(row: pd.Series) -> tuple[float, str]:
    """Score depth availability. Has depth → points. Depth outlier → reduced."""
    depth = row.get("total_depth_m")
    if depth is None or (isinstance(depth, float) and pd.isna(depth)):
        return 0.0, "total_depth not available"
    if row.get("qa_depth_outlier", False):
        return 10.0, f"total_depth={depth}m available but QA depth_outlier flag set"
    return 20.0, f"total_depth={depth}m available"


def _score_hgu(row: pd.Series) -> tuple[float, str]:
    """Score hydrogeological unit availability."""
    hgu = row.get("hydrogeological_unit")
    if hgu is None or (isinstance(hgu, float) and pd.isna(hgu)):
        return 0.0, "hydrogeological_unit not available (QA missing_hgu flag)"
    if isinstance(hgu, str) and hgu.strip() == "":
        return 0.0, "hydrogeological_unit not available"
    return 15.0, f"hydrogeological_unit='{hgu}' available"


def _score_aquifer_system(row: pd.Series) -> tuple[float, str]:
    """Score aquifer system availability."""
    asi = row.get("aquifer_system")
    if asi is None or (isinstance(asi, float) and pd.isna(asi)):
        return 0.0, "aquifer_system not available"
    if isinstance(asi, str) and asi.strip() == "":
        return 0.0, "aquifer_system not available"
    return 10.0, f"aquifer_system='{asi}' available"


def _score_detail_url(row: pd.Series) -> tuple[float, str]:
    """Score detail URL availability. All records have one."""
    url = row.get("detail_url")
    if url is None or (isinstance(url, float) and pd.isna(url)):
        return 0.0, "detail_url not available"
    return 10.0, "detail_url available"


def _score_point_nature(row: pd.Series) -> tuple[float, str]:
    """Score point nature for AquaScan validation utility."""
    nature = row.get("point_nature")
    if nature is None or (isinstance(nature, float) and pd.isna(nature)):
        return POINT_NATURE_DEFAULT, "point_nature unknown"
    score = POINT_NATURE_SCORES.get(str(nature).strip(), POINT_NATURE_DEFAULT)
    return float(score), f"point_nature='{nature}' → {score} pts"


def _score_drilling_method(row: pd.Series) -> tuple[float, str]:
    """Score drilling method — known method is better for enrichment."""
    method = row.get("drilling_method")
    if method is None or (isinstance(method, float) and pd.isna(method)):
        return 0.0, "drilling_method not available"
    method_lower = str(method).strip().lower()
    for pattern in UNKNOWN_DRILLING_PATTERNS:
        if pattern in method_lower:
            return 0.0, f"drilling_method='{method}' is unknown/unspecified"
    if method_lower == "":
        return 0.0, "drilling_method not available"
    return 5.0, f"drilling_method='{method}' known"


def _score_construction_date(row: pd.Series) -> tuple[float, str]:
    """Score construction date quality. Clean > suspicious > invalid/null.

    CRITICAL: suspicious_date is NOT the same as clean_date.
    """
    date_val = row.get("construction_date")
    has_date = date_val is not None and not (isinstance(date_val, float) and pd.isna(date_val))

    if not has_date:
        if row.get("qa_invalid_construction_date", False):
            return 0.0, "construction_date absent — QA invalid flag"
        return 0.0, "construction_date absent"

    if row.get("qa_invalid_construction_date", False):
        return 0.0, f"construction_date='{date_val}' present but QA invalid flag"
    if row.get("qa_suspicious_construction_date", False):
        return float(DATE_SUSPICIOUS), (
            f"construction_date='{date_val}' available but QA suspicious flag"
        )
    return float(DATE_CLEAN), f"construction_date='{date_val}' clean"


def _score_reference_flow(row: pd.Series) -> tuple[float, str]:
    """Score reference flow availability. Presence counts, NOT magnitude.

    CRITICAL: reference_flow_l_s = 0 gets FULL points — it is NOT a dry well indicator.
    Flow magnitude MUST NOT increase the score.
    """
    flow = row.get("reference_flow_l_s")
    if flow is None or (isinstance(flow, float) and pd.isna(flow)):
        return 0.0, "reference_flow not available"

    if row.get("qa_negative_reference_flow", False):
        return float(FLOW_NEGATIVE), (
            f"reference_flow={flow} L/s available but QA negative flag — magnitude ignored"
        )

    return float(FLOW_AVAILABLE), (f"reference_flow={flow} L/s available (magnitude NOT scored)")


def _compute_qa_penalty(row: pd.Series) -> tuple[float, list[str]]:
    """Compute total QA penalty and list of applied penalties."""
    total = 0.0
    reasons: list[str] = []
    for flag_name, penalty in QA_PENALTIES.items():
        qa_col = f"qa_{flag_name}"
        if qa_col in row.index and row.get(qa_col, False):
            total += penalty
            reasons.append(f"{flag_name}: {penalty:+d}")
    return total, reasons


# Scorer registry — each returns (points, reason_string)
COMPONENT_SCORERS = {
    "coordinates": _score_coordinates,
    "total_depth": _score_total_depth,
    "hydrogeological_unit": _score_hgu,
    "aquifer_system": _score_aquifer_system,
    "detail_url": _score_detail_url,
    "point_nature": _score_point_nature,
    "drilling_method": _score_drilling_method,
    "construction_date": _score_construction_date,
    "reference_flow": _score_reference_flow,
}


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class CandidatePriorityEngine:
    """Deterministic, transparent Candidate Priority scoring engine.

    Every record produces:
    - candidate_priority_score (0-100)
    - candidate_priority_reasons (list of strings, machine-readable)
    - candidate_tier (P1/P2/P3/P4, from score distribution)
    """

    def __init__(self, scoring_version: str = SCORING_VERSION):
        self.scoring_version = scoring_version
        self._tier_thresholds: dict[str, float] | None = None

    def score_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Score all records in a DataFrame.

        Adds columns:
        - candidate_priority_score
        - candidate_priority_reasons
        - candidate_tier (after _assign_tiers)

        Args:
            df: DataFrame from v2 Parquet.

        Returns:
            DataFrame with scoring columns added.
        """
        result = df.copy()

        scores: list[float] = []
        all_reasons: list[list[str]] = []

        for _idx, row in result.iterrows():
            component_scores: dict[str, float] = {}
            component_reasons: dict[str, str] = {}

            # Compute each component
            for name, scorer in COMPONENT_SCORERS.items():
                pts, reason = scorer(row)
                component_scores[name] = pts
                component_reasons[name] = reason

            base_score = sum(component_scores.values())

            # QA penalty
            penalty, penalty_reasons = _compute_qa_penalty(row)

            final_score = max(0.0, base_score + penalty)
            scores.append(final_score)

            # Build reason list
            reasons_list = [
                f"base={base_score:.0f}",
                *(
                    f"+{name}={v:.0f}: {component_reasons[name]}"
                    for name, v in component_scores.items()
                ),
                f"qa_penalty={penalty:+.0f}: "
                + ("none" if not penalty_reasons else "; ".join(penalty_reasons)),
                f"final={final_score:.0f}",
            ]
            all_reasons.append(reasons_list)

        result["candidate_priority_score"] = scores
        result["candidate_priority_reasons"] = all_reasons

        return result

    def assign_tiers(
        self,
        df: pd.DataFrame,
        p1_pct: float = 90,
        p2_pct: float = 75,
        p3_pct: float = 50,
    ) -> pd.DataFrame:
        """Assign candidate tiers (P1-P4) based on score percentiles.

        Default thresholds:
        - P1: score ≥ P90  (top 10%)
        - P2: score ≥ P75  (top 25%)
        - P3: score ≥ P50  (top 50%)
        - P4: score < P50  (bottom 50%)

        Args:
            df: DataFrame with candidate_priority_score column.
            p1_pct: Percentile threshold for P1 (default 90).
            p2_pct: Percentile threshold for P2 (default 75).
            p3_pct: Percentile threshold for P3 (default 50).

        Returns:
            DataFrame with candidate_tier column added.
        """
        scores = df["candidate_priority_score"]

        t1 = scores.quantile(p1_pct / 100.0)
        t2 = scores.quantile(p2_pct / 100.0)
        t3 = scores.quantile(p3_pct / 100.0)

        self._tier_thresholds = {
            "P1": t1,
            "P2": t2,
            "P3": t3,
        }

        def _assign(score: float) -> str:
            if score >= t1:
                return "P1"
            elif score >= t2:
                return "P2"
            elif score >= t3:
                return "P3"
            return "P4"

        result = df.copy()
        result["candidate_tier"] = result["candidate_priority_score"].apply(_assign)
        return result

    @property
    def tier_thresholds(self) -> dict[str, float] | None:
        """Return tier thresholds after assign_tiers has been called."""
        return self._tier_thresholds


def score_dataset(
    df: pd.DataFrame,
    p1_pct: float = 90,
    p2_pct: float = 75,
    p3_pct: float = 50,
) -> pd.DataFrame:
    """Convenience: score and tier a dataset in one call.

    Args:
        df: v2 DataFrame.
        p1_pct, p2_pct, p3_pct: Tier percentile thresholds.

    Returns:
        DataFrame with candidate_priority_score, candidate_priority_reasons,
        and candidate_tier columns.
    """
    engine = CandidatePriorityEngine()
    scored = engine.score_dataframe(df)
    tiered = engine.assign_tiers(scored, p1_pct, p2_pct, p3_pct)
    return tiered
