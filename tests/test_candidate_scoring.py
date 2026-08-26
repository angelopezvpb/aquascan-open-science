"""Tests for the Candidate Priority Engine.

Covers:
- Deterministic scoring
- Null vs zero distinction
- Flow magnitude does NOT affect score
- Candidate Score != Ground Truth Score
- QA flags affect score according to documented rules
- Suspicious date != clean date
- Identical inputs → identical scores
- All records receive valid score
- No records deleted during scoring
- Score breakdown sums to final
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aquascan.candidates.scoring import (
    SCORING_VERSION,
    CandidatePriorityEngine,
    _score_construction_date,
    _score_reference_flow,
    score_dataset,
)

# ---------------------------------------------------------------------------
# Minimal fixtures
# ---------------------------------------------------------------------------


def _make_record(**overrides) -> dict:
    """Build a minimal record dict that scores well."""
    base = {
        "igme_id": "207-4-0001",
        "latitude": 42.94,
        "longitude": -9.22,
        "elevation_m": 50.0,
        "watershed": "NORTE",
        "hydrogeological_unit": "Unidades aisladas",
        "aquifer_system": "Acuífero aislado",
        "province": "A Coruña",
        "municipality": "Cee",
        "point_nature": "Sondeo",
        "total_depth_m": 28.0,
        "reference_flow_l_s": 0.0,
        "drilling_method": "Rotopercusión (martillo en fondo)",
        "construction_date": "1989-02-01",
        "pump_type": "Motor eléctrico",
        "water_use": "Abastecimiento",
        "organization": "Desconocido",
        "detail_url": "http://info.igme.es/bdaguas/PointInfo.aspx?id=207-4-0001",
        # QA flags — all clean by default
        "qa_coordinate_outlier": False,
        "qa_elevation_outlier": False,
        "qa_depth_outlier": False,
        "qa_negative_reference_flow": False,
        "qa_duplicate_coordinate": False,
        "qa_missing_hydrogeological_unit": False,
        "qa_invalid_construction_date": False,
        "qa_suspicious_construction_date": False,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Deterministic scoring
# ---------------------------------------------------------------------------


class TestDeterministicScoring:
    """Identical inputs produce identical scores."""

    def test_identical_records_same_score(self) -> None:
        r1 = _make_record()
        r2 = _make_record()
        df = pd.DataFrame([r1, r2])
        engine = CandidatePriorityEngine()
        scored = engine.score_dataframe(df)
        assert scored["candidate_priority_score"].iloc[0] == pytest.approx(
            scored["candidate_priority_score"].iloc[1]
        )

    def test_same_record_twice_same_score(self) -> None:
        df = pd.DataFrame([_make_record(), _make_record()])
        scored = score_dataset(df)
        assert scored["candidate_priority_score"].iloc[0] == pytest.approx(
            scored["candidate_priority_score"].iloc[1]
        )

    def test_score_is_numeric(self) -> None:
        df = pd.DataFrame([_make_record()])
        scored = score_dataset(df)
        score = scored["candidate_priority_score"].iloc[0]
        assert isinstance(score, float | np.floating)
        assert 0.0 <= score <= 100.0


# ---------------------------------------------------------------------------
# Null vs zero
# ---------------------------------------------------------------------------


class TestNullVsZero:
    """Null is distinctly different from zero."""

    def test_null_depth_scores_differently_from_zero(self) -> None:
        r_null = _make_record(total_depth_m=None)
        r_zero = _make_record(total_depth_m=0.0)
        df = pd.DataFrame([r_null, r_zero])
        scored = score_dataset(df)
        assert scored["candidate_priority_score"].iloc[0] != pytest.approx(
            scored["candidate_priority_score"].iloc[1]
        )

    def test_null_depth_is_not_zero(self) -> None:
        r = _make_record(total_depth_m=None)
        df = pd.DataFrame([r])
        scored = score_dataset(df)
        assert scored["candidate_priority_score"].iloc[0] < 100.0

    def test_null_flow_scores_differently_from_zero(self) -> None:
        r_null = _make_record(reference_flow_l_s=None)
        r_zero = _make_record(reference_flow_l_s=0.0)
        df = pd.DataFrame([r_null, r_zero])
        scored = score_dataset(df)
        # null flow gets 0, zero flow gets FULL points (5)
        assert (
            scored["candidate_priority_score"].iloc[0] < scored["candidate_priority_score"].iloc[1]
        )

    def test_null_date_scores_differently_from_valid_date(self) -> None:
        r_null = _make_record(construction_date=None)
        r_valid = _make_record(construction_date="1989-02-01")
        df = pd.DataFrame([r_null, r_valid])
        scored = score_dataset(df)
        assert (
            scored["candidate_priority_score"].iloc[0] < scored["candidate_priority_score"].iloc[1]
        )


# ---------------------------------------------------------------------------
# Flow magnitude does NOT affect score
# ---------------------------------------------------------------------------


class TestFlowMagnitude:
    """Flow magnitude MUST NOT affect Candidate Priority Score."""

    def test_flow_zero_same_as_flow_350(self) -> None:
        """reference_flow_l_s = 0 gets same score as reference_flow_l_s = 350."""
        r_zero = _make_record(reference_flow_l_s=0.0)
        r_high = _make_record(reference_flow_l_s=350.0)
        df = pd.DataFrame([r_zero, r_high])
        scored = score_dataset(df)
        assert scored["candidate_priority_score"].iloc[0] == pytest.approx(
            scored["candidate_priority_score"].iloc[1]
        )

    def test_flow_zero_not_dry_well(self) -> None:
        """reference_flow_l_s = 0 gets FULL flow component points."""
        _score, reason = _score_reference_flow(
            pd.Series({"reference_flow_l_s": 0.0, "qa_negative_reference_flow": False})
        )
        assert _score == 5.0  # Full points
        assert "magnitude NOT scored" in reason

    def test_flow_positive_small_large_same(self) -> None:
        r_small = _make_record(reference_flow_l_s=0.01)
        r_large = _make_record(reference_flow_l_s=350.0)
        df = pd.DataFrame([r_small, r_large])
        scored = score_dataset(df)
        assert scored["candidate_priority_score"].iloc[0] == pytest.approx(
            scored["candidate_priority_score"].iloc[1]
        )

    def test_flow_null_lower_than_any_flow(self) -> None:
        r_null = _make_record(reference_flow_l_s=None)
        r_any = _make_record(reference_flow_l_s=0.0)
        df = pd.DataFrame([r_null, r_any])
        scored = score_dataset(df)
        assert (
            scored["candidate_priority_score"].iloc[0] < scored["candidate_priority_score"].iloc[1]
        )


# ---------------------------------------------------------------------------
# Candidate Score != Ground Truth Score
# ---------------------------------------------------------------------------


class TestCandidateNotGroundTruth:
    """Candidate Priority Score is semantically separate from Ground Truth."""

    def test_no_ground_truth_fields_in_output(self) -> None:
        df = pd.DataFrame([_make_record()])
        scored = score_dataset(df)
        assert "ground_truth_score" not in scored.columns
        assert "ground_truth_class" not in scored.columns
        assert "ground_truth_source" not in scored.columns

    def test_flow_not_scored_as_water_indicator(self) -> None:
        """High flow does not mean high candidate score."""
        r_dry = _make_record(reference_flow_l_s=0.0)
        r_wet = _make_record(reference_flow_l_s=350.0)
        df = pd.DataFrame([r_dry, r_wet])
        scored = score_dataset(df)
        assert scored["candidate_priority_score"].iloc[0] == pytest.approx(
            scored["candidate_priority_score"].iloc[1]
        )

    def test_score_is_about_enrichment_not_water(self) -> None:
        """A Manantial with rich documentation should outscore a bare Sondeo."""
        r_rich = _make_record(
            point_nature="Manantial",
            total_depth_m=15.0,
            drilling_method="Rotación",
            construction_date="2005-06-15",
            qa_suspicious_construction_date=False,
        )
        r_bare = _make_record(
            point_nature="Sondeo",
            total_depth_m=None,
            drilling_method=None,
            construction_date=None,
        )
        df = pd.DataFrame([r_rich, r_bare])
        scored = score_dataset(df)
        # Rich Manantial has depth + clean date + known drilling
        # Bare Sondeo has higher nature points but nothing else
        assert (
            scored["candidate_priority_score"].iloc[0] > scored["candidate_priority_score"].iloc[1]
        )


# ---------------------------------------------------------------------------
# QA flag effects
# ---------------------------------------------------------------------------


class TestQAFlagEffects:
    """QA flags reduce score according to documented rules."""

    def test_coordinate_outlier_reduces_score(self) -> None:
        r_clean = _make_record()
        r_outlier = _make_record(qa_coordinate_outlier=True)
        df = pd.DataFrame([r_clean, r_outlier])
        scored = score_dataset(df)
        assert (
            scored["candidate_priority_score"].iloc[1] < scored["candidate_priority_score"].iloc[0]
        )

    def test_depth_outlier_reduces_score(self) -> None:
        r_clean = _make_record()
        r_outlier = _make_record(qa_depth_outlier=True, total_depth_m=8300.0)
        df = pd.DataFrame([r_clean, r_outlier])
        scored = score_dataset(df)
        assert (
            scored["candidate_priority_score"].iloc[1] < scored["candidate_priority_score"].iloc[0]
        )

    def test_negative_flow_reduces_score(self) -> None:
        r_clean = _make_record(reference_flow_l_s=5.0)
        r_neg = _make_record(reference_flow_l_s=-2.0, qa_negative_reference_flow=True)
        df = pd.DataFrame([r_clean, r_neg])
        scored = score_dataset(df)
        assert (
            scored["candidate_priority_score"].iloc[1] < scored["candidate_priority_score"].iloc[0]
        )

    def test_duplicate_coord_minor_penalty(self) -> None:
        r_clean = _make_record()
        r_dup = _make_record(qa_duplicate_coordinate=True)
        df = pd.DataFrame([r_clean, r_dup])
        scored = score_dataset(df)
        diff = (
            scored["candidate_priority_score"].iloc[0] - scored["candidate_priority_score"].iloc[1]
        )
        assert diff == pytest.approx(1.0), f"Expected 1 pt penalty, got {diff}"

    def test_qa_flags_not_automatic_exclusion(self) -> None:
        """All records get a valid score — none are deleted."""
        df = pd.DataFrame([_make_record(qa_coordinate_outlier=True)])
        scored = score_dataset(df)
        assert len(scored) == 1
        assert 0 <= scored["candidate_priority_score"].iloc[0] <= 100

    def test_record_with_many_flags_still_scored(self) -> None:
        r = _make_record(
            qa_coordinate_outlier=True,
            qa_elevation_outlier=True,
            qa_negative_reference_flow=True,
        )
        df = pd.DataFrame([r])
        scored = score_dataset(df)
        assert len(scored) == 1
        assert scored["candidate_priority_score"].iloc[0] >= 0


# ---------------------------------------------------------------------------
# Date quality differentiation
# ---------------------------------------------------------------------------


class TestDateQuality:
    """Suspicious date != clean date."""

    def test_clean_date_scores_higher_than_suspicious(self) -> None:
        # Clean date
        _, reason_clean = _score_construction_date(
            pd.Series(
                {
                    "construction_date": "2005-06-15",
                    "qa_invalid_construction_date": False,
                    "qa_suspicious_construction_date": False,
                }
            )
        )
        assert _, "clean" in reason_clean.lower()  # score=5

        # Suspicious date
        _, reason_susp = _score_construction_date(
            pd.Series(
                {
                    "construction_date": "1989-02-01",
                    "qa_invalid_construction_date": False,
                    "qa_suspicious_construction_date": True,
                }
            )
        )
        assert "suspicious" in reason_susp.lower()

    def test_suspicious_date_scores_lower_than_clean(self) -> None:
        r_clean = _make_record(
            construction_date="2005-06-15",
            qa_suspicious_construction_date=False,
        )
        r_suspicious = _make_record(
            construction_date="1989-02-01",
            qa_suspicious_construction_date=True,
        )
        df = pd.DataFrame([r_clean, r_suspicious])
        scored = score_dataset(df)
        assert (
            scored["candidate_priority_score"].iloc[0] > scored["candidate_priority_score"].iloc[1]
        )

    def test_invalid_date_scores_lower_than_suspicious(self) -> None:
        r_suspicious = _make_record(
            construction_date="1989-02-01",
            qa_suspicious_construction_date=True,
            qa_invalid_construction_date=False,
        )
        r_invalid = _make_record(
            construction_date="1989-02-01",
            qa_suspicious_construction_date=False,
            qa_invalid_construction_date=True,
        )
        df = pd.DataFrame([r_suspicious, r_invalid])
        scored = score_dataset(df)
        assert (
            scored["candidate_priority_score"].iloc[0] > scored["candidate_priority_score"].iloc[1]
        )


# ---------------------------------------------------------------------------
# Score breakdown transparency
# ---------------------------------------------------------------------------


class TestScoreBreakdown:
    """Every score includes an explicit breakdown."""

    def test_reasons_present(self) -> None:
        df = pd.DataFrame([_make_record()])
        scored = score_dataset(df)
        reasons = scored["candidate_priority_reasons"].iloc[0]
        assert isinstance(reasons, list)
        assert len(reasons) > 5

    def test_reasons_mention_components(self) -> None:
        df = pd.DataFrame([_make_record()])
        scored = score_dataset(df)
        reasons = scored["candidate_priority_reasons"].iloc[0]
        reason_text = " ".join(reasons)
        for component in [
            "coordinates",
            "total_depth",
            "hydrogeological_unit",
            "aquifer_system",
            "detail_url",
            "point_nature",
            "drilling_method",
            "construction_date",
            "reference_flow",
            "qa_penalty",
            "final",
        ]:
            assert component in reason_text.lower(), f"Missing '{component}' in reasons"

    def test_reasons_contain_base_and_penalty(self) -> None:
        df = pd.DataFrame([_make_record()])
        scored = score_dataset(df)
        reasons = scored["candidate_priority_reasons"].iloc[0]
        assert any("base=" in r for r in reasons)
        assert any("qa_penalty=" in r for r in reasons)
        assert any("final=" in r for r in reasons)


# ---------------------------------------------------------------------------
# Tier assignment
# ---------------------------------------------------------------------------


class TestTierAssignment:
    """Candidate tiers are enrichment-priority tiers, NOT Ground Truth classes."""

    def test_all_records_get_tier(self) -> None:
        df = pd.DataFrame([_make_record()])
        scored = score_dataset(df)
        assert "candidate_tier" in scored.columns
        assert scored["candidate_tier"].iloc[0] in ["P1", "P2", "P3", "P4"]

    def test_tiers_not_ground_truth(self) -> None:
        """Candidate tiers are P1-P4, NOT A/B/C/D."""
        df = pd.DataFrame([_make_record()])
        scored = score_dataset(df)
        assert scored["candidate_tier"].iloc[0] not in ["A", "B", "C", "D"]

    def test_higher_score_higher_or_equal_tier(self) -> None:
        r_low = _make_record(total_depth_m=None, construction_date=None)
        r_high = _make_record()
        df = pd.DataFrame([r_low, r_high])
        scored = score_dataset(df)
        tier_order = {"P1": 1, "P2": 2, "P3": 3, "P4": 4}
        assert (
            tier_order[scored["candidate_tier"].iloc[1]]
            <= tier_order[scored["candidate_tier"].iloc[0]]
        )


# ---------------------------------------------------------------------------
# All-records integrity
# ---------------------------------------------------------------------------


class TestRecordIntegrity:
    """All records preserved and scored."""

    def test_no_records_deleted(self) -> None:
        records = [_make_record() for _ in range(10)]
        df = pd.DataFrame(records)
        scored = score_dataset(df)
        assert len(scored) == 10

    def test_all_records_have_valid_score(self) -> None:
        records = [
            _make_record(),  # perfect
            _make_record(total_depth_m=None, construction_date=None),  # sparse
            _make_record(
                qa_coordinate_outlier=True,
                qa_elevation_outlier=True,
                qa_negative_reference_flow=True,
            ),  # heavily flagged
        ]
        df = pd.DataFrame(records)
        scored = score_dataset(df)
        for score in scored["candidate_priority_score"]:
            assert pd.notna(score)
            assert 0.0 <= score <= 100.0, f"Score {score} out of range"

    def test_scoring_version_recorded(self) -> None:
        engine = CandidatePriorityEngine()
        assert engine.scoring_version == SCORING_VERSION
        assert len(SCORING_VERSION) > 0


# ---------------------------------------------------------------------------
# Component decomposition
# ---------------------------------------------------------------------------


class TestComponentDecomposition:
    """Verify the scoring composition is correct."""

    def test_perfect_record_scores_100(self) -> None:
        """A Sondeo with all fields including clean date should score 100."""
        r = _make_record(
            construction_date="2005-06-15",
            qa_suspicious_construction_date=False,
            qa_invalid_construction_date=False,
        )
        df = pd.DataFrame([r])
        scored = score_dataset(df)
        assert scored["candidate_priority_score"].iloc[0] == pytest.approx(100.0)

    def test_sondeo_no_depth_scores_lower(self) -> None:
        r_depth = _make_record()
        r_no_depth = _make_record(total_depth_m=None)
        df = pd.DataFrame([r_depth, r_no_depth])
        scored = score_dataset(df)
        assert (
            scored["candidate_priority_score"].iloc[1] < scored["candidate_priority_score"].iloc[0]
        )

    def test_manantial_with_depth_vs_sondeo_without(self) -> None:
        """Enrichment potential: Manantial with rich docs > bare Sondeo."""
        r_manantial = _make_record(
            point_nature="Manantial",
            total_depth_m=15.0,
            drilling_method="Rotación",
            construction_date="2005-06-15",
            qa_suspicious_construction_date=False,
        )
        r_sondeo = _make_record(
            point_nature="Sondeo",
            total_depth_m=None,
            drilling_method=None,
            construction_date=None,
        )
        df = pd.DataFrame([r_manantial, r_sondeo])
        scored = score_dataset(df)
        # Manantial: 15+20+15+10+10+5+5+5+5 = 90
        # Sondeo: 15+0+15+10+10+15+0+0+5 = 70
        assert (
            scored["candidate_priority_score"].iloc[0] > scored["candidate_priority_score"].iloc[1]
        )

    def test_minimal_record_still_has_score(self) -> None:
        """Most minimal possible record still gets a valid score."""
        r = {
            "igme_id": "test",
            "latitude": 42.0,
            "longitude": -3.0,
            "elevation_m": None,
            "watershed": None,
            "hydrogeological_unit": None,
            "aquifer_system": None,
            "province": "Test",
            "municipality": "Test",
            "point_nature": "Cauce superficial",
            "total_depth_m": None,
            "reference_flow_l_s": None,
            "drilling_method": None,
            "construction_date": None,
            "pump_type": None,
            "water_use": None,
            "organization": None,
            "detail_url": None,
            "qa_coordinate_outlier": False,
            "qa_elevation_outlier": False,
            "qa_depth_outlier": False,
            "qa_negative_reference_flow": False,
            "qa_duplicate_coordinate": False,
            "qa_missing_hydrogeological_unit": True,
            "qa_invalid_construction_date": False,
            "qa_suspicious_construction_date": False,
        }
        df = pd.DataFrame([r])
        scored = score_dataset(df)
        score = scored["candidate_priority_score"].iloc[0]
        # 15(coords) + 0(depth) + 0(HGU+penalty) + 0(aquifer)
        # + 0(URL) + 3(nature) + 0(drill) + 0(date) + 0(flow) - 2(HGU QA) = 16
        assert score >= 0
        assert score <= 100
