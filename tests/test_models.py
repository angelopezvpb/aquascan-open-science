"""Tests for AquaScan data models."""

import json

import pytest

from aquascan.data.models import (
    CampaignRecord,
    EvidenceScore,
    ExperimentRecord,
    GroundTruthClass,
    SyntheticLayer,
    SyntheticProfile,
    WaterPrediction,
)


class TestCampaignRecord:
    """Test the canonical CampaignRecord model."""

    def test_minimal_record(self) -> None:
        """A record with only required fields should be valid."""
        record = CampaignRecord(
            campaign_id="TEST-001",
            site_id="SITE-001",
        )
        assert record.campaign_id == "TEST-001"
        assert record.site_id == "SITE-001"
        assert record.schema_version == "1.0.0"

    def test_full_record(self) -> None:
        """A fully populated record should preserve all fields."""
        record = CampaignRecord(
            campaign_id="CAMPAIGN-001",
            site_id="SITE-001",
            well_id="WELL-001",
            latitude=40.4168,
            longitude=-3.7038,
            elevation=667.0,
            geology_domain="Granite",
            hydrogeological_unit="HGU-Alpha",
            ert_raw_uri="file://data/raw/ert_001.dat",
            ert_quality=0.85,
            seismic_quality=0.72,
            ert_resistivity_mean=45.3,
            seismic_vp_mean=1800.0,
            water_depth_min=25.0,
            water_depth_max=35.0,
            total_depth=100.0,
            static_water_level=28.5,
            pump_test_flow=5.2,
            pump_test_duration=24.0,
            water_electrical_conductivity=850.0,
            water_tds=550.0,
            ground_truth_score=0.85,
            ground_truth_class=GroundTruthClass.A,
            ground_truth_source="borehole_log",
            ground_truth_reasons=["Complete borehole log with pump test"],
        )

        assert record.latitude == 40.4168
        assert record.ert_resistivity_mean == 45.3
        assert record.ground_truth_class == GroundTruthClass.A

    def test_null_not_zero(self) -> None:
        """Unknown values must be null, not zero."""
        record = CampaignRecord(
            campaign_id="NULL-TEST",
            site_id="SITE-NULL",
            # pump_test_flow intentionally omitted
        )
        assert record.pump_test_flow is None
        # Verify it's NOT zero
        assert record.pump_test_flow != 0

    def test_empty_string_to_none(self) -> None:
        """Empty geology_domain string should become None."""
        record = CampaignRecord(
            campaign_id="EMPTY-TEST",
            site_id="SITE-EMPTY",
            geology_domain="",
        )
        assert record.geology_domain is None

    def test_json_schema_export(self) -> None:
        """Exporting to JSON Schema should produce valid JSON."""
        schema = CampaignRecord.model_json_schema()
        assert schema["type"] == "object"
        assert "campaign_id" in schema["properties"]
        # Validate it's valid JSON
        json.dumps(schema)

    def test_coordinate_ranges(self) -> None:
        """Latitude must be in [-90, 90], longitude in [-180, 180]."""
        # Valid
        r = CampaignRecord(campaign_id="C1", site_id="S1", latitude=0, longitude=0)
        assert r.latitude == 0

        # Invalid latitude
        with pytest.raises(Exception):
            CampaignRecord(campaign_id="C2", site_id="S2", latitude=91)

        # Invalid longitude
        with pytest.raises(Exception):
            CampaignRecord(campaign_id="C3", site_id="S3", longitude=181)

    def test_quality_score_ranges(self) -> None:
        """Quality scores must be in [0, 1]."""
        # Valid
        r = CampaignRecord(
            campaign_id="C1",
            site_id="S1",
            ert_quality=0.5,
            seismic_quality=0.5,
        )
        assert r.ert_quality == 0.5

        # Invalid
        with pytest.raises(Exception):
            CampaignRecord(campaign_id="C2", site_id="S2", ert_quality=1.5)

        with pytest.raises(Exception):
            CampaignRecord(campaign_id="C3", site_id="S3", ert_quality=-0.1)


class TestGroundTruthClass:
    """Test GroundTruthClass enum."""

    def test_all_classes(self) -> None:
        assert GroundTruthClass.A.value == "A"
        assert GroundTruthClass.B.value == "B"
        assert GroundTruthClass.C.value == "C"
        assert GroundTruthClass.D.value == "D"


class TestWaterPrediction:
    """Test WaterPrediction model."""

    def test_prediction_with_abstention(self) -> None:
        pred = WaterPrediction(
            campaign_id="TEST-001",
            water_score=0.90,
            evidence_score=0.50,
            abstain=True,
            confidence=0.3,
        )
        assert pred.water_score == 0.90
        assert pred.evidence_score == 0.50
        assert pred.abstain is True

    def test_prediction_no_abstention(self) -> None:
        pred = WaterPrediction(
            campaign_id="TEST-002",
            water_score=0.85,
            evidence_score=0.80,
            abstain=False,
            confidence=0.85,
        )
        assert pred.abstain is False


class TestEvidenceScore:
    """Test EvidenceScore model."""

    def test_evidence_score_components(self) -> None:
        ev = EvidenceScore(
            score=0.65,
            components={"ert_quality": 0.8, "seismic_quality": 0.5},
            reasons=["Good ERT quality", "Moderate seismic quality"],
        )
        assert ev.score == 0.65
        assert len(ev.components) == 2
        assert len(ev.reasons) == 2


class TestExperimentRecord:
    """Test ExperimentRecord model."""

    def test_experiment_record(self) -> None:
        exp = ExperimentRecord(
            dataset_version="v1",
            model="logistic_regression",
            hyperparameters={"C": 1.0},
            metrics={"f1": 0.75},
        )
        assert exp.model == "logistic_regression"
        assert exp.metrics["f1"] == 0.75


class TestSyntheticProfile:
    """Test synthetic data models."""

    def test_synthetic_profile(self) -> None:
        layer = SyntheticLayer(
            depth_top=0, depth_bottom=10, resistivity=100, vp=500, saturated=False
        )
        profile = SyntheticProfile(
            layers=[layer],
            water_table_depth=30.0,
        )
        assert profile.label == "SYNTHETIC"
        assert "SYNTHETIC DATA" in profile.warnings[0]
