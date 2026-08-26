"""Tests for Ground Truth quality and Candidate Priority scoring."""

from aquascan.data.models import GroundTruthClass
from aquascan.ground_truth.quality import score_ground_truth
from aquascan.ground_truth.scoring import compute_candidate_priority_score


class TestGroundTruthScoring:
    """Test ground truth quality scoring."""

    def test_class_a_strong(self) -> None:
        record = {
            "latitude": 40.0,
            "longitude": -3.0,
            "total_depth": 100.0,
            "lithology_log": "Granite, weathered...",
            "water_entries": "25m, 45m",
            "pump_test_flow": 10.0,
            "pump_test_duration": 24.0,
            "static_water_level": 28.0,
            "water_electrical_conductivity": 800.0,
            "documentation_ref": "BOREHOLE-123",
        }
        result = score_ground_truth(record)
        assert result.score >= 0.75
        assert result.gt_class == GroundTruthClass.A

    def test_class_d_unsuitable(self) -> None:
        record = {}  # No data at all
        result = score_ground_truth(record)
        assert result.score < 0.25
        assert result.gt_class == GroundTruthClass.D

    def test_class_b_or_c_partial(self) -> None:
        record = {
            "latitude": 40.0,
            "longitude": -3.0,
            "total_depth": 100.0,
        }
        result = score_ground_truth(record)
        # Should be C or better
        assert result.score >= 0.25
        assert result.gt_class != GroundTruthClass.D

    def test_missing_coordinates_reduces_score(self) -> None:
        record_full = {
            "latitude": 40.0,
            "longitude": -3.0,
            "total_depth": 100.0,
            "lithology_log": "yes",
            "water_entries": "yes",
            "pump_test_flow": 10.0,
            "pump_test_duration": 24.0,
        }
        result_full = score_ground_truth(record_full)

        record_no_coords = {
            k: v for k, v in record_full.items() if k not in ("latitude", "longitude")
        }
        result_no_coords = score_ground_truth(record_no_coords)

        assert result_no_coords.score < result_full.score

    def test_pump_test_required_for_high_score(self) -> None:
        record = {
            "latitude": 40.0,
            "longitude": -3.0,
            "total_depth": 100.0,
            "lithology_log": "yes",
            "water_entries": "yes",
            "static_water_level": 28.0,
            # pump_test_flow and pump_test_duration missing
        }
        result = score_ground_truth(record)
        assert result.gt_class != GroundTruthClass.A  # Should not reach A without pump test


class TestCandidatePriorityScore:
    """Test Candidate Priority Score computation."""

    def test_empty_record(self) -> None:
        result = compute_candidate_priority_score({})
        assert result.score == 0.0

    def test_full_record_high_score(self) -> None:
        record = {
            "total_depth": 50.0,
            "latitude": 40.0,
            "longitude": -3.0,
            "pump_test_flow": 10.0,
            "hydrogeological_unit": "HGU-A",
            "observations": "Detailed borehole log",
            "timestamp": "2023-06-15T00:00:00Z",
            "point_type": "sondeo",
        }
        result = compute_candidate_priority_score(record)
        assert result.score > 0.5

    def test_candidate_score_is_not_ground_truth(self) -> None:
        """Candidate Priority Score must be distinguishable from Ground Truth."""
        record = {
            "total_depth": 50.0,
            "latitude": 40.0,
            "longitude": -3.0,
        }
        candidate = compute_candidate_priority_score(record)
        gt = score_ground_truth(record)

        # They use different criteria — should produce different scores
        assert candidate.score != gt.score or candidate.components != gt.components

    def test_preferred_point_types(self) -> None:
        record_base = {
            "total_depth": 50.0,
            "latitude": 40.0,
            "longitude": -3.0,
            "pump_test_flow": 10.0,
            "hydrogeological_unit": "HGU-A",
        }
        result_sondeo = compute_candidate_priority_score({**record_base, "point_type": "sondeo"})
        result_unknown = compute_candidate_priority_score({**record_base, "point_type": "unknown"})
        assert result_sondeo.score >= result_unknown.score
