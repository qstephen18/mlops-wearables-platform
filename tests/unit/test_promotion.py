"""Unit tests for the promotion gate.

No Spark, no workspace, no registry. These run in under a second, which is
why they can run on every pull request.
"""

import pytest

from src.promotion import evaluate_gate, gate_tags

FLOOR = 0.75
TOLERANCE = 0.01


def test_promotes_clean_candidate_with_no_champion():
    result = evaluate_gate(0.83, has_signature=True, floor=FLOOR)
    assert result.passed
    assert result.failed_checks == []
    assert "no existing champion" in result.comparison


def test_rejects_below_floor():
    result = evaluate_gate(0.71, has_signature=True, floor=FLOOR)
    assert not result.passed
    assert result.failed_checks == ["above_floor"]


def test_rejects_missing_signature():
    """A model without a signature cannot have its inputs validated at load."""
    result = evaluate_gate(0.90, has_signature=False, floor=FLOOR)
    assert not result.passed
    assert "signature_present" in result.failed_checks


def test_rejects_regression_beyond_tolerance():
    result = evaluate_gate(
        0.80, has_signature=True, floor=FLOOR, champion_score=0.85,
        regression_tolerance=TOLERANCE,
    )
    assert not result.passed
    assert result.failed_checks == ["no_regression"]


def test_allows_small_regression_inside_tolerance():
    """A trivially worse model is not a failure; noise is not regression."""
    result = evaluate_gate(
        0.845, has_signature=True, floor=FLOOR, champion_score=0.85,
        regression_tolerance=TOLERANCE,
    )
    assert result.passed


def test_promotes_improvement_over_champion():
    result = evaluate_gate(
        0.88, has_signature=True, floor=FLOOR, champion_score=0.85,
    )
    assert result.passed
    assert "+0.0300" in result.comparison


@pytest.mark.parametrize(
    "score,expected",
    [
        (FLOOR, True),           # exactly at the floor passes
        (FLOOR - 1e-9, False),   # a hair below does not
    ],
)
def test_floor_boundary_is_inclusive(score, expected):
    assert evaluate_gate(score, has_signature=True, floor=FLOOR).passed is expected


def test_tolerance_boundary_is_inclusive():
    """Exactly at the tolerance edge passes. Rejecting at equality surprises people."""
    result = evaluate_gate(
        0.84, has_signature=True, floor=FLOOR, champion_score=0.85,
        regression_tolerance=0.01,
    )
    assert result.passed


def test_multiple_failures_are_all_reported():
    """The failure message should name every reason, not just the first."""
    result = evaluate_gate(
        0.60, has_signature=False, floor=FLOOR, champion_score=0.85,
    )
    assert not result.passed
    assert set(result.failed_checks) == {
        "signature_present", "above_floor", "no_regression",
    }


class TestGateTags:
    def test_no_reserved_characters_in_keys(self):
        """Unity Catalog reserves . = > < % & ? and backslash in tag keys."""
        result = evaluate_gate(0.83, has_signature=True, floor=FLOOR)
        tags = gate_tags(
            result, candidate_score=0.83, metric="holdout_macro_f1",
            floor=FLOOR, regression_tolerance=TOLERANCE, run_id="abc123",
        )
        reserved = set(".=><%&?\\")
        for key in tags:
            assert not (set(key) & reserved), f"reserved character in tag key: {key}"

    def test_records_every_check(self):
        result = evaluate_gate(0.71, has_signature=True, floor=FLOOR)
        tags = gate_tags(
            result, candidate_score=0.71, metric="holdout_macro_f1",
            floor=FLOOR, regression_tolerance=TOLERANCE, run_id="abc123",
        )
        assert tags["gate_passed"] == "false"
        assert tags["gate_check_above_floor"] == "false"
        assert tags["gate_check_signature_present"] == "true"
        assert tags["source_run_id"] == "abc123"

    def test_all_values_are_strings(self):
        """The registry API takes strings; a float here fails at call time."""
        result = evaluate_gate(0.83, has_signature=True, floor=FLOOR)
        tags = gate_tags(
            result, candidate_score=0.83, metric="holdout_macro_f1",
            floor=FLOOR, regression_tolerance=TOLERANCE, run_id="abc123",
        )
        assert all(isinstance(v, str) for v in tags.values())
