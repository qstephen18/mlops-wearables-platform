"""Unit tests for drift statistics.

The edge cases matter more than the happy path here. A monitor that
returns 0.0 when it should return NaN reports "no drift" and is
indistinguishable from a working monitor.
"""

import numpy as np
import pandas as pd
import pytest

from src.drift import PSI_SIGNIFICANT, categorical_psi, drift_status, psi

rng = np.random.default_rng(42)


@pytest.fixture
def reference():
    return pd.Series(rng.normal(0, 1, 5000))


def test_identical_distributions_have_near_zero_psi(reference):
    current = pd.Series(rng.normal(0, 1, 5000))
    assert psi(reference, current) < 0.05


def test_shifted_distribution_is_detected(reference):
    current = pd.Series(rng.normal(2.0, 1, 5000))
    assert psi(reference, current) > PSI_SIGNIFICANT


def test_widened_distribution_is_detected(reference):
    """Same centre, different spread. Mean-based checks would miss this."""
    current = pd.Series(rng.normal(0, 3.0, 5000))
    assert psi(reference, current) > PSI_SIGNIFICANT


def test_psi_is_never_negative(reference):
    for scale in (0.5, 1.0, 2.0):
        value = psi(reference, pd.Series(rng.normal(0, scale, 2000)))
        assert value >= 0


def test_empty_current_returns_nan(reference):
    assert np.isnan(psi(reference, pd.Series([], dtype=float)))


def test_too_few_reference_rows_returns_nan():
    assert np.isnan(psi(pd.Series([1.0, 2.0, 3.0]), pd.Series(rng.normal(0, 1, 100))))


def test_constant_reference_returns_nan():
    """A constant reference collapses the quantile edges. NaN, not 0.0."""
    assert np.isnan(psi(pd.Series([5.0] * 1000), pd.Series(rng.normal(0, 1, 1000))))


def test_nulls_are_dropped_not_counted(reference):
    with_nulls = pd.concat([reference, pd.Series([np.nan] * 500)])
    clean = psi(reference, pd.Series(rng.normal(0, 1, 5000)))
    dirty = psi(with_nulls, pd.Series(rng.normal(0, 1, 5000)))
    assert abs(clean - dirty) < 0.05


def test_disjoint_ranges_do_not_produce_infinity(reference):
    """An empty bin would divide by zero without the epsilon floor."""
    current = pd.Series(rng.normal(100, 1, 1000))
    value = psi(reference, current)
    assert np.isfinite(value)
    assert value > PSI_SIGNIFICANT


class TestCategoricalPsi:
    def test_identical_class_mix_is_near_zero(self):
        ref = pd.Series([1, 1, 2, 2, 3] * 200)
        cur = pd.Series([1, 1, 2, 2, 3] * 200)
        assert categorical_psi(ref, cur) < 0.01

    def test_unseen_category_is_detected(self):
        ref = pd.Series([1, 2, 3] * 300)
        cur = pd.Series([1, 2, 3, 99] * 300)
        assert categorical_psi(ref, cur) > 0

    def test_empty_input_returns_nan(self):
        assert np.isnan(categorical_psi(pd.Series([1, 2, 3]), pd.Series([], dtype=int)))


class TestDriftStatus:
    def test_degradation_outranks_drift(self):
        """Measured performance loss beats every proxy signal."""
        status = drift_status(
            features_significant=12, prediction_psi=0.9, performance_delta=-0.20
        )
        assert status == "performance_degraded"

    def test_drift_with_intact_performance_only_warns(self):
        """The real observed case: inputs moved, the model held up."""
        status = drift_status(
            features_significant=8, prediction_psi=0.05, performance_delta=-0.01
        )
        assert status == "drift_detected"

    def test_stable_run_is_ok(self):
        status = drift_status(
            features_significant=0, prediction_psi=0.02, performance_delta=0.004
        )
        assert status == "ok"

    def test_improvement_is_not_degradation(self):
        status = drift_status(
            features_significant=0, prediction_psi=0.01, performance_delta=0.08
        )
        assert status == "ok"

    def test_tolerance_boundary(self):
        """Exactly at tolerance is not degraded."""
        assert drift_status(
            features_significant=0, prediction_psi=0.0,
            performance_delta=-0.05, performance_tolerance=0.05,
        ) == "ok"
