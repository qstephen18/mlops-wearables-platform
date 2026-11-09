"""Drift statistics.

Extracted from 06_drift_monitor. Pure numpy and pandas, no Spark, so the
maths can be tested directly. PSI has several edge cases that fail quietly
if they are not handled — an empty comparison set, a constant reference,
a bin with no mass — and quiet failure in a monitor is worse than no
monitor at all.
"""

import numpy as np
import pandas as pd

PSI_MODERATE = 0.10
PSI_SIGNIFICANT = 0.25
_EPS = 1e-6


def psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """Population Stability Index between a reference and a current sample.

    Quantile edges rather than equal width, so bins carry comparable mass
    regardless of the feature's shape. An epsilon floor keeps an empty bin
    from producing infinity.

    Returns NaN — rather than raising or returning 0.0 — when the inputs
    cannot support a meaningful comparison. NaN propagates visibly;
    returning 0.0 would look like "no drift" and be indistinguishable from
    a real result.
    """
    reference = pd.Series(reference).dropna()
    current = pd.Series(current).dropna()

    if len(reference) < bins or len(current) == 0:
        return float("nan")

    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        # A constant or near-constant reference collapses the edges.
        return float("nan")
    edges[0], edges[-1] = -np.inf, np.inf

    ref_pct = np.clip(np.histogram(reference, bins=edges)[0] / len(reference), _EPS, None)
    cur_pct = np.clip(np.histogram(current, bins=edges)[0] / len(current), _EPS, None)

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def categorical_psi(reference: pd.Series, current: pd.Series) -> float:
    """PSI over a discrete distribution — used for predicted class mix."""
    reference = pd.Series(reference).dropna()
    current = pd.Series(current).dropna()

    if len(reference) == 0 or len(current) == 0:
        return float("nan")

    categories = sorted(set(reference.unique()) | set(current.unique()))
    ref_pct = np.clip(
        reference.value_counts(normalize=True).reindex(categories, fill_value=0).values, _EPS, None
    )
    cur_pct = np.clip(
        current.value_counts(normalize=True).reindex(categories, fill_value=0).values, _EPS, None
    )
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def drift_status(
    *,
    features_significant: int,
    prediction_psi: float,
    performance_delta: float,
    performance_tolerance: float = 0.05,
) -> str:
    """Classify a monitoring run.

    Order matters. Measured performance degradation outranks every drift
    signal, because drift is a proxy for a degradation you usually cannot
    measure yet. When you can measure it, the proxy stops being the
    interesting number.
    """
    if performance_delta < -performance_tolerance:
        return "performance_degraded"
    if features_significant > 0 or prediction_psi >= PSI_SIGNIFICANT:
        return "drift_detected"
    if prediction_psi >= PSI_MODERATE:
        return "drift_moderate"
    return "ok"
