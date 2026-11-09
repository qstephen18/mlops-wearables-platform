"""Promotion gate logic.

Extracted from 04_register_promote so it can be unit tested without a
workspace, a registry, or a cluster. The notebook keeps the MLflow calls;
this module holds the decision.

That split is the point: the thing most worth testing is the rule that
decides what reaches production, and a notebook is the one place you
cannot test it.
"""

from dataclasses import dataclass

# Absorbs floating-point representation error at threshold boundaries.
_FLOAT_SLACK = 1e-9


@dataclass(frozen=True)
class GateResult:
    """Outcome of evaluating a candidate model against the gate."""

    passed: bool
    checks: dict
    comparison: str

    @property
    def failed_checks(self) -> list:
        return [name for name, ok in self.checks.items() if not ok]


def evaluate_gate(
    candidate_score: float,
    *,
    has_signature: bool,
    floor: float,
    champion_score: float | None = None,
    regression_tolerance: float = 0.01,
) -> GateResult:
    """Decide whether a candidate model should become champion.

    Three checks, all of which must pass:

    1. A model signature is present. Unity Catalog uses it to reject input
       schema mismatches at load time rather than at inference time.
    2. The candidate clears an absolute floor, so a broken pipeline cannot
       promote a model merely because no champion exists yet.
    3. The candidate does not regress against the incumbent by more than
       the tolerance, measured on the same holdout.

    champion_score of None means no champion exists; the regression check
    passes by default in that case.

    Boundaries are inclusive: a candidate exactly at the floor passes, and
    a candidate exactly at the tolerance edge passes. Both are deliberate —
    a gate that rejects at equality is surprising to whoever hits it.
    """
    if champion_score is None:
        no_regression = True
        comparison = "no existing champion"
    else:
        delta = candidate_score - champion_score
        # Subtracting two floats introduces representation error: 0.84 - 0.85
        # is -0.010000000000000009, so a candidate exactly at the tolerance
        # edge would be rejected by a naive >= comparison. _FLOAT_SLACK
        # absorbs that. Found by the boundary test, not by reading the code.
        no_regression = delta >= -(regression_tolerance + _FLOAT_SLACK)
        comparison = f"champion {champion_score:.4f}, delta {delta:+.4f}"

    checks = {
        "signature_present": bool(has_signature),
        "above_floor": candidate_score >= floor,
        "no_regression": no_regression,
    }

    return GateResult(
        passed=all(checks.values()),
        checks=checks,
        comparison=comparison,
    )


def gate_tags(
    result: GateResult,
    *,
    candidate_score: float,
    metric: str,
    floor: float,
    regression_tolerance: float,
    run_id: str,
) -> dict:
    """Tags written onto the model version so the registry is the audit trail.

    Unity Catalog reserves '.' in tag keys, so every key uses underscores.
    That is a real difference from open-source MLflow and the reason this
    function exists rather than the dict being built inline.
    """
    tags = {
        "gate_passed": str(result.passed).lower(),
        "gate_metric": metric,
        "gate_candidate_score": f"{candidate_score:.4f}",
        "gate_floor": str(floor),
        "gate_regression_tolerance": str(regression_tolerance),
        "gate_comparison": result.comparison,
        "source_run_id": run_id,
    }
    for name, ok in result.checks.items():
        tags[f"gate_check_{name}"] = str(ok).lower()
    return tags
