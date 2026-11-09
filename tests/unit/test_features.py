"""Unit tests for the feature transforms.

These need a local Spark session, so they are slower than the pure-Python
tests — seconds rather than milliseconds. Still fast enough for CI.

The NaN test is the one that matters. It encodes the bug that cost the
most time on this project: NaN is not null in Spark, isNull() is false for
NaN, and last(ignorenulls=True) skips nulls but not NaN. Without the
normalisation the forward fill does nothing and every downstream aggregate
returns NaN while every stage reports success.
"""

import math

import pytest

# Skip the whole module if PySpark is absent rather than failing at import.
# Deselecting by marker does not help: pytest still *collects* the module,
# so a top-level import error kills the run before markers are evaluated.
pytest.importorskip("pyspark")

from pyspark.sql import functions as F  # noqa: E402

from src.features import (  # noqa: E402
    assign_split,
    drop_partial_windows,
    forward_fill_within_activity,
    normalize_nan_to_null,
)

pytestmark = pytest.mark.spark


class TestNormalizeNanToNull:
    def test_nan_becomes_null(self, spark):
        df = spark.createDataFrame(
            [(1, float("nan")), (2, 72.0)], ["id", "heart_rate"]
        )
        out = normalize_nan_to_null(df, ["heart_rate"])
        assert out.filter(F.col("heart_rate").isNull()).count() == 1

    def test_nan_is_not_null_without_normalization(self, spark):
        """The bug itself, pinned as a test. isNull() is false for NaN."""
        df = spark.createDataFrame([(1, float("nan"))], ["id", "heart_rate"])
        assert df.filter(F.col("heart_rate").isNull()).count() == 0

    def test_real_values_survive(self, spark):
        df = spark.createDataFrame(
            [(1, 72.0), (2, 0.0), (3, -1.5)], ["id", "heart_rate"]
        )
        out = normalize_nan_to_null(df, ["heart_rate"]).collect()
        assert [r["heart_rate"] for r in out] == [72.0, 0.0, -1.5]

    def test_zero_is_not_treated_as_missing(self, spark):
        """A real zero reading must not be confused with a missing one."""
        df = spark.createDataFrame([(1, 0.0)], ["id", "heart_rate"])
        out = normalize_nan_to_null(df, ["heart_rate"])
        assert out.filter(F.col("heart_rate").isNull()).count() == 0

    def test_unlisted_columns_untouched(self, spark):
        df = spark.createDataFrame(
            [(1, float("nan"), float("nan"))], ["id", "heart_rate", "temp_c"]
        )
        out = normalize_nan_to_null(df, ["heart_rate"]).collect()[0]
        assert out["heart_rate"] is None
        assert math.isnan(out["temp_c"])


class TestForwardFill:
    def test_carries_last_value_forward(self, spark):
        df = spark.createDataFrame(
            [
                (1, 1, 1.0, 70.0),
                (1, 1, 2.0, None),
                (1, 1, 3.0, None),
                (1, 1, 4.0, 75.0),
            ],
            ["subject_id", "activity_id", "timestamp_s", "heart_rate"],
        )
        out = forward_fill_within_activity(
            df, "heart_rate",
            partition_cols=["subject_id", "activity_id"],
            order_col="timestamp_s",
        ).orderBy("timestamp_s").collect()
        assert [r["heart_rate_filled"] for r in out] == [70.0, 70.0, 70.0, 75.0]

    def test_does_not_carry_across_activity_boundary(self, spark):
        """The bounding is the whole point — a reading must not survive a change."""
        df = spark.createDataFrame(
            [
                (1, 1, 1.0, 70.0),
                (1, 2, 2.0, None),
            ],
            ["subject_id", "activity_id", "timestamp_s", "heart_rate"],
        )
        out = forward_fill_within_activity(
            df, "heart_rate",
            partition_cols=["subject_id", "activity_id"],
            order_col="timestamp_s",
        ).orderBy("timestamp_s").collect()
        assert out[0]["heart_rate_filled"] == 70.0
        assert out[1]["heart_rate_filled"] is None

    def test_does_not_carry_across_subjects(self, spark):
        df = spark.createDataFrame(
            [
                (1, 1, 1.0, 70.0),
                (2, 1, 2.0, None),
            ],
            ["subject_id", "activity_id", "timestamp_s", "heart_rate"],
        )
        out = forward_fill_within_activity(
            df, "heart_rate",
            partition_cols=["subject_id", "activity_id"],
            order_col="timestamp_s",
        ).orderBy("subject_id").collect()
        assert out[1]["heart_rate_filled"] is None

    def test_leading_nulls_stay_null(self, spark):
        """Nothing to carry forward from. Backfilling would invent data."""
        df = spark.createDataFrame(
            [(1, 1, 1.0, None), (1, 1, 2.0, 70.0)],
            ["subject_id", "activity_id", "timestamp_s", "heart_rate"],
        )
        out = forward_fill_within_activity(
            df, "heart_rate",
            partition_cols=["subject_id", "activity_id"],
            order_col="timestamp_s",
        ).orderBy("timestamp_s").collect()
        assert out[0]["heart_rate_filled"] is None


class TestAssignSplit:
    def test_subjects_land_in_the_right_split(self, spark):
        df = spark.createDataFrame([(101,), (107,), (999,)], ["subject_id"])
        out = {r["subject_id"]: r["split"] for r in assign_split(df).collect()}
        assert out == {101: "train", 107: "holdout", 999: "unassigned"}

    def test_no_subject_appears_in_both_splits(self, spark):
        """Guards against the leak the whole evaluation strategy exists to prevent."""
        df = spark.createDataFrame([(s,) for s in range(101, 110)], ["subject_id"])
        out = assign_split(df)
        overlap = (
            out.filter(F.col("split") == "train").select("subject_id").distinct()
            .intersect(
                out.filter(F.col("split") == "holdout").select("subject_id").distinct()
            )
        )
        assert overlap.count() == 0


class TestDropPartialWindows:
    def test_keeps_complete_windows(self, spark):
        df = spark.createDataFrame([(1, 512)], ["win", "sample_count"])
        assert drop_partial_windows(df).count() == 1

    def test_drops_stubs(self, spark):
        df = spark.createDataFrame([(1, 40)], ["win", "sample_count"])
        assert drop_partial_windows(df).count() == 0

    def test_tolerates_small_sample_loss(self, spark):
        """A few dropped samples should not discard an otherwise good window."""
        df = spark.createDataFrame([(1, 500)], ["win", "sample_count"])
        assert drop_partial_windows(df).count() == 1
