"""Integration tests against real Unity Catalog tables.

These need a workspace and real tables, so they do not run in CI — they
run in Databricks after a deploy. See databricks/07_run_tests.py.

Unit tests ask whether the logic is correct. These ask whether the thing
that actually got deployed produced data with the shape and the
properties the next stage depends on.

Set CATALOG and SCHEMA via environment variables to point at the
environment under test.
"""

import os

import pytest
from pyspark.sql import functions as F

CATALOG = os.environ.get("TEST_CATALOG", "workspace")
SCHEMA = os.environ.get("TEST_SCHEMA", "pamap2")

BRONZE = f"{CATALOG}.{SCHEMA}.bronze_readings"
FEATURES = f"{CATALOG}.{SCHEMA}.gold_features"
PREDICTIONS = f"{CATALOG}.{SCHEMA}.predictions"

EXPECTED_SUBJECTS = 9
HR_NULL_RATE_RANGE = (0.85, 0.95)


@pytest.fixture(scope="module")
def spark():
    from pyspark.sql import SparkSession

    return SparkSession.builder.getOrCreate()


class TestBronze:
    def test_table_exists_and_is_populated(self, spark):
        assert spark.table(BRONZE).count() > 0

    def test_all_subjects_present(self, spark):
        n = spark.table(BRONZE).select("subject_id").distinct().count()
        assert n == EXPECTED_SUBJECTS, f"expected {EXPECTED_SUBJECTS} subjects, found {n}"

    def test_heart_rate_null_rate_matches_source_sampling(self, spark):
        """HR samples near 9Hz against 100Hz IMU, so ~90% of rows have none.

        A rate near zero means NaN was not normalised to null and the
        forward fill downstream will silently do nothing.
        """
        rate = (
            spark.table(BRONZE)
            .select(F.avg(F.col("heart_rate").isNull().cast("int")).alias("r"))
            .first()["r"]
        )
        low, high = HR_NULL_RATE_RANGE
        assert low <= rate <= high, f"heart rate null rate {rate:.3f} outside {low}-{high}"

    def test_transient_activity_removed(self, spark):
        """Activity 0 is the gap between activities, not a class."""
        assert spark.table(BRONZE).filter(F.col("activity_id") == 0).count() == 0

    def test_no_orientation_columns(self, spark):
        """Documented invalid by the dataset authors; dropped at ingest."""
        cols = spark.table(BRONZE).columns
        assert not [c for c in cols if "orient_" in c]


class TestGoldFeatures:
    def test_table_exists_and_is_populated(self, spark):
        assert spark.table(FEATURES).count() > 0

    def test_no_null_keys(self, spark):
        df = spark.table(FEATURES)
        for key in ("subject_id", "activity_id", "win", "split"):
            assert df.filter(F.col(key).isNull()).count() == 0, f"null {key}"

    def test_expected_columns_present(self, spark):
        cols = set(spark.table(FEATURES).columns)
        required = {
            "subject_id", "activity_id", "win", "phase", "split",
            "heart_rate_mean", "heart_rate_std", "sample_count",
        }
        missing = required - cols
        assert not missing, f"missing columns: {missing}"

    def test_feature_null_rates_are_low(self, spark):
        """Confirms the NaN normalisation survived the forward fill and windowing."""
        row = (
            spark.table(FEATURES)
            .select(
                F.avg(F.col("heart_rate_mean").isNull().cast("int")).alias("hr"),
                F.avg(F.col("hand_acc16_x_mean").isNull().cast("int")).alias("acc"),
            )
            .first()
        )
        assert row["hr"] < 0.05, f"HR feature null rate {row['hr']:.3f}"
        assert row["acc"] < 0.05, f"IMU feature null rate {row['acc']:.3f}"

    def test_no_subject_in_both_splits(self, spark):
        """The leak the entire evaluation strategy exists to prevent."""
        df = spark.table(FEATURES)
        train = df.filter(F.col("split") == "train").select("subject_id").distinct()
        holdout = df.filter(F.col("split") == "holdout").select("subject_id").distinct()
        assert train.intersect(holdout).count() == 0

    def test_both_splits_populated(self, spark):
        df = spark.table(FEATURES)
        assert df.filter(F.col("split") == "train").count() > 0
        assert df.filter(F.col("split") == "holdout").count() > 0

    def test_every_activity_appears_in_both_splits(self, spark):
        """A class missing from holdout cannot be evaluated at all."""
        df = spark.table(FEATURES)
        train = {r["activity_id"] for r in df.filter(F.col("split") == "train").select("activity_id").distinct().collect()}
        holdout = {r["activity_id"] for r in df.filter(F.col("split") == "holdout").select("activity_id").distinct().collect()}
        assert train == holdout, f"only in train: {train - holdout}, only in holdout: {holdout - train}"

    def test_no_partial_windows(self, spark):
        assert spark.table(FEATURES).filter(F.col("sample_count") < 460).count() == 0


class TestPredictions:
    def test_every_prediction_carries_lineage(self, spark):
        """A prediction with no model version cannot be traced to anything."""
        df = spark.table(PREDICTIONS)
        if df.count() == 0:
            pytest.skip("no predictions yet")
        for col in ("model_name", "model_version", "model_run_id", "scored_at"):
            assert df.filter(F.col(col).isNull()).count() == 0, f"null {col}"

    def test_predicted_classes_are_known_activities(self, spark):
        """Guards against a label-encoding mismatch between train and score."""
        preds = spark.table(PREDICTIONS)
        if preds.count() == 0:
            pytest.skip("no predictions yet")
        known = {r["activity_id"] for r in spark.table(FEATURES).select("activity_id").distinct().collect()}
        predicted = {r["predicted_activity_id"] for r in preds.select("predicted_activity_id").distinct().collect()}
        assert predicted <= known, f"unknown predicted classes: {predicted - known}"
