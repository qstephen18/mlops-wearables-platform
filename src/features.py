"""Feature engineering transforms.

Extracted from 02_features. Each function takes a DataFrame and returns a
DataFrame, with no table names and no side effects, so it can be tested
against a small hand-built frame on a local Spark session.

The notebook keeps the reads and writes; this module keeps the logic.
"""

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

SAMPLE_HZ = 100
WINDOW_SECONDS = 5.12

TRAIN_SUBJECTS = [101, 102, 103, 104, 105, 106]
HOLDOUT_SUBJECTS = [107, 108, 109]


def normalize_nan_to_null(df: DataFrame, columns: list[str]) -> DataFrame:
    """Replace NaN with null in the named numeric columns.

    NaN and null are different in Spark: isNull() is false for NaN, and
    last(ignorenulls=True) skips nulls but not NaN. The source encodes
    missing readings as NaN, so without this the forward fill silently
    does nothing and every aggregate over a gap returns NaN — while every
    stage reports success.
    """
    return df.select(
        *[
            F.when(F.isnan(F.col(c)), None).otherwise(F.col(c)).alias(c)
            if c in columns
            else F.col(c)
            for c in df.columns
        ]
    )


def forward_fill_within_activity(
    df: DataFrame,
    column: str,
    *,
    partition_cols: list[str],
    order_col: str,
    output_col: str | None = None,
) -> DataFrame:
    """Carry the last observed value forward, bounded by the partition.

    Bounded deliberately: a heart rate reading must never carry across an
    activity change, because the gap between activities is exactly where
    the value is least likely to still be true.
    """
    output_col = output_col or f"{column}_filled"
    w = (
        Window.partitionBy(*partition_cols)
        .orderBy(order_col)
        .rowsBetween(Window.unboundedPreceding, 0)
    )
    return df.withColumn(output_col, F.last(column, ignorenulls=True).over(w))


def assign_split(
    df: DataFrame,
    *,
    subject_col: str = "subject_id",
    train_subjects: list[int] = None,
    holdout_subjects: list[int] = None,
) -> DataFrame:
    """Label rows train or holdout by subject, never by row.

    Rows inside a window are highly correlated. A random split puts the
    same person on both sides and the resulting score measures memory
    rather than generalisation.
    """
    train_subjects = train_subjects or TRAIN_SUBJECTS
    holdout_subjects = holdout_subjects or HOLDOUT_SUBJECTS
    return df.withColumn(
        "split",
        F.when(F.col(subject_col).isin(train_subjects), F.lit("train"))
        .when(F.col(subject_col).isin(holdout_subjects), F.lit("holdout"))
        .otherwise(F.lit("unassigned")),
    )


def drop_partial_windows(
    df: DataFrame,
    *,
    count_col: str = "sample_count",
    window_seconds: float = WINDOW_SECONDS,
    sample_hz: int = SAMPLE_HZ,
    completeness: float = 0.9,
) -> DataFrame:
    """Drop windows at activity boundaries that are mostly empty.

    The 90% threshold tolerates the occasional dropped sample without
    keeping stubs that would otherwise produce unstable aggregates.
    """
    expected = int(sample_hz * window_seconds)
    return df.filter(F.col(count_col) >= expected * completeness)
