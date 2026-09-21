# Databricks notebook source
# MAGIC %md
# MAGIC # 02 — Features
# MAGIC
# MAGIC Bronze readings to a windowed feature table.
# MAGIC
# MAGIC ## Decisions worth defending
# MAGIC
# MAGIC **Window size 5.12s (512 samples at 100Hz), 50% overlap.** Standard for
# MAGIC this dataset. Long enough to capture a full gait cycle, short enough that
# MAGIC a window rarely spans two activities.
# MAGIC
# MAGIC **Heart rate is forward-filled within subject and activity.** The HR
# MAGIC monitor samples at ~9Hz against 100Hz IMU, so ~90% of rows are null.
# MAGIC Filling is bounded by activity so a reading never carries across an
# MAGIC activity change.
# MAGIC
# MAGIC **Split is by subject, not by row.** Rows within a window are correlated;
# MAGIC a random split leaks the same person into train and test and produces an
# MAGIC accuracy number that means nothing. Subjects 101–106 train, 107–109 held
# MAGIC out — which also gives the drift monitor a genuine population shift to
# MAGIC detect rather than injected noise.

# COMMAND ----------

from pyspark.sql import Window
from pyspark.sql import functions as F

CATALOG = "workspace"
SCHEMA = "pamap2"
BRONZE = f"{CATALOG}.{SCHEMA}.bronze_readings"
FEATURES = f"{CATALOG}.{SCHEMA}.gold_features"

SAMPLE_HZ = 100
WINDOW_SECONDS = 5.12
OVERLAP = 0.5

TRAIN_SUBJECTS = [101, 102, 103, 104, 105, 106]
HOLDOUT_SUBJECTS = [107, 108, 109]

# COMMAND ----------

bronze = spark.table(BRONZE)

# Forward-fill heart rate within (subject, activity), ordered by time.
hr_window = (
    Window.partitionBy("subject_id", "activity_id")
    .orderBy("timestamp_s")
    .rowsBetween(Window.unboundedPreceding, 0)
)

filled = bronze.withColumn(
    "heart_rate_filled", F.last("heart_rate", ignorenulls=True).over(hr_window)
)

# COMMAND ----------

# Assign each row to an overlapping window. Two passes with a half-window offset
# is the cheap way to get 50% overlap without a self-join.
step = WINDOW_SECONDS * OVERLAP

windowed = filled.withColumn(
    "window_id", F.floor(F.col("timestamp_s") / F.lit(WINDOW_SECONDS))
).withColumn(
    "window_offset_id",
    F.floor((F.col("timestamp_s") - F.lit(step)) / F.lit(WINDOW_SECONDS)),
)

base = windowed.select(
    "subject_id", "activity_id", "heart_rate_filled",
    F.col("window_id").alias("win"), F.lit(0).alias("phase"),
    *[c for c in bronze.columns if c.startswith(("hand_", "chest_", "ankle_"))],
)

offset = windowed.select(
    "subject_id", "activity_id", "heart_rate_filled",
    F.col("window_offset_id").alias("win"), F.lit(1).alias("phase"),
    *[c for c in bronze.columns if c.startswith(("hand_", "chest_", "ankle_"))],
)

stacked = base.unionByName(offset)

# COMMAND ----------

SENSOR_COLS = [c for c in bronze.columns if c.startswith(("hand_", "chest_", "ankle_"))]

aggs = []
for col in SENSOR_COLS:
    aggs += [
        F.avg(col).alias(f"{col}_mean"),
        F.stddev(col).alias(f"{col}_std"),
        F.min(col).alias(f"{col}_min"),
        F.max(col).alias(f"{col}_max"),
        # Signal energy — mean of squares. Separates static from dynamic activity.
        F.avg(F.col(col) * F.col(col)).alias(f"{col}_energy"),
    ]

aggs += [
    F.avg("heart_rate_filled").alias("heart_rate_mean"),
    F.stddev("heart_rate_filled").alias("heart_rate_std"),
    F.max("heart_rate_filled").alias("heart_rate_max"),
    F.count("*").alias("sample_count"),
]

expected_samples = int(SAMPLE_HZ * WINDOW_SECONDS)

features = (
    stacked.groupBy("subject_id", "activity_id", "win", "phase")
    .agg(*aggs)
    # Drop partial windows at activity boundaries. 90% threshold tolerates the
    # occasional dropped sample without keeping stubs.
    .filter(F.col("sample_count") >= expected_samples * 0.9)
    .withColumn(
        "split",
        F.when(F.col("subject_id").isin(TRAIN_SUBJECTS), F.lit("train"))
        .when(F.col("subject_id").isin(HOLDOUT_SUBJECTS), F.lit("holdout"))
        .otherwise(F.lit("unassigned")),
    )
    .withColumn("generated_at", F.current_timestamp())
)

(
    features.write.mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(FEATURES)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Checks
# MAGIC
# MAGIC Class balance matters here — PAMAP2 is uneven, and knowing that before
# MAGIC training is what makes macro F1 the right metric instead of accuracy.

# COMMAND ----------

display(
    spark.table(FEATURES)
    .groupBy("split")
    .agg(
        F.count("*").alias("windows"),
        F.countDistinct("subject_id").alias("subjects"),
        F.countDistinct("activity_id").alias("activities"),
    )
)

display(
    spark.table(FEATURES)
    .groupBy("activity_id", "split")
    .count()
    .orderBy("activity_id", "split")
)
