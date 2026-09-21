# Databricks notebook source
# MAGIC %md
# MAGIC # 01 — Ingest
# MAGIC
# MAGIC Raw PAMAP2 `.dat` files to a bronze Delta table.
# MAGIC
# MAGIC The files are space-delimited with no header. Layout, per the dataset
# MAGIC documentation: column 1 timestamp, column 2 activity ID, column 3 heart
# MAGIC rate, then three 17-column IMU blocks (hand, chest, ankle).
# MAGIC
# MAGIC Two things are dropped here and both are documented dataset properties
# MAGIC rather than judgement calls:
# MAGIC
# MAGIC - **Activity 0** is the transient period between activities, not a class.
# MAGIC - **Orientation columns** are flagged invalid by the dataset authors.
# MAGIC
# MAGIC Subject ID is derived from the filename, which is what makes the later
# MAGIC subject-level train/holdout split possible.

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, DoubleType

CATALOG = "workspace"
SCHEMA = "pamap2"
VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/raw/protocol"
BRONZE = f"{CATALOG}.{SCHEMA}.bronze_readings"

IMU_SENSORS = ["hand", "chest", "ankle"]

# 17 columns per IMU, in file order.
IMU_COLS = (
    ["temp_c"]
    + [f"acc16_{ax}" for ax in "xyz"]
    + [f"acc6_{ax}" for ax in "xyz"]
    + [f"gyro_{ax}" for ax in "xyz"]
    + [f"mag_{ax}" for ax in "xyz"]
    + [f"orient_{i}" for i in range(4)]  # documented invalid, dropped below
)

COLUMNS = ["timestamp_s", "activity_id", "heart_rate"] + [
    f"{sensor}_{col}" for sensor in IMU_SENSORS for col in IMU_COLS
]

assert len(COLUMNS) == 54, f"Expected 54 columns, built {len(COLUMNS)}"

# COMMAND ----------

schema = StructType([StructField(name, DoubleType(), True) for name in COLUMNS])

raw = (
    spark.read.option("delimiter", " ")
    .option("nanValue", "NaN")
    .schema(schema)
    .csv(VOLUME_PATH)
    .withColumn("source_file", F.col("_metadata.file_name"))
)

# subject101.dat -> 101
readings = raw.withColumn(
    "subject_id",
    F.regexp_extract(F.col("source_file"), r"subject(\d+)", 1).cast("int"),
)

# COMMAND ----------

drop_cols = [c for c in readings.columns if "orient_" in c]

bronze = (
    readings.drop(*drop_cols)
    .filter(F.col("activity_id") != 0)  # transient periods
    .filter(F.col("subject_id").isNotNull())
    .withColumn("activity_id", F.col("activity_id").cast("int"))
    .withColumn("ingested_at", F.current_timestamp())
)

(
    bronze.write.mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(BRONZE)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Sanity checks
# MAGIC
# MAGIC Run these rather than assuming the load worked. Nine subjects, activity
# MAGIC IDs in the documented range, and heart rate mostly null — the HR monitor
# MAGIC samples at roughly 9Hz against 100Hz IMU, so most rows have no reading.
# MAGIC That gap is filled in the feature notebook, not here: bronze stays faithful
# MAGIC to the source.

# COMMAND ----------

display(
    spark.table(BRONZE)
    .groupBy("subject_id")
    .agg(
        F.count("*").alias("rows"),
        F.countDistinct("activity_id").alias("activities"),
        F.round(F.avg(F.col("heart_rate").isNull().cast("int")), 3).alias("hr_null_rate"),
    )
    .orderBy("subject_id")
)
