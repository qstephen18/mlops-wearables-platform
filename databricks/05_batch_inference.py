# Databricks notebook source
# MAGIC %md
# MAGIC # 05 — Batch inference
# MAGIC
# MAGIC Scores feature windows with whatever model currently holds `@champion`,
# MAGIC and writes predictions with full lineage.
# MAGIC
# MAGIC Loading by alias rather than version number is what decouples inference
# MAGIC from promotion: when the gate moves `@champion`, the next scheduled run
# MAGIC picks up the new model with no code change. Rollback is moving the alias
# MAGIC back.
# MAGIC
# MAGIC Every prediction row records the model version and run that produced it.
# MAGIC If a prediction is ever questioned, it traces back to a specific model,
# MAGIC a specific training run, and the gate decision tagged on that version.

# COMMAND ----------

import mlflow
from mlflow import MlflowClient
from pyspark.sql import functions as F

CATALOG = "workspace"
SCHEMA = "pamap2"
MODEL_NAME = f"{CATALOG}.{SCHEMA}.activity_classifier"
FEATURES = f"{CATALOG}.{SCHEMA}.gold_features"
PREDICTIONS = f"{CATALOG}.{SCHEMA}.predictions"

# Scoring the holdout subjects stands in for new production traffic —
# people the model has never seen.
SCORE_SPLIT = "holdout"

mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()

# COMMAND ----------

champion = client.get_model_version_by_alias(MODEL_NAME, "champion")
model_uri = f"models:/{MODEL_NAME}@champion"
model = mlflow.pyfunc.load_model(model_uri)

# Use the model's own signature to select inputs, rather than re-deriving the
# feature list here. One source of truth for what the model expects.
input_cols = [c.name for c in model.metadata.get_input_schema().inputs]

print(f"scoring with v{champion.version} (run {champion.run_id}), {len(input_cols)} inputs")

# COMMAND ----------

batch = spark.table(FEATURES).filter(F.col("split") == SCORE_SPLIT).toPandas()

batch["predicted_activity_id"] = model.predict(batch[input_cols]).astype(int)

out = batch[["subject_id", "activity_id", "win", "phase", "predicted_activity_id"]].copy()
out["model_name"] = MODEL_NAME
out["model_version"] = int(champion.version)
out["model_run_id"] = champion.run_id

predictions = spark.createDataFrame(out).withColumn("scored_at", F.current_timestamp())

# Append, not overwrite: the predictions table is a history of what each
# model version said, which is exactly what the drift monitor reads.
predictions.write.mode("append").option("mergeSchema", "true").saveAsTable(PREDICTIONS)

# COMMAND ----------

scored = spark.table(PREDICTIONS).filter(F.col("model_version") == int(champion.version))

display(
    scored.agg(
        F.count("*").alias("rows"),
        F.avg((F.col("predicted_activity_id") == F.col("activity_id")).cast("int")).alias("accuracy"),
        F.max("scored_at").alias("latest_run"),
    )
)
