# Databricks notebook source
# MAGIC %md
# MAGIC # 06 — Drift monitoring
# MAGIC
# MAGIC Compares the scored batch against the distribution the champion was
# MAGIC trained on, and records the result.
# MAGIC
# MAGIC ## Three things are measured, and they are not the same thing
# MAGIC
# MAGIC **Feature drift** — have the inputs moved? Population Stability Index per
# MAGIC feature, reference being the training distribution.
# MAGIC
# MAGIC **Prediction drift** — has the output mix moved? PSI over the predicted
# MAGIC class distribution against the training label distribution.
# MAGIC
# MAGIC **Realised performance** — is the model actually still right? Macro F1 on
# MAGIC the scored batch, compared against the score the champion was promoted on.
# MAGIC
# MAGIC The third is the one that matters and the one you usually cannot have.
# MAGIC Labels arrive late, or never. That is the entire reason feature and
# MAGIC prediction drift exist: they are early proxies for a performance drop you
# MAGIC will not be able to measure for weeks. This dataset has labels, so all
# MAGIC three are computed here — which makes it possible to show whether the
# MAGIC proxies actually tracked the real thing.
# MAGIC
# MAGIC ## Why there is real drift to find
# MAGIC
# MAGIC The holdout is different people, not held-out rows from the same people.
# MAGIC Different body mechanics, sensor placement and resting heart rate produce
# MAGIC a genuine population shift. Nothing is injected.

# COMMAND ----------

import numpy as np
import pandas as pd
import mlflow
from mlflow import MlflowClient
from pyspark.sql import functions as F

CATALOG = "workspace"
SCHEMA = "pamap2"
MODEL_NAME = f"{CATALOG}.{SCHEMA}.activity_classifier"
FEATURES = f"{CATALOG}.{SCHEMA}.gold_features"
PREDICTIONS = f"{CATALOG}.{SCHEMA}.predictions"
DRIFT = f"{CATALOG}.{SCHEMA}.drift_metrics"

# PSI convention: <0.1 stable, 0.1-0.25 moderate, >0.25 significant.
PSI_MODERATE = 0.10
PSI_SIGNIFICANT = 0.25
BINS = 10
# Largest tolerated drop in macro F1 against the promoted score.
PERFORMANCE_TOLERANCE = 0.05

mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()

# COMMAND ----------

def psi(reference: pd.Series, current: pd.Series, bins: int = BINS) -> float:
    """Population Stability Index.

    Bins the reference into quantiles and compares proportions. Quantile edges
    rather than equal width, so bins carry comparable mass regardless of the
    feature's shape. Epsilon floor keeps an empty bin from producing infinity.
    """
    reference = reference.dropna()
    current = current.dropna()
    if len(reference) < bins or len(current) == 0:
        return np.nan

    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return np.nan
    edges[0], edges[-1] = -np.inf, np.inf

    eps = 1e-6
    ref_pct = np.histogram(reference, bins=edges)[0] / len(reference)
    cur_pct = np.histogram(current, bins=edges)[0] / len(current)
    ref_pct = np.clip(ref_pct, eps, None)
    cur_pct = np.clip(cur_pct, eps, None)

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def categorical_psi(reference: pd.Series, current: pd.Series) -> float:
    """PSI over a discrete distribution — used for predicted classes."""
    categories = sorted(set(reference.unique()) | set(current.unique()))
    eps = 1e-6
    ref_pct = np.clip(reference.value_counts(normalize=True).reindex(categories, fill_value=0).values, eps, None)
    cur_pct = np.clip(current.value_counts(normalize=True).reindex(categories, fill_value=0).values, eps, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))

# COMMAND ----------

champion = client.get_model_version_by_alias(MODEL_NAME, "champion")
champion_run = client.get_run(champion.run_id)
promoted_score = champion_run.data.metrics.get("holdout_macro_f1")

# Reference is the data the champion was trained on, not "last month's batch".
# Drift is measured against what the model actually learned.
reference = spark.table(FEATURES).filter(F.col("split") == "train").toPandas()

scored = (
    spark.table(PREDICTIONS)
    .filter(F.col("model_version") == int(champion.version))
    .toPandas()
)
assert not scored.empty, f"No predictions for champion v{champion.version} — run 05 first."

# Join predictions back to their feature rows.
current = (
    spark.table(FEATURES)
    .filter(F.col("split") == "holdout")
    .toPandas()
    .merge(
        scored[["subject_id", "win", "phase", "predicted_activity_id"]],
        on=["subject_id", "win", "phase"],
        how="inner",
    )
)

print(f"champion v{champion.version}, promoted at {promoted_score:.4f}")
print(f"reference {len(reference)} rows, current {len(current)} rows")

# COMMAND ----------

ID_COLS = {"subject_id", "activity_id", "win", "phase", "split", "generated_at",
           "sample_count", "predicted_activity_id"}
feature_cols = [c for c in reference.columns if c not in ID_COLS]

feature_psi = pd.DataFrame({
    "feature": feature_cols,
    "psi": [psi(reference[c], current[c]) for c in feature_cols],
}).dropna().sort_values("psi", ascending=False)

drifted = feature_psi[feature_psi["psi"] >= PSI_SIGNIFICANT]
moderate = feature_psi[
    (feature_psi["psi"] >= PSI_MODERATE) & (feature_psi["psi"] < PSI_SIGNIFICANT)
]

print(f"{len(drifted)} significant, {len(moderate)} moderate, of {len(feature_psi)} features")
display(feature_psi.head(20))

# COMMAND ----------

prediction_psi = categorical_psi(
    reference["activity_id"], current["predicted_activity_id"]
)

from sklearn.metrics import f1_score

realised_f1 = f1_score(
    current["activity_id"], current["predicted_activity_id"], average="macro"
)
performance_delta = realised_f1 - promoted_score

print(f"prediction PSI: {prediction_psi:.4f}")
print(f"realised macro F1: {realised_f1:.4f} (delta {performance_delta:+.4f})")

# COMMAND ----------

status = "ok"
if performance_delta < -PERFORMANCE_TOLERANCE:
    status = "performance_degraded"
elif len(drifted) > 0 or prediction_psi >= PSI_SIGNIFICANT:
    status = "drift_detected"
elif len(moderate) > 0 or prediction_psi >= PSI_MODERATE:
    status = "drift_moderate"

row = pd.DataFrame([{
    "model_name": MODEL_NAME,
    "model_version": int(champion.version),
    "model_run_id": champion.run_id,
    "reference_rows": len(reference),
    "current_rows": len(current),
    "features_significant": len(drifted),
    "features_moderate": len(moderate),
    "max_feature_psi": float(feature_psi["psi"].max()),
    "top_drifted_feature": feature_psi.iloc[0]["feature"],
    "prediction_psi": prediction_psi,
    "promoted_macro_f1": float(promoted_score),
    "realised_macro_f1": float(realised_f1),
    "performance_delta": float(performance_delta),
    "status": status,
}])

(
    spark.createDataFrame(row)
    .withColumn("computed_at", F.current_timestamp())
    .write.mode("append").option("mergeSchema", "true").saveAsTable(DRIFT)
)

print(f"status: {status}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Alerting
# MAGIC
# MAGIC Degraded performance raises, which fails the job task and surfaces through
# MAGIC the workflow's own notification path rather than a bespoke alerting
# MAGIC mechanism. Drift alone warns but does not fail: inputs moving is a signal
# MAGIC to investigate, not evidence the model is wrong. Only a measured drop in
# MAGIC performance is evidence of that.
# MAGIC
# MAGIC In production, where labels are delayed, this inverts — significant drift
# MAGIC becomes the actionable signal precisely because realised performance is
# MAGIC not yet knowable.

# COMMAND ----------

if status == "performance_degraded":
    raise AssertionError(
        f"Champion v{champion.version} macro F1 {realised_f1:.4f} is "
        f"{abs(performance_delta):.4f} below its promoted score "
        f"{promoted_score:.4f} (tolerance {PERFORMANCE_TOLERANCE}). "
        f"Top drifted feature: {feature_psi.iloc[0]['feature']} "
        f"(PSI {feature_psi.iloc[0]['psi']:.3f})."
    )

if status.startswith("drift"):
    print(
        f"WARNING: {status}. {len(drifted)} features at or above PSI "
        f"{PSI_SIGNIFICANT}, prediction PSI {prediction_psi:.3f}. "
        f"Performance still within tolerance."
    )

display(spark.table(DRIFT).orderBy(F.col("computed_at").desc()))
