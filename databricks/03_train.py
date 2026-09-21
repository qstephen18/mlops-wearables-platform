# Databricks notebook source
# MAGIC %md
# MAGIC # 03 — Train
# MAGIC
# MAGIC Gradient-boosted classifier over the windowed features, tracked in MLflow.
# MAGIC
# MAGIC The model is deliberately boring. What matters is that the run is
# MAGIC reproducible and the artifact is promotable: params, metrics, the input
# MAGIC signature, and the exact subject split all recorded on the run, so a
# MAGIC result can be traced back to the data that produced it.
# MAGIC
# MAGIC **Macro F1 is the headline metric, not accuracy.** PAMAP2 classes are
# MAGIC unbalanced — accuracy rewards a model that predicts the common activities
# MAGIC and ignores the rare ones.

# COMMAND ----------

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, f1_score, accuracy_score

CATALOG = "workspace"
SCHEMA = "pamap2"
FEATURES = f"{CATALOG}.{SCHEMA}.gold_features"
EXPERIMENT = "/Shared/pamap2-activity-classifier"

mlflow.set_experiment(EXPERIMENT)

# COMMAND ----------

df = spark.table(FEATURES).toPandas()

ID_COLS = ["subject_id", "activity_id", "win", "phase", "split", "generated_at", "sample_count"]
FEATURE_COLS = [c for c in df.columns if c not in ID_COLS]

train = df[df["split"] == "train"]
holdout = df[df["split"] == "holdout"]

X_train, y_train = train[FEATURE_COLS], train["activity_id"]
X_holdout, y_holdout = holdout[FEATURE_COLS], holdout["activity_id"]

print(f"train: {X_train.shape}, holdout: {X_holdout.shape}, features: {len(FEATURE_COLS)}")

# COMMAND ----------

PARAMS = {
    "max_iter": 200,
    "learning_rate": 0.1,
    "max_depth": 8,
    "l2_regularization": 1.0,
    "early_stopping": True,
    "validation_fraction": 0.1,
    "random_state": 42,
}

with mlflow.start_run(run_name="hgb-baseline") as run:
    # HistGradientBoosting handles NaN natively — no imputation step to drift
    # out of sync between training and inference.
    model = HistGradientBoostingClassifier(**PARAMS)
    model.fit(X_train, y_train)

    preds = model.predict(X_holdout)

    metrics = {
        "holdout_macro_f1": f1_score(y_holdout, preds, average="macro"),
        "holdout_weighted_f1": f1_score(y_holdout, preds, average="weighted"),
        "holdout_accuracy": accuracy_score(y_holdout, preds),
        "train_rows": len(X_train),
        "holdout_rows": len(X_holdout),
        "n_features": len(FEATURE_COLS),
    }

    mlflow.log_params(PARAMS)
    mlflow.log_metrics(metrics)

    # Provenance: what data produced this model, not just what score it got.
    mlflow.set_tags({
        "train_subjects": ",".join(map(str, sorted(train["subject_id"].unique()))),
        "holdout_subjects": ",".join(map(str, sorted(holdout["subject_id"].unique()))),
        "feature_table": FEATURES,
        "split_strategy": "subject-level holdout",
    })

    report = classification_report(y_holdout, preds, zero_division=0)
    mlflow.log_text(report, "classification_report.txt")

    # The signature is what lets the registry reject an input schema mismatch
    # at deploy time instead of at 3am in production.
    signature = infer_signature(X_train, model.predict(X_train))

    mlflow.sklearn.log_model(
        model,
        name="model",
        signature=signature,
        input_example=X_train.head(3),
    )

    print(f"run_id: {run.info.run_id}")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    print()
    print(report)
