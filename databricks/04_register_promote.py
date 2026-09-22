# Databricks notebook source
# MAGIC %md
# MAGIC # 04 — Register and promote
# MAGIC
# MAGIC Registers the latest training run in the Unity Catalog model registry, then
# MAGIC decides whether it becomes the `champion`.
# MAGIC
# MAGIC ## Why aliases, not stages
# MAGIC
# MAGIC MLflow's Staging/Production stages are deprecated in the Unity Catalog
# MAGIC registry. Aliases replace them: a named, movable pointer to a model version.
# MAGIC Consumers load `@champion`, and promotion is moving the alias — no code
# MAGIC change downstream, and rollback is moving it back.
# MAGIC
# MAGIC ## The gate
# MAGIC
# MAGIC A candidate becomes champion only if all of these hold:
# MAGIC
# MAGIC 1. **Signature present.** The registry uses it to reject schema mismatches.
# MAGIC 2. **Absolute floor.** Holdout macro F1 at or above a minimum, so a broken
# MAGIC    pipeline can't promote a model just because there's no champion yet.
# MAGIC 3. **No regression.** If a champion exists, the candidate may not score
# MAGIC    meaningfully worse on the same holdout.
# MAGIC
# MAGIC Every decision is written onto the model version as tags — pass or fail,
# MAGIC with reasons — so the registry itself is the audit trail. A failed gate
# MAGIC raises, which fails the job. The procedure is enforced by the system, not
# MAGIC by someone remembering to check.

# COMMAND ----------

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

CATALOG = "workspace"
SCHEMA = "pamap2"
MODEL_NAME = f"{CATALOG}.{SCHEMA}.activity_classifier"
EXPERIMENT = "/Shared/pamap2-activity-classifier"

METRIC = "holdout_macro_f1"
ABSOLUTE_FLOOR = 0.75
REGRESSION_TOLERANCE = 0.01  # candidate may trail champion by at most this

mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()

# COMMAND ----------

# Most recent finished run. In the scheduled job this is the run the training
# task just produced.
experiment = mlflow.get_experiment_by_name(EXPERIMENT)
runs = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="attributes.status = 'FINISHED'",
    order_by=["attributes.start_time DESC"],
    max_results=1,
)
assert not runs.empty, f"No finished runs in {EXPERIMENT}"

run_id = runs.iloc[0]["run_id"]
candidate_score = runs.iloc[0][f"metrics.{METRIC}"]
print(f"candidate run {run_id}: {METRIC}={candidate_score:.4f}")

# COMMAND ----------

version = mlflow.register_model(f"runs:/{run_id}/model", MODEL_NAME)
print(f"registered {MODEL_NAME} version {version.version}")

# COMMAND ----------

def champion_score():
    """Score of the current champion, or None if there isn't one."""
    try:
        champ = client.get_model_version_by_alias(MODEL_NAME, "champion")
    except MlflowException:
        return None, None
    champ_run = client.get_run(champ.run_id)
    return champ.version, champ_run.data.metrics.get(METRIC)


champ_version, champ_score = champion_score()

checks = {}

model_info = mlflow.models.get_model_info(f"models:/{MODEL_NAME}/{version.version}")
checks["signature_present"] = model_info.signature is not None

checks["above_floor"] = candidate_score >= ABSOLUTE_FLOOR

if champ_score is None:
    checks["no_regression"] = True
    regression_note = "no existing champion"
else:
    checks["no_regression"] = candidate_score >= champ_score - REGRESSION_TOLERANCE
    regression_note = (
        f"champion v{champ_version} {champ_score:.4f}, "
        f"delta {candidate_score - champ_score:+.4f}"
    )

passed = all(checks.values())

# COMMAND ----------

# Record the decision on the version itself, pass or fail.
tags = {
    "gate.passed": str(passed).lower(),
    "gate.metric": METRIC,
    "gate.candidate_score": f"{candidate_score:.4f}",
    "gate.floor": str(ABSOLUTE_FLOOR),
    "gate.regression_tolerance": str(REGRESSION_TOLERANCE),
    "gate.comparison": regression_note,
    "source.run_id": run_id,
}
for name, ok in checks.items():
    tags[f"gate.check.{name}"] = str(ok).lower()

for k, v in tags.items():
    client.set_model_version_tag(MODEL_NAME, version.version, k, v)

# COMMAND ----------

if passed:
    client.set_registered_model_alias(MODEL_NAME, "champion", version.version)
    print(f"PROMOTED v{version.version} to @champion ({regression_note})")
else:
    client.set_registered_model_alias(MODEL_NAME, "challenger", version.version)
    failed = [n for n, ok in checks.items() if not ok]
    raise AssertionError(
        f"Promotion gate failed for v{version.version}: {failed}. "
        f"Score {candidate_score:.4f}, {regression_note}. "
        f"Version kept as @challenger."
    )
