# Databricks notebook source
# MAGIC %md
# MAGIC # 07 — Integration tests
# MAGIC
# MAGIC Runs the integration suite against the tables that actually exist in
# MAGIC this workspace. Unit tests run in CI; these cannot, because they need
# MAGIC a workspace and real data.
# MAGIC
# MAGIC As a job task this fails the run when the deployed data is wrong,
# MAGIC which is the point — a test that only prints is not a test once
# MAGIC nobody is watching.

# COMMAND ----------

# MAGIC %pip install pytest
# MAGIC %restart_python

# COMMAND ----------

import os
import sys

import pytest

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "pamap2")

os.environ["TEST_CATALOG"] = dbutils.widgets.get("catalog")
os.environ["TEST_SCHEMA"] = dbutils.widgets.get("schema")

# Repo root on the path so `from src...` resolves inside a Git folder.
repo_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# COMMAND ----------

exit_code = pytest.main([
    f"{repo_root}/tests/integration",
    "-v", "-p", "no:cacheprovider",
])

if exit_code != 0:
    raise AssertionError(f"Integration tests failed (pytest exit code {exit_code})")

print("Integration tests passed.")
