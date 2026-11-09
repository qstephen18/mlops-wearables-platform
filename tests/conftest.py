"""Shared pytest fixtures.

The Spark session is session-scoped because creating one costs several
seconds. Two local cores and shuffle partitions set to 1 keep the tiny
test frames from being split across 200 empty partitions, which is the
usual reason local Spark tests feel slow.
"""

import pytest


@pytest.fixture(scope="session")
def spark():
    from pyspark.sql import SparkSession

    session = (
        SparkSession.builder.master("local[2]")
        .appName("unit-tests")
        .config("spark.sql.shuffle.partitions", "1")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()
