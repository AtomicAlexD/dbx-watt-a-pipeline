# Could have used databricks connect here or just run them on databricks but figured i'd go with local. Might come back to bite me but I don't know what dev env people prefer. 
import pytest
from pyspark.sql import SparkSession


@pytest.fixture(scope="session")
def spark():
    return (
        SparkSession.builder
        .master("local[1]")
        .appName("tests")
        .getOrCreate()
    )