from pyspark import pipelines as dp

from functions import clean_turbine_readings, report_remaining_nulls

# Read at module scope, not inside the function body: the fully-qualified
# name below is needed at decoration time to route this table into the
# silver catalog/schema instead of the pipeline's bronze default.
_silver_catalog = spark.conf.get("turbine_pipeline.silver_catalog")
_silver_schema = spark.conf.get("turbine_pipeline.silver_schema")


@dp.materialized_view(
    name=f"{_silver_catalog}.{_silver_schema}.silver_turbine_readings",
    comment=(
        "Cleaned turbine sensor readings: deduplicated by (turbine_id, "
        "timestamp), physically invalid values nulled and flagged, "
        "missing values forward-filled per turbine where a prior valid "
        "reading exists. See functions.py for the full contract of each "
        "transform."
    ),
)
def silver_turbine_readings():
    bronze_df = spark.read.table("bronze_turbine_readings")
    cleaned_df = clean_turbine_readings(bronze_df)
    report_remaining_nulls(cleaned_df)

    return cleaned_df