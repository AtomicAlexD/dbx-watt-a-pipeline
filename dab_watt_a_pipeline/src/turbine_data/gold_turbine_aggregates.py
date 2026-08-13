from pyspark import pipelines as dp

from functions import compute_daily_summary, flag_anomalies

# Read at module scope: needed at decoration time to route these tables
# into the gold catalog/schema rather than the pipeline's bronze default.
_gold_catalog = spark.conf.get("turbine_pipeline.gold_catalog")
_gold_schema = spark.conf.get("turbine_pipeline.gold_schema")
_silver_catalog = spark.conf.get("turbine_pipeline.silver_catalog")
_silver_schema = spark.conf.get("turbine_pipeline.silver_schema")
_silver_table = f"{_silver_catalog}.{_silver_schema}.silver_turbine_readings"


@dp.materialized_view(
    name=f"{_gold_catalog}.{_gold_schema}.daily_turbine_summary",
    comment=(
        "Per-turbine, per-day min/max/avg power output. Includes "
        "imputed values (forward-fill cannot introduce a new min or "
        "max); imputed_count is surfaced so a heavily-fabricated day "
        "is visible rather than silently blended in."
    ),
)
def daily_turbine_summary():
    silver_df = spark.read.table(_silver_table)
    return compute_daily_summary(silver_df)


@dp.materialized_view(
    name=f"{_gold_catalog}.{_gold_schema}.turbine_anomalies",
    comment=(
        "Real (non-imputed) turbine readings flagged where power_output "
        "falls outside 2 standard deviations of that turbine's own "
        "readings for that day."
    ),
)
def turbine_anomalies():
    silver_df = spark.read.table(_silver_table)
    return flag_anomalies(silver_df)