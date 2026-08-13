"""
Silver-layer transformations for turbine sensor readings.

Every function here is pure: DataFrame in, DataFrame out (or, for
report_remaining_nulls, DataFrame in, log output out). None of them
touch Spark I/O, Autoloader, or Unity Catalog, so all of them are unit
testable with a small local SparkSession and hand-built fixtures.

Intended usage, chained via .transform():

    silver_df = (
        bronze_df
        .transform(dedupe_readings)
        .transform(flag_and_nullify_invalid)
        .transform(impute_missing_values)
    )
    report_remaining_nulls(silver_df)

Design notes:
- "Outliers" here means physically impossible readings (negative power,
  wind direction outside 0-360 degrees) — not statistical anomalies.
  Statistical anomaly detection (>2 std dev) is a Gold-layer concern, not
  a data-quality one, since it's a judgement about otherwise-valid data.
- Invalid/missing values are imputed (forward-filled per turbine, ordered
  by time) rather than dropped, to preserve time-series continuity.
- Every fabricated value is flagged, not just silently overwritten. This
  keeps the cleaning process auditable and turns "which sensors are
  flaky" into a queryable question rather than a byproduct.
- Scope assumption: "missing entries" is interpreted as null values
  within rows that exist, not whole absent rows in the time series. The
  provided sample data contains neither (744 complete hourly rows per
  turbine, zero nulls, zero gaps), so this is a stated design choice
  rather than something verifiable from the data itself. Detecting
  wholly-absent rows would require generating an expected timestamp grid
  per turbine and left-joining onto it — treated as out of scope here.
- Known, accepted limitation: forward-fill has no value to look back to
  for a turbine's very first reading(s) if it's null. This is treated as
  a legitimate outcome (e.g. a sensor broken since install has genuinely
  no signal to recover), not a bug to engineer around — see
  report_remaining_nulls below, which surfaces this rather than either
  hiding it or halting the pipeline over it.
"""

import logging

from pyspark.sql import DataFrame, functions as F
from pyspark.sql.window import Window

VALUE_COLUMNS = ["power_output", "wind_speed", "wind_direction"]

logger = logging.getLogger(__name__)


def dedupe_readings(df: DataFrame) -> DataFrame:
    """Keep one row per (turbine_id, timestamp), preferring the most
    recently ingested version if the same reading arrived more than once."""
    w = Window.partitionBy("turbine_id", "timestamp").orderBy(
        F.col("_ingested_at").desc()
    )
    return (
        df.withColumn("_row_num", F.row_number().over(w))
        .filter(F.col("_row_num") == 1)
        .drop("_row_num")
    )


def flag_and_nullify_invalid(df: DataFrame) -> DataFrame:
    """Record *why* a reading is missing or invalid before touching
    anything, then null out physically impossible values so they're
    picked up by imputation rather than left in place or silently dropped.

    Adds, per value column:
      - {col}_was_missing:  the raw value was null or NaN
      - {col}_was_invalid:  the raw value was present but out of bounds

    Bounds:
      - power_output < 0: physically impossible. No upper bound is
        applied — that would require each turbine's rated capacity,
        which isn't provided by the brief or the data. Documented gap
        rather than an invented threshold.
      - wind_speed < 0: physically impossible. wind_speed > 100 (m/s) is
        a defensive sanity ceiling for an obviously-faulty reading.
      - wind_direction outside [0, 360): a compass bearing has no valid
        values beyond a full rotation.

    NaN vs null: Spark treats these as distinct for float columns, and
    isNull() alone does not catch NaN. wind_direction is an integer
    column and cannot hold NaN, so it's checked with isNull() only.

    The invalid check is only evaluated once a value is confirmed not
    missing (`F.when(~is_missing, ...)`) — this matters because Spark
    orders NaN as greater than any value, including for plain comparison
    operators, so an unguarded upper-bound check would otherwise
    misclassify a NaN as "invalid" rather than "missing".
    """
    float_columns = {"power_output", "wind_speed"}
    bounds = {
        "power_output": lambda c: c < 0,
        "wind_speed": lambda c: (c < 0) | (c > 100),
        "wind_direction": lambda c: (c < 0) | (c >= 360),
    }

    for col in VALUE_COLUMNS:
        c = F.col(col)
        is_invalid = bounds[col]
        is_missing = (c.isNull() | F.isnan(c)) if col in float_columns else c.isNull()

        df = (
            df.withColumn(f"{col}_was_missing", is_missing)
            .withColumn(
                f"{col}_was_invalid",
                F.when(~is_missing, is_invalid(c)).otherwise(F.lit(False)),
            )
            .withColumn(col, F.when(is_missing | is_invalid(c), None).otherwise(c))
        )
    return df


def impute_missing_values(df: DataFrame) -> DataFrame:
    """Forward-fill nulls per turbine, ordered by time. Requires
    flag_and_nullify_invalid to have run first so invalid values are
    already null, not just missing values.

    Adds, per value column:
      - {col}_was_imputed: this row's null was successfully filled from
        a prior reading. False both when the value was never null AND
        when it was null but nothing existed before it to fill from
        (e.g. a turbine's very first reading) — the flag reflects
        whether a fill actually happened, not merely whether one was
        attempted.
    """
    w = (
        Window.partitionBy("turbine_id")
        .orderBy("timestamp")
        .rowsBetween(Window.unboundedPreceding, Window.currentRow)
    )

    for col in VALUE_COLUMNS:
        was_null = F.col(col).isNull()
        filled = F.last(F.col(col), ignorenulls=True).over(w)
        df = df.withColumn(
            f"{col}_was_imputed", was_null & filled.isNotNull()
        ).withColumn(col, filled)
    return df


def clean_turbine_readings(df: DataFrame) -> DataFrame:
    """Convenience wrapper chaining the full Silver cleaning sequence.
    Order matters: dedupe before flagging (so duplicates aren't double
    counted), flag/nullify before impute (so invalid values are null
    before the forward-fill pass runs)."""
    return (
        df.transform(dedupe_readings)
        .transform(flag_and_nullify_invalid)
        .transform(impute_missing_values)
    )


def report_remaining_nulls(df: DataFrame, columns: list[str] = VALUE_COLUMNS) -> None:
    """Logs a warning for any nulls surviving the full cleaning chain,
    rather than raising. A remaining null means imputation had nothing
    to fill from (e.g. a turbine's data starts null, such as a sensor
    broken since install) — an accepted, expected outcome given the
    scope assumption above, not a pipeline failure. Warning rather than
    silence keeps it visible in run logs and explains gaps a reviewer
    would otherwise see unexplained in downstream charts, without
    blocking the rest of the data from processing."""
    for col in columns:
        null_count = df.filter(F.col(col).isNull()).count()
        if null_count > 0:
            logger.warning(
                "%s unfillable nulls remain in %s after cleaning "
                "(no prior valid reading existed to impute from)",
                null_count,
                col,
            )