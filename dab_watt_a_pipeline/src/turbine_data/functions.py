from pyspark.sql import DataFrame, functions as F
from pyspark.sql.window import Window

VALUE_COLUMNS = ["power_output", "wind_speed", "wind_direction"]


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
      - {col}_was_missing:  the raw value was already null
      - {col}_was_invalid:  the raw value was present but out of bounds
    """
    # Kept this simple with some lambda functions rather than a more complex UDF since the bounds are simple and unlikely to change.
    # Also the data actually isn't that bad after inspection I'll still add these anyway and prove them via unit tests.
    bounds = {
        "power_output": lambda c: c < 0, # Data shows between 1.5 and 4.5 here so I could be more restrictive. Demonstrated slightly tighter bounds on the wind speed.
        "wind_speed": lambda c: (c < 0) | (c > 100), # 100mph is pretty ridiculour but adding in an upper bound just in case.
        "wind_direction": lambda c: (c < 0) | (c >= 360),
    }

    for col in VALUE_COLUMNS:
        is_invalid = bounds[col]
        df = (
            df.withColumn(f"{col}_was_missing", F.col(col).isNull())
            .withColumn(
                f"{col}_was_invalid",
                F.when(F.col(col).isNotNull(), is_invalid(F.col(col))).otherwise(
                    F.lit(False)
                ),
            )
            .withColumn(col, F.when(is_invalid(F.col(col)), None).otherwise(F.col(col)))
        )
    return df


def impute_missing_values(df: DataFrame) -> DataFrame:
    """Forward-fill nulls per turbine, ordered by time. Requires
    flag_and_nullify_invalid to have run first so invalid values are
    already null, not just missing values.

    Adds, per value column:
      - {col}_was_imputed: this row's value was fabricated by forward-fill
    """
    w = (
        Window.partitionBy("turbine_id")
        .orderBy("timestamp")
        .rowsBetween(Window.unboundedPreceding, Window.currentRow)
    )

    for col in VALUE_COLUMNS:
        df = df.withColumn(f"{col}_was_imputed", F.col(col).isNull()).withColumn(
            col, F.last(F.col(col), ignorenulls=True).over(w)
        )
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


def validate_no_remaining_nulls(df: DataFrame, columns: list[str] = VALUE_COLUMNS) -> None:
    """Post-cleaning assertion, not a .transform step — raises rather than
    returning a DataFrame. A null surviving to this point means Silver's
    cleaning logic has a bug, not that the data is bad."""
    for col in columns:
        null_count = df.filter(F.col(col).isNull()).count()
        if null_count > 0:
            raise ValueError(f"Unexpected nulls in {col} after cleaning: {null_count}")