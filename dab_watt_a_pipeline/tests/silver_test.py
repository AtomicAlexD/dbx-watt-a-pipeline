"""
Tests for silver-layer cleaning transformations (functions.py).

Testing philosophy: each test proves one specific claim about the
production code's behavior, stated in the docstring rather than left
implicit. Where a test exists because of a real bug found during
development (not just an anticipated edge case), that's named explicitly
as a regression test — it's evidence the logic was actually exercised
against tricky cases, not just written to spec.

One test class per function in functions.py, mirroring that file 1:1.
"""

import logging
from datetime import datetime

import pytest
from pyspark.sql.types import (
    StructType, StructField, IntegerType, DoubleType, TimestampType
)

from src.turbine_data.functions import (
    dedupe_readings,
    flag_and_nullify_invalid,
    impute_missing_values,
    report_remaining_nulls,
)


class TestFlagAndNullifyInvalid:
    """Contract: for each of power_output, wind_speed, wind_direction,
    every row gets a `{col}_was_missing` and `{col}_was_invalid` flag,
    the two are mutually exclusive, and any row where either is true has
    its value nulled out (never silently left in place, never dropped)."""

    def test_negative_power_output_is_flagged_invalid_and_nulled(self, spark):
        """Negative generation is the clearest physically-impossible
        case in the brief. Proves the invalid path nulls the value
        rather than leaving the bad reading in place."""
        df = spark.createDataFrame(
            [(1, -5.0, 10.0, 180)],
            ["turbine_id", "power_output", "wind_speed", "wind_direction"],
        )

        result = flag_and_nullify_invalid(df).collect()[0]

        assert result["power_output"] is None
        assert result["power_output_was_invalid"] is True
        assert result["power_output_was_missing"] is False

    def test_wind_speed_nan_is_missing_not_invalid(self, spark):
        """Regression test. Originally, wind_speed's upper bound
        (`c > 100`) accidentally caught NaN too, because Spark orders
        NaN as greater than any value — including for plain comparison
        operators, not just sorting. That misclassified a garbage
        sensor reading as 'invalid' instead of 'missing', which matters
        because these flags feed a sensor-health report; the two mean
        different things operationally. Fixed by checking isnan()
        explicitly and short-circuiting the invalid check once a value
        is already known missing."""
        df = spark.createDataFrame(
            [(1, 3.0, float("nan"), 180)],
            ["turbine_id", "power_output", "wind_speed", "wind_direction"],
        )

        result = flag_and_nullify_invalid(df).collect()[0]

        assert result["wind_speed"] is None
        assert result["wind_speed_was_missing"] is True
        assert result["wind_speed_was_invalid"] is False

    @pytest.mark.parametrize(
        "wind_direction,expect_invalid",
        [
            pytest.param(0, False, id="lower_bound_inclusive"),
            pytest.param(359, False, id="upper_bound_inclusive"),
            pytest.param(360, True, id="upper_bound_exclusive_edge"),
            pytest.param(-1, True, id="just_below_lower_bound"),
            pytest.param(180, False, id="mid_range_valid"),
        ],
    )
    def test_wind_direction_bounds(self, spark, wind_direction, expect_invalid):
        """A compass bearing is valid on [0, 360). This proves the exact
        boundary behavior at both ends, not just 'roughly the middle
        works' — 0 and 359 must be accepted, 360 and -1 must not, since
        an off-by-one here would silently corrupt real readings sitting
        exactly at the boundary rather than obviously-wrong values."""
        df = spark.createDataFrame(
            [(1, 3.0, 10.0, wind_direction)],
            ["turbine_id", "power_output", "wind_speed", "wind_direction"],
        )

        result = flag_and_nullify_invalid(df).collect()[0]

        assert result["wind_direction_was_invalid"] is expect_invalid

    def test_null_wind_direction_is_missing_not_invalid(self, spark):
        """Regression-shaped guard: applying the bounds lambda directly
        to a null value in Spark's three-valued logic would produce
        `None`, not `False` — a silent type bug where a filter like
        `.filter("wind_direction_was_invalid")` would just drop the row
        instead of correctly treating it as valid-but-missing. Proves
        the missing/invalid guard actually short-circuits."""
        schema = StructType([
            StructField("turbine_id", IntegerType()),
            StructField("power_output", DoubleType()),
            StructField("wind_speed", DoubleType()),
            StructField("wind_direction", IntegerType()),
        ])
        df = spark.createDataFrame([(1, 3.0, 10.0, None)], schema=schema)

        result = flag_and_nullify_invalid(df).collect()[0]

        assert result["wind_direction_was_missing"] is True
        assert result["wind_direction_was_invalid"] is False  # not None

    def test_valid_row_is_untouched(self, spark):
        """Negative-space test: proves the function doesn't flag or
        modify anything for a row with no problems, i.e. it's not
        accidentally over-eager and nulling out good data."""
        df = spark.createDataFrame(
            [(1, 3.0, 10.0, 180)],
            ["turbine_id", "power_output", "wind_speed", "wind_direction"],
        )

        result = flag_and_nullify_invalid(df).collect()[0]

        assert (result["power_output"], result["wind_speed"], result["wind_direction"]) == (3.0, 10.0, 180)
        assert not any(
            result[f"{c}_{suffix}"]
            for c in ["power_output", "wind_speed", "wind_direction"]
            for suffix in ["was_missing", "was_invalid"]
        )


class TestDedupeReadings:
    """Contract: exactly one row survives per (turbine_id, timestamp),
    keeping the most recently ingested version when the same reading
    arrived more than once."""

    SCHEMA = StructType([
        StructField("turbine_id", IntegerType()),
        StructField("timestamp", TimestampType()),
        StructField("power_output", DoubleType()),
        StructField("_ingested_at", TimestampType()),
    ])

    def test_duplicate_reading_keeps_most_recently_ingested(self, spark):
        """Two rows for the same turbine at the same timestamp, differing
        only in when they were ingested and what value they carry —
        proves the later ingestion wins, not just an arbitrary one."""
        df = spark.createDataFrame(
            [
                (1, datetime(2026, 1, 1, 0, 0), 3.0, datetime(2026, 1, 1, 1, 0)),
                (1, datetime(2026, 1, 1, 0, 0), 3.5, datetime(2026, 1, 1, 2, 0)),  # ingested later
            ],
            schema=self.SCHEMA,
        )

        result = dedupe_readings(df).collect()

        assert len(result) == 1
        assert result[0]["power_output"] == 3.5

    def test_dedupe_is_scoped_per_turbine_and_timestamp(self, spark):
        """Two different turbines sharing the exact same timestamp are
        NOT duplicates of each other. Proves the window is partitioned
        by (turbine_id, timestamp) together, not by timestamp alone —
        a bug here would silently drop real readings from one turbine
        whenever another turbine happened to report at the same time."""
        df = spark.createDataFrame(
            [
                (1, datetime(2026, 1, 1, 0, 0), 3.0, datetime(2026, 1, 1, 1, 0)),
                (2, datetime(2026, 1, 1, 0, 0), 4.0, datetime(2026, 1, 1, 1, 0)),
            ],
            schema=self.SCHEMA,
        )

        result = dedupe_readings(df).collect()

        assert len(result) == 2
        assert {r["turbine_id"] for r in result} == {1, 2}

    def test_no_duplicates_leaves_all_rows_untouched(self, spark):
        """Negative-space test: proves dedupe doesn't drop legitimate
        rows when there's nothing to deduplicate."""
        df = spark.createDataFrame(
            [
                (1, datetime(2026, 1, 1, 0, 0), 3.0, datetime(2026, 1, 1, 1, 0)),
                (1, datetime(2026, 1, 1, 1, 0), 3.2, datetime(2026, 1, 1, 2, 0)),
            ],
            schema=self.SCHEMA,
        )

        result = dedupe_readings(df).collect()

        assert len(result) == 2


class TestImputeMissingValues:
    """Contract: nulls are forward-filled per turbine, ordered by time,
    from the nearest prior valid reading. {col}_was_imputed is true only
    when a fill genuinely happened — not merely when a value started
    null — so it can't be used to distinguish 'we fixed this' from 'we
    tried and had nothing to fill from'."""

    SCHEMA = StructType([
        StructField("turbine_id", IntegerType()),
        StructField("timestamp", TimestampType()),
        StructField("power_output", DoubleType()),
        StructField("wind_speed", DoubleType()),
        StructField("wind_direction", IntegerType()),
    ])

    def _row(self, turbine_id, ts, power=None, wind_speed=None, direction=None):
        return (turbine_id, ts, power, wind_speed, direction)

    def test_mid_series_gap_is_forward_filled_from_last_valid_reading(self, spark):
        """The core contract, e.g. Monday and Tuesday have real
        readings, Wednesday is null: Wednesday should be filled from
        Tuesday's value, and correctly flagged as imputed. This is the
        ordinary, expected case a real gap in an otherwise-healthy
        turbine's data looks like."""
        df = spark.createDataFrame(
            [
                self._row(1, datetime(2026, 1, 1, 0, 0), power=3.0),   # Monday
                self._row(1, datetime(2026, 1, 2, 0, 0), power=3.2),   # Tuesday
                self._row(1, datetime(2026, 1, 3, 0, 0), power=None),  # Wednesday - gap
            ],
            schema=self.SCHEMA,
        )

        result = impute_missing_values(df).orderBy("timestamp").collect()

        assert result[2]["power_output"] == 3.2  # filled from Tuesday
        assert result[2]["power_output_was_imputed"] is True
        assert result[0]["power_output_was_imputed"] is False
        assert result[1]["power_output_was_imputed"] is False

    def test_imputation_is_scoped_per_turbine(self, spark):
        """A gap in turbine 1's data must not get filled from turbine
        2's readings, even if turbine 2's row sits earlier in the
        window ordering. Proves the partition boundary is respected,
        not just the time ordering."""
        df = spark.createDataFrame(
            [
                self._row(2, datetime(2026, 1, 1, 0, 0), power=9.0),
                self._row(1, datetime(2026, 1, 1, 1, 0), power=None),
            ],
            schema=self.SCHEMA,
        )

        result = impute_missing_values(df).filter("turbine_id = 1").collect()[0]

        # If turbine 2's value leaked across, this would be 9.0.
        assert result["power_output"] is None

    def test_leading_null_is_left_null_and_not_marked_imputed(self, spark):
        """Regression test. Previously, was_imputed was set based only
        on whether a value started null — so a turbine's unfillable
        first reading (nothing exists before it to copy from) was
        incorrectly flagged as successfully imputed, when nothing was
        actually filled. This documents the accepted, deliberate scope:
        a permanently missing leading value (e.g. a sensor broken since
        install) stays null, and the flag now honestly reflects that no
        fill occurred."""
        df = spark.createDataFrame(
            [
                self._row(1, datetime(2026, 1, 1, 0, 0), power=None),
                self._row(1, datetime(2026, 1, 1, 1, 0), power=3.0),
            ],
            schema=self.SCHEMA,
        )

        result = impute_missing_values(df).orderBy("timestamp").collect()

        assert result[0]["power_output"] is None
        assert result[0]["power_output_was_imputed"] is False

    def test_valid_reading_is_not_flagged_as_imputed(self, spark):
        """Negative-space test: proves the function doesn't mark good
        data as fabricated."""
        df = spark.createDataFrame(
            [self._row(1, datetime(2026, 1, 1, 0, 0), power=3.0, wind_speed=10.0, direction=180)],
            schema=self.SCHEMA,
        )

        result = impute_missing_values(df).collect()[0]

        assert result["power_output_was_imputed"] is False
        assert result["wind_speed_was_imputed"] is False
        assert result["wind_direction_was_imputed"] is False


class TestReportRemainingNulls:
    """Contract: logs a warning per column with unfillable nulls, and
    never raises — a remaining null is an accepted outcome (per the
    scope assumption that we don't fabricate data for permanently
    broken sensors), not a pipeline failure."""

    SCHEMA = StructType([
        StructField("turbine_id", IntegerType()),
        StructField("power_output", DoubleType()),
        StructField("wind_speed", DoubleType()),
        StructField("wind_direction", IntegerType()),
    ])

    def test_warns_when_nulls_remain(self, spark, caplog):
        """Proves a remaining null produces a visible warning rather
        than silently vanishing into downstream tables unexplained."""
        df = spark.createDataFrame(
            [(1, None, 10.0, 180)], schema=self.SCHEMA
        )

        with caplog.at_level(logging.WARNING):
            report_remaining_nulls(df)

        assert any("power_output" in record.message for record in caplog.records)

    def test_does_not_raise_when_nulls_remain(self, spark):
        """Regression-shaped guard: this function used to raise
        ValueError on any remaining null, which would halt the whole
        pipeline over an accepted, expected outcome. Proves that no
        longer happens."""
        df = spark.createDataFrame(
            [(1, None, 10.0, 180)], schema=self.SCHEMA
        )

        report_remaining_nulls(df)  # should not raise

    def test_no_warning_when_data_is_clean(self, spark, caplog):
        """Negative-space test: proves it doesn't warn unnecessarily on
        fully clean data, which would train reviewers to ignore the
        warning as noise."""
        df = spark.createDataFrame(
            [(1, 3.0, 10.0, 180)], schema=self.SCHEMA
        )

        with caplog.at_level(logging.WARNING):
            report_remaining_nulls(df)

        assert len(caplog.records) == 0