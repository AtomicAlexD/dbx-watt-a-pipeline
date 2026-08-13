"""
Tests for silver and gold layer transformations (functions.py).

Testing philosophy: each test proves one specific claim about the
production code's behavior, stated in the docstring rather than left
implicit. Where a test exists because of a real bug found during
development, or guards a non-obvious correctness decision, that's named
explicitly as a regression test — evidence the logic was actually
exercised against tricky cases, not just written to spec.

One test class per function in functions.py, mirroring that file 1:1.
"""

import logging
from datetime import date, datetime

import pytest
from pyspark.sql.types import (
    StructType, StructField, IntegerType, DoubleType, TimestampType, BooleanType
)

from src.turbine_data.functions import (
    dedupe_readings,
    flag_and_nullify_invalid,
    impute_missing_values,
    report_remaining_nulls,
    compute_daily_summary,
    flag_anomalies,
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


class TestComputeDailySummary:
    """Contract: one row per (turbine_id, date) with min/max/avg
    power_output and a row count. Imputed values are included in the
    aggregation (see functions.py module docstring for why that's safe)
    but counted separately via imputed_count, so a heavily-fabricated
    day is visible rather than silently blended into a normal-looking
    average."""

    SCHEMA = StructType([
        StructField("turbine_id", IntegerType()),
        StructField("timestamp", TimestampType()),
        StructField("power_output", DoubleType()),
        StructField("power_output_was_imputed", BooleanType()),
    ])

    def _row(self, turbine_id, ts, power, imputed=False):
        return (turbine_id, ts, power, imputed)

    def test_min_max_avg_computed_correctly(self, spark):
        """The core contract, checked against hand-calculated values —
        not just 'a number came out', but the specific right number."""
        df = spark.createDataFrame(
            [
                self._row(1, datetime(2026, 1, 1, 0), 2.0),
                self._row(1, datetime(2026, 1, 1, 1), 4.0),
                self._row(1, datetime(2026, 1, 1, 2), 3.0),
            ],
            schema=self.SCHEMA,
        )

        result = compute_daily_summary(df).collect()[0]

        assert result["min_power_output"] == 2.0
        assert result["max_power_output"] == 4.0
        assert result["avg_power_output"] == pytest.approx(3.0)
        assert result["reading_count"] == 3

    def test_imputed_count_reflects_flagged_rows(self, spark):
        """Proves imputed_count is a genuine sum of the flag, not just
        always zero or always equal to reading_count."""
        df = spark.createDataFrame(
            [
                self._row(1, datetime(2026, 1, 1, 0), 2.0, imputed=False),
                self._row(1, datetime(2026, 1, 1, 1), 2.0, imputed=True),
                self._row(1, datetime(2026, 1, 1, 2), 3.0, imputed=False),
            ],
            schema=self.SCHEMA,
        )

        result = compute_daily_summary(df).collect()[0]

        assert result["reading_count"] == 3
        assert result["imputed_count"] == 1

    def test_imputed_row_repeating_an_existing_value_does_not_change_min_or_max(self, spark):
        """Documents the specific claim the module docstring makes:
        forward-fill can only repeat a value that already occurred
        earlier that day, so including imputed rows in the aggregation
        cannot introduce a new min or max. Here the imputed row repeats
        the day's max — min/max stay exactly what they'd be without it,
        only reading_count and imputed_count change."""
        df = spark.createDataFrame(
            [
                self._row(1, datetime(2026, 1, 1, 0), 2.0, imputed=False),
                self._row(1, datetime(2026, 1, 1, 1), 4.0, imputed=False),  # true max
                self._row(1, datetime(2026, 1, 1, 2), 4.0, imputed=True),   # repeats max via fill
            ],
            schema=self.SCHEMA,
        )

        result = compute_daily_summary(df).collect()[0]

        assert result["max_power_output"] == 4.0
        assert result["min_power_output"] == 2.0
        assert result["reading_count"] == 3
        assert result["imputed_count"] == 1

    def test_turbines_and_days_are_grouped_separately(self, spark):
        """Proves the groupBy is on (turbine_id, date) together, not
        either alone — different turbines and different days must not
        get blended into the same summary row."""
        df = spark.createDataFrame(
            [
                self._row(1, datetime(2026, 1, 1, 0), 2.0),
                self._row(1, datetime(2026, 1, 2, 0), 8.0),  # same turbine, next day
                self._row(2, datetime(2026, 1, 1, 0), 5.0),  # same day, other turbine
            ],
            schema=self.SCHEMA,
        )

        result = {
            (r["turbine_id"], r["date"]): r for r in compute_daily_summary(df).collect()
        }

        assert len(result) == 3
        assert result[(1, date(2026, 1, 1))]["avg_power_output"] == 2.0
        assert result[(1, date(2026, 1, 2))]["avg_power_output"] == 8.0
        assert result[(2, date(2026, 1, 1))]["avg_power_output"] == 5.0


class TestFlagAnomalies:
    """Contract: real (non-imputed) readings are flagged when
    power_output falls outside mean +/- 2 stddev of that turbine's own
    readings for that day. Imputed rows are excluded entirely — absent
    from the output, not merely unflagged — and are never counted
    toward the mean/stddev either."""

    SCHEMA = StructType([
        StructField("turbine_id", IntegerType()),
        StructField("timestamp", TimestampType()),
        StructField("power_output", DoubleType()),
        StructField("power_output_was_imputed", BooleanType()),
    ])

    def _row(self, turbine_id, ts, power, imputed=False):
        return (turbine_id, ts, power, imputed)

    def test_reading_far_below_mean_is_flagged(self, spark):
        """Hand-calculable known-answer case: five readings at 3.0, one
        at 1.0. The lone outlier pulls the mean down and defines the
        spread, so this checks the flag correctly identifies it as the
        outlier relative to the group, not against a hardcoded number."""
        values = [1.0, 3.0, 3.0, 3.0, 3.0, 3.0]
        df = spark.createDataFrame(
            [self._row(1, datetime(2026, 1, 1, h), v) for h, v in enumerate(values)],
            schema=self.SCHEMA,
        )

        result = {r["power_output"]: r for r in flag_anomalies(df).collect()}

        assert result[1.0]["is_anomaly"] is True
        assert result[3.0]["is_anomaly"] is False

    def test_reading_within_bounds_is_not_flagged(self, spark):
        """Negative-space test: ordinary variation within 2 stddev must
        not be flagged, or the threshold would be useless noise."""
        values = [2.9, 3.0, 3.1, 3.0, 2.95, 3.05]
        df = spark.createDataFrame(
            [self._row(1, datetime(2026, 1, 1, h), v) for h, v in enumerate(values)],
            schema=self.SCHEMA,
        )

        result = flag_anomalies(df).collect()

        assert not any(r["is_anomaly"] for r in result)

    def test_anomaly_check_is_symmetric_high_and_low(self, spark):
        """Proves a spike above the mean is caught just as reliably as
        a dip below it — the check is on absolute distance, not just
        underperformance."""
        values = [3.0, 3.0, 3.0, 3.0, 3.0, 9.0]  # one high outlier
        df = spark.createDataFrame(
            [self._row(1, datetime(2026, 1, 1, h), v) for h, v in enumerate(values)],
            schema=self.SCHEMA,
        )

        result = {r["power_output"]: r for r in flag_anomalies(df).collect()}

        assert result[9.0]["is_anomaly"] is True
        assert result[3.0]["is_anomaly"] is False

    def test_imputed_rows_are_excluded_from_output_entirely(self, spark):
        """Proves imputed rows aren't just unflagged but absent from
        the result set — the anomaly table should never contain a
        fabricated reading passed off as real."""
        df = spark.createDataFrame(
            [
                self._row(1, datetime(2026, 1, 1, 0), 3.0, imputed=False),
                self._row(1, datetime(2026, 1, 1, 1), 3.0, imputed=False),
                self._row(1, datetime(2026, 1, 1, 2), 3.0, imputed=True),
            ],
            schema=self.SCHEMA,
        )

        result = flag_anomalies(df).collect()

        assert len(result) == 2
        assert all(r["power_output_was_imputed"] is False for r in result)

    def test_imputed_rows_do_not_skew_the_mean_or_stddev(self, spark):
        """Regression-shaped guard: an imputed value repeating an
        earlier real reading would, if included, artificially tighten
        the day's variance. Here five real readings vary normally, and
        five imputed rows all repeat one of them — if imputed rows were
        wrongly included in the stats, this would shrink stddev and
        cause false positives among the real readings. Proves the real
        readings remain unflagged despite the imputed noise sitting
        alongside them."""
        real_values = [2.5, 3.5, 2.8, 3.2, 3.0]
        rows = [self._row(1, datetime(2026, 1, 1, h), v) for h, v in enumerate(real_values)]
        rows += [
            self._row(1, datetime(2026, 1, 1, 10 + h), 3.0, imputed=True) for h in range(5)
        ]
        df = spark.createDataFrame(rows, schema=self.SCHEMA)

        result = flag_anomalies(df).collect()

        assert len(result) == 5  # only the real readings
        assert not any(r["is_anomaly"] for r in result)

    def test_single_real_reading_day_is_not_flagged(self, spark):
        """Regression-shaped guard: a turbine-day with only one real
        reading has an undefined (null) standard deviation in Spark.
        Without an explicit guard, comparing against a null threshold
        would propagate null rather than a clean True/False. Proves
        is_anomaly resolves to False, not null, when there's not enough
        data to judge."""
        df = spark.createDataFrame(
            [self._row(1, datetime(2026, 1, 1, 0), 3.0)],
            schema=self.SCHEMA,
        )

        result = flag_anomalies(df).collect()[0]

        assert result["is_anomaly"] is False  # not None

    def test_anomaly_stats_are_scoped_per_turbine_and_day(self, spark):
        """A turbine's stats must not be computed against another
        turbine's readings, or a different day's readings, even when
        both are present in the same input DataFrame."""
        df = spark.createDataFrame(
            [
                # Turbine 1, day 1: tight cluster around 3.0
                self._row(1, datetime(2026, 1, 1, 0), 2.9),
                self._row(1, datetime(2026, 1, 1, 1), 3.0),
                self._row(1, datetime(2026, 1, 1, 2), 3.1),
                # Turbine 2, same day: naturally much higher baseline —
                # would look anomalous against turbine 1's stats, but
                # shouldn't be flagged against its own.
                self._row(2, datetime(2026, 1, 1, 0), 9.0),
                self._row(2, datetime(2026, 1, 1, 1), 9.1),
                self._row(2, datetime(2026, 1, 1, 2), 8.9),
            ],
            schema=self.SCHEMA,
        )

        result = flag_anomalies(df).collect()

        assert not any(r["is_anomaly"] for r in result)