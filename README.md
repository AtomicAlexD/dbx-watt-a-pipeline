# watt-a-pipeline ⚡

A Medallion-architecture (Bronze/Silver/Gold) data pipeline for cleaning, validating, and
analyzing wind turbine sensor data — built with PySpark and Lakeflow Declarative Pipelines
on Databricks, deployed via Databricks Asset Bundles.

For setup, deployment, and day-to-day commands, see
[`dab_watt_a_pipeline/README.md`](dab_watt_a_pipeline/README.md). This document covers the
problem, the design, and the reasoning behind it.

## Problem

A wind farm of turbines reports sensor data (timestamp, wind speed, wind direction, power
output) as CSVs, appended daily, five turbines per file, with a particular turbine's data
always landing in the same file. The upstream system occasionally misses entries due to
sensor malfunctions. The pipeline needs to:

1. Clean the raw data — handle missing values and outliers.
2. Calculate per-turbine summary statistics (min/max/avg power output) over a 24-hour period.
3. Identify turbines whose output deviates more than 2 standard deviations from expected.
4. Store the cleaned data and summary statistics for further analysis.

## Architecture

```
Landing Volume (CSV)
        │  Autoloader (cloudFiles)
        ▼
   Bronze  (bronze_turbine_readings)      — raw, typed, unmodified
        │  dedupe → flag/nullify invalid → impute
        ▼
   Silver  (silver_turbine_readings)      — cleaned, every fix auditable
        │  daily aggregation           anomaly detection (excludes imputed)
        ▼                                       ▼
   Gold: daily_turbine_summary          Gold: turbine_anomalies
```

Each layer is a separate Unity Catalog catalog (`raw_<env>`, `bronze_<env>`, `silver_<env>`,
`gold_<env>`), all bundle-managed (schemas and volumes defined as Databricks Asset Bundle
resources, not created by hand). `<env>` is `dev` or empty for `prod`, controlled by a bundle
variable, so both environments deploy from the same `databricks.yml` with no duplicated config.

**Bronze** ingests via Autoloader with an explicit schema (not inferred — see Design Decisions)
and `cloudFiles.rescuedDataColumn` so anything unexpected is captured rather than dropped.

**Silver** is a `@dp.materialized_view`, not a streaming table. This is deliberate: the
cleaning logic uses window functions (`row_number` for dedup, unbounded forward-fill for
imputation) that require a full partition's data at once, which Structured Streaming's
incremental micro-batch model doesn't support. A materialized view fully recomputes on each
pipeline run — the right semantics for this logic, at this data volume.

**Gold** is two materialized views reading Silver as a batch source: `daily_turbine_summary`
(per-turbine, per-day min/max/avg) and `turbine_anomalies` (row-level, real readings only,
flagged where they fall outside 2σ of that turbine-day's own mean).

## Design decisions and assumptions

These are the calls made where the brief was ambiguous or silent, and why:

- **"Missing entries" means null values within rows that exist, not whole absent rows.**
  The provided sample data contains neither (every turbine has exactly 744 complete hourly
  rows across the full month, zero nulls, zero timestamp gaps), so this is a stated
  interpretation rather than something the data itself could confirm. Detecting genuinely
  absent rows would require generating an expected timestamp grid per turbine and
  left-joining onto it — treated as out of scope.

- **Daily file updates are handled with `cloudFiles.allowOverwrites: true`.** The brief
  doesn't specify whether "updated with the last 24 hours" means a new file lands each day
  or existing files are overwritten in place. Without this option, Autoloader's default
  behavior (never re-read a previously-seen file path) would silently drop overwritten data
  entirely — a correctness risk, not just an efficiency one. With it on, the pipeline is
  correct under either interpretation; Silver's dedup on `(turbine_id, timestamp)` absorbs
  any resulting reprocessing at negligible cost given the data volume.

- **Late-arriving data is handled for free by full reprocessing, but that's a different
  problem from permanently missing data.** Because Silver recomputes from all of Bronze on
  each run, a reading that arrives late (rather than never) gets naturally incorporated,
  with no watermarking or special handling needed. This does not help with data that's
  never sent at all (e.g. a sensor broken since install) — see the leading-null limitation
  below.

- **Invalid values are physical impossibilities, not statistical outliers.** `power_output
  < 0`, `wind_speed < 0` or `> 100` (a defensive sanity ceiling — genuine wind doesn't reach
  that speed), and `wind_direction` outside `[0, 360)` are treated as data-quality issues at
  Silver. Statistical anomalies (>2σ) are a Gold-layer judgment about otherwise-valid data,
  kept deliberately separate.

- **No upper bound on `power_output`.** This would require each turbine's rated capacity,
  which isn't provided by the brief or the data. Documented as a gap rather than an invented
  threshold — the natural next validation to add given that data.

- **Missing/invalid values are imputed (forward-filled per turbine, ordered by time), not
  dropped**, to preserve time-series continuity — and every fabricated value is flagged
  (`{col}_was_imputed`, `{col}_was_missing`, `{col}_was_invalid`), not just silently
  overwritten. This keeps cleaning auditable and makes "which sensors are flaky" a
  queryable question.

- **Known, accepted limitation: a turbine's very first reading, if null, stays null.**
  Forward-fill has nothing to look back to. This is treated as a legitimate outcome (a
  sensor broken since install genuinely has no signal to recover) rather than a bug —
  `report_remaining_nulls` logs a warning rather than halting the pipeline, since some
  gaps are real and expected, not a processing failure.

- **Gold's daily summary includes imputed values; anomaly detection excludes them.**
  Forward-fill can only ever repeat a value that already occurred earlier that day, so it
  cannot introduce a new min or max — the only effect on the daily summary is a slight pull
  on the average, which is why imputed rows stay in (with `imputed_count` surfaced
  alongside, so a heavily-fabricated day is visible rather than hidden). Anomaly detection
  excludes imputed rows entirely, from both the mean/stddev calculation and from candidacy
  — including them would let a run of repeated fabricated values artificially shrink that
  day's variance, making genuinely unusual real readings look more extreme than they should.

- **"24-hour period" is interpreted as calendar date**, per the brief's own example. Each
  turbine-day is self-contained — stats and anomaly thresholds are computed fresh per day,
  with no baseline carried across days.

- **Data quality checks use Lakeflow's built-in mechanisms, not DQX.** Databricks Labs' DQX
  framework was considered for more structured, reusable quality rules, but wasn't needed
  here — the checks required are simple enough that plain PySpark functions (tested,
  documented, and reused via `.transform()`) are the lighter-weight and more transparent
  choice for this scope.

## Testing philosophy

Two tiers, deliberately:

- **Unit tests** (`tests/test_functions.py`, 30 tests) cover every Silver and Gold
  transformation — all pure functions (DataFrame in, DataFrame out), tested offline against
  a local PySpark session with hand-built fixtures. No Databricks connection needed. Several
  are explicitly labeled regression tests, guarding real bugs found during development
  (e.g. Spark ordering NaN as greater than any value for plain comparisons, not just
  sorting, which originally misclassified a NaN reading as "invalid" instead of "missing").
- **Bronze's Autoloader ingestion is not unit tested**, deliberately — `cloudFiles` is a
  Databricks-only streaming source with no local equivalent. It's verified instead via a
  real dev deployment and pipeline run, which is the appropriate integration check for
  that piece.

See [`dab_watt_a_pipeline/README.md`](dab_watt_a_pipeline/README.md) for exact commands.

## Scalability considerations

At the current data volume (15 turbines, ~11k rows/month) none of the following are
necessary for correctness — they're documented here as the awareness points that would
matter at real scale, in line with what's actually configured versus what would be the
next step:

- **Triggered, not continuous, pipeline mode** — data genuinely arrives once a day, so a
  continuously-running stream would be pure overhead (and, on a Free Edition workspace,
  unnecessary compute quota usage).
- **Partition by date, not by turbine_id**, on Silver/Gold tables — at only 15 turbines,
  partitioning by turbine_id as well would create excessive tiny partitions; date is the
  natural query-scoping dimension as turbine count grows.
- **Small-file compaction** (`delta.autoOptimize.optimizeWrite` / `autoCompact`) on Bronze,
  to avoid accumulating many tiny files from repeated small daily writes.
- **Explicit broadcast** on the anomaly-stats join in `flag_anomalies` — the per-turbine-day
  stats side of the join is tiny, and Spark's adaptive execution would likely broadcast it
  automatically, but an explicit `F.broadcast()` makes that guaranteed rather than
  AQE-dependent.

## Known limitations / future work

- No detection of wholly-missing rows (only null values within existing rows) — see the
  scope assumption above.
- No upper bound on `power_output` — would need each turbine's rated capacity.
- A turbine's unfillable leading null (broken since install) stays null indefinitely; a
  gap-length cap on forward-fill, or a fallback data source, would be the next step if this
  needs closing.
- A small dashboard reading Silver's audit flags (`_was_missing`/`_was_invalid` roll-ups
  per turbine) to surface "which turbines are flaky" — a natural extension of data already
  being tracked, not yet built.

# Development notes

Fully aware i've focused way too much on dab's and testing instead of the pipeline itself but I feel like the pipeline itself is bread and butter autoloader with declarative pipelines. Might mess up somewhere in that but at least i'm proving I can create a pipeline that will last and scale.

Could have done tests on databricks (Feb release?) or could have used databricks connect, plenty of ways to skin a cat but decided to go with local, just got to make sure you have java installed and pyspark setup with uv or it doesn't work.

First real data note was on the whole when data arrives. New data every 24 hours is ambiguous, new files or updates to files, I think from the wording it's updates to files so i'll fully scan for changes in the files. Once we confirm exact delivery we can tune accordingly. Might have missed a thing here but it is what it is, just an option on the autoloader.

Also worth mentioning I would have loved to implement some kind of data quality testing with DQX and ODCS data contracts here but way overkill for this, just something i've been doign recently which could have been handy to demonstrate.

The whole imputed stuff was probably the most complicated bit, felt like I got stuck trying to fix scenarios like data being null at the start and whether I should backwards fill but i'm dealing wtih a scenario of a broken sensor from the start. The data begins when the data starts flowing from the sensor and only if I miss the odd bit should I impute to keep the time series in place.

Also gone down a rabbit hole of trying to figure out if the missing data is actually fully missing rows or just nulls. This will define how my imputed data is constructed.

Hold up, did some analysis and every turbine has exactly 744 rows, one per hour, for the entire 31-day span (2022-03-01 to 2022-03-31), zero gaps in the timestamp sequence i'm losing my marbles haha.

I'm going to implement based on the row arriving as a null for now, not going to do both because time constraints but this will demonstrate i've thought about it.

Okay going a little overboard on the testing but i've added a new file with specifically broken data for turbine 42 so we don't just have unit tests we'll have actually data we can flow through the dev pipeline to show how it reacts to bad data. Will throw it in the repo as well.

Went with a materialised view for the silver because i'm only getting data every day, could have left it as a streaming table but a single refresh once a day should be simple and cheaper.

Nice so the data flows through, one example is some wind direction data that was bad, we mark it as null them impute it and mark the row with a flag.
Something like this
```sql
select * from select * from silver_dev.dev_atedimmock_turbine_data.silver_turbine_readings where turbine_id=42 and wind_direction_was_invalid = true
```

Side note I hate putting this in the silver catalog and calling it silver_ but i'll come back to it, might forget to fix this later. We'll find out.

Back to fighting with the imputable data, i'm going to exclude it from any major calculations like the finding of anomalies. We have a continuous time series on the silver table which is great.

I'll include them in the daily min max avg for now but make it clear through a column how many rows have been imputed.

Query for checking the stddev data on the anomolies:
```sql
select * from gold_dev.dev_atedimmock_turbine_data.turbine_anomalies
where is_anomaly = true
```

Quick query to see the imputed data count on the daily summary:
```sql
select * from gold_dev.dev_atedimmock_turbine_data.daily_turbine_summary where imputed_count>0
```

Not gunna lie, was a tad lazy on test generation, the main thing is they work and they cover all my examples, also with the test sheet I can load that helps me visualise issues. Feel like that's my approval gate, running some SQL against the results and having a gander.

Last minute addition to the silver mat view, decided to show how I would partition or add additional properties for auto compact etc. It's not an after thought but the data volumes right now are so small it's hard to prove how much they would help.