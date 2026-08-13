# dab_watt_a_pipeline

Databricks Asset Bundle for `watt-a-pipeline`. See the [project README](../README.md) for
the problem statement, architecture, and design decisions — this document covers setup,
deployment, and day-to-day commands.

## Prerequisites

- [Databricks CLI](https://docs.databricks.com/en/dev-tools/cli/) (0.218.0+)
- [uv](https://docs.astral.sh/uv/)
- A JDK (e.g. `openjdk-17-jdk`) with `JAVA_HOME` set — PySpark needs one locally even for
  offline unit tests, no cluster involved
- A Databricks workspace and an authenticated CLI profile:
  ```bash
  databricks auth login --host <workspace-url>
  # or, if OAuth login isn't reachable in your environment:
  databricks configure --token --profile <profile-name>
  ```

## Setup

```bash
git clone <this-repo>
cd dab_watt_a_pipeline
uv sync
```

## Running tests

```bash
uv run pytest -v
```

30 unit tests covering every Silver and Gold transformation, run entirely offline against a
local PySpark session — no Databricks connection required. See the project README's Testing
Philosophy section for what is and isn't covered this way, and why.

## Deploying

Both `dev` and `prod` targets deploy to the same workspace/profile in this setup — they're
differentiated by Unity Catalog catalog (`raw_dev`/`bronze_dev`/etc. vs `raw`/`bronze`/etc.)
and by bundle `mode`, not by host.

```bash
databricks bundle validate --profile <profile>          # sanity check first
databricks bundle deploy --profile <profile>             # deploys to dev by default
databricks bundle deploy -t prod --profile <profile>     # deploy to prod
```

## Running the pipeline

```bash
databricks bundle run turbine_bronze --profile <profile>
```

This runs the full Bronze → Silver → Gold pipeline in one go (all three layers are defined
as datasets within the same pipeline). Progress and any data-quality warnings
(`report_remaining_nulls`) are visible in the pipeline's event log in the workspace UI.

## Uploading data

Raw CSVs land in the Bronze landing volume, picked up automatically by Autoloader on the
next pipeline run via Catalog Explorer in the workspace UI: navigate to the volume → Upload files.

## Repo structure

```
dab_watt_a_pipeline/
├── databricks.yml                    # bundle config: variables, targets, resources
├── src/turbine_data/
│   ├── extract_raw.py                # Bronze: Autoloader ingestion
│   ├── functions.py                  # Silver + Gold: all pure transform functions
│   ├── silver_turbine_cleaning.py    # Silver: thin materialized_view wrapper
│   └── gold_turbine_aggregates.py    # Gold: thin materialized_view wrappers
├── tests/
│   ├── conftest.py                   # local SparkSession fixture
│   └── test_functions.py             # unit tests for functions.py
└── pyproject.toml
```

`functions.py` holds all the actual logic (cleaning, imputation, daily stats, anomaly
detection) as pure, independently-testable functions. The pipeline source files
(`extract_raw.py`, `silver_turbine_cleaning.py`, `gold_turbine_aggregates.py`) are
deliberately thin — they wire Databricks-specific I/O (Autoloader reads, catalog/schema
targeting via `@dp.materialized_view`) around calls into `functions.py`, so the business
logic can be tested without a Databricks connection at all.