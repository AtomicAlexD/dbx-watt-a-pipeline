# dbx-watt-a-pipeline
A Medallion-architecture data pipeline for cleaning, validating, and analyzing wind turbine sensor data, built with PySpark for Databricks. ⚡

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