from pyspark import pipelines as dp
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, TimestampType, IntegerType, DoubleType
)

# Explicit schema here avoids Autoloader inference guessing different types but I have got the rescued stuff later on.
TURBINE_SCHEMA = StructType([
    StructField("timestamp", TimestampType(), True),
    StructField("turbine_id", IntegerType(), True),
    StructField("wind_speed", DoubleType(), True),
    StructField("wind_direction", IntegerType(), True),
    StructField("power_output", DoubleType(), True),
])


@dp.table(
    name="bronze_turbine_readings",
    comment="Raw turbine sensor readings ingested via Autoloader, unchanged except for ingestion metadata.",
)
def bronze_turbine_readings():
    landing_path = spark.conf.get("turbine_pipeline.landing_volume_path")

    return (
        spark.readStream.format("cloudFiles")
        .option("cloudFiles.format", "csv")
        .option("header", "true")
        .schema(TURBINE_SCHEMA)
        # Specifically added this just in case some wierd data comes in, don't want to lose it and might be useful for debugging later.
        .option("cloudFiles.rescuedDataColumn", "_rescued_data")
        # Sneaky bit in the spec where it talks about new data arriving every 24 hours, doesn't specify new files or not so i'm going to check every file for updates anyway. Would remove this with clarification.
        .option("cloudFiles.allowOverwrites", "true")
        .load(landing_path)
        .withColumn("_ingested_at", F.current_timestamp())
        .withColumn("_source_file", F.col("_metadata.file_path"))
    )