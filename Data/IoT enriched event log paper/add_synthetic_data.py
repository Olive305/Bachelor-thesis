import duckdb
import os
import pandas as pd
import random


# Directory with parquet files
parquet_directory = os.path.join(os.getcwd(), "Data", "IoT enriched event log paper", "20130794", "Cleaned Event Log", "parquet")
input_file = os.path.join(parquet_directory, "all_combined_new.parquet")


# DuckDB connection
con = duckdb.connect()

# Check if parquet directory and file exist
print(f"Parquet directory exists: {os.path.exists(parquet_directory)}")
print(f"Input parquet file exists: {os.path.exists(input_file)}")

def get_temp_sensor_values():
    # Create a df with all the values of temperature sensors sorted by their timestamp
    df = con.execute(f"""
        SELECT 
            "stream:observation",
            "stream:system",
            "stream:value",
            "stream:timestamp"
        FROM read_parquet('{input_file}')
        WHERE "org:resource" IN ('ov_1', 'mm_1', 'sm_1', 'wt_1', 'vgr_1')
        AND "stream:procedure_type" = 'stream:continuous'
        AND "stream:observation" LIKE '%Temperature%'
        GROUP BY "org:resource", "stream:observation", "stream:system", "stream:value", "stream:timestamp"
        ORDER BY "stream:timestamp" ASC
    """).df()
    # ensure timestamp column is actual datetime objects for safe comparison with datetime.datetime
    df['stream:timestamp'] = pd.to_datetime(df['stream:timestamp'])
    return df
    
def get_smallest_temperature_value():
    result = con.execute(f"""
        SELECT 
            "stream:value",
        FROM read_parquet('{input_file}')
        WHERE "org:resource" IN ('ov_1', 'mm_1', 'sm_1', 'wt_1', 'vgr_1')
        AND "stream:procedure_type" = 'stream:continuous'
        AND "stream:observation" LIKE '%Temperature%'
        GROUP BY "org:resource", "stream:observation", "stream:system", "stream:value", "stream:timestamp"
        ORDER BY "stream:value" ASC
        LIMIT 1
    """).df()
    
    return result['stream:value'].iloc[0]

def get_largest_temperature_value():
    result = con.execute(f"""
        SELECT 
            "stream:value",
        FROM read_parquet('{input_file}')
        WHERE "org:resource" IN ('ov_1', 'mm_1', 'sm_1', 'wt_1', 'vgr_1')
        AND "stream:procedure_type" = 'stream:continuous'
        AND "stream:observation" LIKE '%Temperature%'
        GROUP BY "org:resource", "stream:observation", "stream:system", "stream:value", "stream:timestamp"
        ORDER BY "stream:value" DESC
        LIMIT 1
    """).df()
    
    return result['stream:value'].iloc[0]

# global variables
temp_sensor_values = get_temp_sensor_values()
min_temp_value = get_smallest_temperature_value()
max_temp_value = get_largest_temperature_value()

def synthetic_pressure_per_timestamp(timestamp, pressure):
    # Synthetic value = pressure * temperature_distance 

    # Get last recorded temperature value before or at the given timestamp
    # If that timestamp is before the first temperature reading, use a random temperature value between min_temp_value and max_temp_value (this one would be an anomaly)
    try:
        temp_value = float(
            temp_sensor_values[
                temp_sensor_values['stream:timestamp'] <= timestamp
            ]['stream:value'].iloc[-1]
        )
    except Exception:
        # pick a random temperature within [min_temp_value, max_temp_value]
        temp_value = random.uniform(float(min_temp_value), float(max_temp_value))
        
    # With 5% probability add a random value for temp_value to create anomalies (distribution does not need to be considered since these are anomalies)
    # TODO make it such that random value is far away from the actual value
    if random.random() < 0.05:
        # pick a random temperature within [min_temp_value, max_temp_value]
        temp_value = random.uniform(float(min_temp_value), float(max_temp_value))
    
    # Calculate the synthetic value as difference of temperature to min_temp_value multiplied by pressure
    temperature_distance = float(temp_value) - float(min_temp_value) + 1  # +1 to avoid zero multiplication
    synthetic_value = float(pressure) * float(temperature_distance)
    
    return synthetic_value


# Register the function in DuckDB
con.create_function("value_per_timestamp", synthetic_pressure_per_timestamp, return_type="DOUBLE")

output_file = os.path.join(parquet_directory, "all_combined_with_synthetic.parquet")

# --- Write result directly to Parquet WITHOUT loading all data ---
con.execute(f"""
    COPY (
        WITH base AS (
            SELECT * FROM read_parquet('{input_file}')
        ),

        new_rows AS (
            SELECT
                "trace:SubProcessID",
                "concept:name",
                "org:resource",
                "time:timestamp",
                "operation_end_time",
                "stream:datastream",
                "stream:system",
                "stream:system_type",
                "stream:observation",
                "stream:procedure_type",
                "stream:interaction_type",
                "stream:timestamp",
                value_per_timestamp(CAST("stream:timestamp" AS TIMESTAMP), "stream:value") AS "stream:value",
                "sensor_key"
            FROM base
            WHERE "org:resource" IN ('ov_1', 'mm_1', 'sm_1', 'wt_1', 'vgr_1')
              AND "stream:procedure_type" = 'stream:continuous'
              AND "stream:observation" LIKE '%Pressure%'
        ),

        preserved AS (
            SELECT * FROM base
            WHERE NOT (
                "org:resource" IN ('ov_1', 'mm_1', 'sm_1', 'wt_1', 'vgr_1')
                AND "stream:procedure_type" = 'stream:continuous'
                AND "stream:observation" LIKE '%Pressure%'
            )
        ),

        final AS (
            SELECT * FROM preserved
            UNION ALL
            SELECT * FROM new_rows
        )

        SELECT * FROM final
        ORDER BY "stream:timestamp"
    )
    TO '{output_file}' (FORMAT PARQUET);
""")

print("Done!")
