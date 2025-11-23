import duckdb
import os
import pandas as pd


# Directory with parquet files
parquet_directory = os.path.join(os.getcwd(), "Data", "IoT enriched event log paper", "20130794", "Cleaned Event Log", "parquet")
input_file = os.path.join(parquet_directory, "all_combined_new.parquet")

# Check if parquet directory and file exist
print(f"Parquet directory exists: {os.path.exists(parquet_directory)}")
print(f"Input parquet file exists: {os.path.exists(input_file)}")
def define_values():
    new_value_df = con.execute(f"""
        SELECT DISTINCT
            "stream:value",
            "stream:timestamp",
            "stream:observation",
            "trace:SubProcessID",
            "concept:name"
        FROM read_parquet('{input_file}')
        WHERE "stream:observation" = 'http://iot.uni-trier.de/FTOnto#OV_1_WT_1_Temperature'
        AND "org:resource" = 'ov_1'
        ORDER BY "stream:timestamp"
    """).df()
    
    # Convert timestamps to pandas datetime objects so arithmetic (timestamp - last_timestamp) works
    new_value_df["stream:timestamp"] = pd.to_datetime(new_value_df["stream:timestamp"])
    
    new_value_df["valve"] = None
    new_value_df["temp"] = None
    
    # Fill new columns
    for i in range(len(new_value_df)):
        if i == 0:
            last_valve_state = None
            last_oven_temp = None
            last_timestamp = None
        else:
            last_valve_state = new_value_df.at[i-1, "valve"]
            last_oven_temp = new_value_df.at[i-1, "temp"]
            last_timestamp = new_value_df.at[i-1, "stream:timestamp"]
        
        result = valve(
            new_value_df.at[i, "stream:value"],
            new_value_df.at[i, "concept:name"],
            new_value_df.at[i, "stream:timestamp"],
            new_value_df.at[i, "trace:SubProcessID"],
            last_valve_state,
            last_oven_temp,
            last_timestamp
        )
        
        new_value_df.at[i, "valve"] = result[0]
        new_value_df.at[i, "temp"] = result[1]
    
    return new_value_df
    return new_value_df

# Returns valve sensor value but temperature value is stored in last_oven_temp
def valve(room_temp, event, timestamp, subprocessID, last_valve_state, last_oven_temp, last_timestamp):
    
    # Ensure room_temp is float
    room_temp = float(room_temp)
    
    # Check for each subprocess, if it is burning or tempering
    subprocess_burn_or_temper_df = con.execute(f"""
        SELECT DISTINCT
            "concept:name",
            "trace:SubProcessID",
            CASE
                WHEN "concept:name" LIKE 'temper%' THEN 150
                WHEN "concept:name" LIKE 'burning%' THEN 200
            END AS temp
        FROM read_parquet('{input_file}')
        WHERE "concept:name" LIKE 'burning%' OR "concept:name" LIKE 'temper%'
    """).df()
    
    temp_to_reach = subprocess_burn_or_temper_df.loc[
        subprocess_burn_or_temper_df["trace:SubProcessID"] == subprocessID, "temp"
    ].values[0]
    
    # First instance of calling this
    if last_oven_temp is None:
        last_oven_temp = room_temp
        last_timestamp = timestamp
        last_valve_state = 0.0
        
    # Calculate time difference in seconds
    time_diff = (timestamp - last_timestamp).total_seconds()
    
    # If event is "transporting the workpiece to the inside of the oven"
    if event == "transporting the workpiece to the inside of the oven":
        if last_oven_temp < temp_to_reach:
            last_oven_temp = last_oven_temp + time_diff * (110 * last_valve_state - 0.3 * (last_oven_temp - room_temp))
            last_valve_state = 1.0
            
        else:
            # Caclulate how much the valve needs to be opened to stay at temp_to_reach degrees
            last_valve_state = ((temp_to_reach - last_oven_temp) / time_diff + 0.3 * (last_oven_temp - room_temp)) / 110
            if last_valve_state > 1.0:
                last_valve_state = 1.0
            elif last_valve_state < 0.0:
                last_valve_state = 0.0
            last_oven_temp = last_oven_temp + time_diff * (110 * last_valve_state - 0.3 * (last_oven_temp - room_temp))
            
    elif isinstance(event, str) and event.strip().lower().startswith("burning"):
        # Caclulate how much the valve needs to be opened to stay at temp_to_reach degrees
        last_valve_state = ((temp_to_reach - last_oven_temp) / time_diff + 0.3 * (last_oven_temp - room_temp)) / 110
        if last_valve_state > 1.0:
            last_valve_state = 1.0
        elif last_valve_state < 0.0:
            last_valve_state = 0.0
        last_oven_temp = last_oven_temp + time_diff * (110 * last_valve_state - 0.3 * (last_oven_temp - room_temp))
    
    else:
        # For other events, keep the valve closed and calculate temperature drop
        last_valve_state = 0.0
        last_oven_temp = last_oven_temp + time_diff * (110 * last_valve_state - 0.3 * (last_oven_temp - room_temp))

    last_timestamp = timestamp
    return [last_valve_state, last_oven_temp]


# DuckDB connection
con = duckdb.connect()


values_df = define_values()

def value_per_timestamp(timestamp, temp: bool):
    # if temp, then return the temp value, else return the valve value
    
    if temp:
        matched_row = values_df[values_df["stream:timestamp"] == timestamp]
        if not matched_row.empty:
            return matched_row.iloc[0]["temp"]
        else:
            return None
    
    else:
        matched_row = values_df[values_df["stream:timestamp"] == timestamp]
        if not matched_row.empty:
            return matched_row.iloc[0]["valve"]
        else:
            return None


con.create_function(
    "value_per_timestamp",
    value_per_timestamp,
    return_type="DOUBLE",
    parameters=["TIMESTAMP", "BOOLEAN"]
)

output_file = os.path.join(parquet_directory, "all_combined_with_synthetic.parquet")

# --- Write result directly to Parquet WITHOUT loading all data ---
con.execute(f"""
    COPY (
        WITH base AS (
            SELECT * FROM read_parquet('{input_file}')
            ORDER BY "stream:timestamp"
        ),
        
        valve_rows AS (
            SELECT
                "trace:SubProcessID",
                "concept:name",
                "org:resource",
                "time:timestamp",
                "operation_end_time",
                "stream:datastream",
                'http://iot.uni-trier.de/FTOnto#OV_1_Synthetic' AS "stream:system",
                "stream:system_type",
                'http://iot.uni-trier.de/FTOnto#Valve_State' AS "stream:observation",
                "stream:procedure_type",
                "stream:interaction_type",
                "stream:timestamp",
                value_per_timestamp(CAST("stream:timestamp" AS TIMESTAMP), FALSE) AS "stream:value",
                "sensor_key"    
            FROM base
            WHERE "stream:observation" = 'http://iot.uni-trier.de/FTOnto#OV_1_WT_1_Temperature'
            AND "org:resource" = 'ov_1'
        ),
        
        temp_rows AS (
            SELECT
                "trace:SubProcessID",
                "concept:name",
                "org:resource",
                "time:timestamp",
                "operation_end_time",
                "stream:datastream",
                'http://iot.uni-trier.de/FTOnto#OV_1_Synthetic' AS "stream:system",
                "stream:system_type",
                'http://iot.uni-trier.de/FTOnto#Oven_Temp' AS "stream:observation",
                "stream:procedure_type",
                "stream:interaction_type",
                "stream:timestamp",
                value_per_timestamp(CAST("stream:timestamp" AS TIMESTAMP), TRUE) AS "stream:value",
                "sensor_key"    
            FROM base
            WHERE "stream:observation" = 'http://iot.uni-trier.de/FTOnto#OV_1_WT_1_Temperature'
            AND "org:resource" = 'ov_1'
        )
        SELECT * FROM base
        UNION ALL
        SELECT * FROM valve_rows
        UNION ALL
        SELECT * FROM temp_rows
    )
    TO '{output_file}' (FORMAT PARQUET);
""")

print("Done!")
