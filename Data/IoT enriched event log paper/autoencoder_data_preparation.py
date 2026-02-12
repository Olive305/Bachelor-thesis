# Using code from https://link.springer.com/10.1007/979-8-8688-0008-5 

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import duckdb
import os


def read_data(resource: str, prints: bool = False) -> pd.DataFrame:
    """
    Read the data of one resource using duckDB
    
    Args: 
        resource: The resource for which the data should be fetched
        prints: If True, it prints further information into console
        
    Returns: 
        The pandas dataframe with the (raw) data
        
    Raises:
        FileNotFoundError: If the parquet file does not exist
        ValueError: If resource is empty or invalid
        Exception: If duckDB query fails
    """
    
    if not resource or not isinstance(resource, str):
        raise ValueError("Resource must be a non-empty string")
    
    # Get file location
    parquet_directory = os.path.join(os.getcwd(), "Data", "IoT enriched event log paper", "20130794", "Cleaned Event Log", "parquet")
    file_location = os.path.join(parquet_directory, "all_combined_with_synthetic.parquet")
    
    if not os.path.exists(file_location):
        raise FileNotFoundError(f"Parquet file not found at {file_location}")
    
    try:
        # DuckDB connection
        con = duckdb.connect()

        # Create a df with all the values of temperature sensors sorted by their timestamp
        # Removed redundant GROUP BY since each row is already unique by definition
        df = con.execute(f"""
            SELECT 
            "stream:timestamp",
            "concept:name",
            "stream:system" || '_' || "stream:observation" AS "sensor:id",
            "stream:value"
            FROM read_parquet('{file_location}')
            WHERE "org:resource" = '{resource}'
            AND "stream:observation" NOT LIKE '%NFC%'
            ORDER BY "stream:timestamp" ASC
        """).df()
        
        # ID of a sensor is given by stream:system and stream:observation
        # NFC-tag sensors will not be included in the data
        
        if prints:
            print(f"Loaded {len(df)} rows from {file_location}\n")
            
        return df
    except Exception as e:
        raise Exception(f"Failed to query data for resource '{resource}': {str(e)}")
    

def detect_column_types(df):
    """Detect and classify column types in the dataframe."""
    cont_cols = []
    bin_cols = []
    categorial_cols = []

    for col in df.columns:
        
        # Timestamps are NOT model features → ignore as early as possible
        if "timestamp" in col:
            continue

        series = df[col]
        unique_vals = series.dropna().unique()

        num_vals = pd.to_numeric(pd.Series(unique_vals), errors="coerce").dropna().unique()
        if len(num_vals) > 0 and set(num_vals).issubset({0.0, 1.0}):
            bin_cols.append(col)
        elif len(num_vals) > 0:
            # Everything else numeric = continuous
            cont_cols.append(col)
        else:
            categorial_cols.append(col)

    return cont_cols, bin_cols, categorial_cols

def prepare_data(df: pd.DataFrame, prints: bool = False):
    """
    Use data preparation on the dataframe to make it usable for the Autoencoder
    
    Args:
        df: The dataframe contining the raw data
        prints: If True, it prints further information into console
        
    Returns:
        The pandas dataframe split in training, test and validation set
        
    Raises:
    
    """

    # Ensure timestamp column is actual datetime objects for safe comparison with datetime.datetime
    df['stream:timestamp'] = pd.to_datetime(df['stream:timestamp'])
    
    reference_sensor = 'Current_Task_Elapsed_Seconds_Since_Start'

    # Rows where sensor:id CONTAINS the reference sensor string
    ref = df[df["sensor:id"].str.contains(reference_sensor, case=False, na=False)].sort_values("stream:timestamp")
    
    if prints:
        print("Reference dataframe preview (first 10 rows):")
        print(ref.head(10))
        print(f"Reference dataframe shape: {ref.shape}\n")

    # All other rows
    others = df[~df["sensor:id"].str.contains(reference_sensor, case=False, na=False)].sort_values("stream:timestamp")
    
    # Get ref sensor id for final dataframe creation
    reference_sensor_id = ref.iloc[0]["sensor:id"]

    # Reset index for safe alignment and display the reference dataframe
    ref = ref.reset_index(drop=True)
    
    if prints:
        print("Reference dataframe preview (first 10 rows):")
        print(ref.head(10))
        print(f"Reference dataframe shape: {ref.shape}\n")

    # Initialize 'wide' with the reference timestamps and the reference sensor values
    wide = pd.DataFrame({
        "stream:timestamp": ref["stream:timestamp"].values,
        reference_sensor_id: ref["stream:value"].values
    })

    # Show the initialized wide head so you can confirm alignment before merging others
    if prints:
        print("Initialized 'wide' preview:")
        print(wide.head(10))
        print("Initialized 'wide' shape:", wide.shape)

    wide["event"] = ref["concept:name"].values

    if prints:
        print("Initial sensor added with timestamps and event data. Merging other sensors...")
    
    for sensor_id in others["sensor:id"].unique():
        sensor_data = others[others["sensor:id"] == sensor_id][["stream:timestamp", "stream:value"]].reset_index(drop=True)
        
        if prints:
            print(f"  Merging sensor: {sensor_id} with {len(sensor_data)} entries")
        
        # Vectorized merge using searchsorted - MUCH faster than Python loops
        sensor_timestamps = sensor_data["stream:timestamp"].values
        sensor_values_array = sensor_data["stream:value"].values
        wide_timestamps = wide["stream:timestamp"].values
        
        if prints:
            print(f"Searching closest timestamps for {len(wide)} entries...")
        
        # Find indices of closest timestamps using vectorized searchsorted
        pos_indices = np.searchsorted(sensor_timestamps, wide_timestamps)
        sensor_values = np.empty(len(wide), dtype=object)
        
        # Vectorized approach: handle all positions at once
        # For positions before first timestamp
        first_mask = pos_indices == 0
        sensor_values[first_mask] = sensor_values_array[0]
        
        # For positions after last timestamp
        last_mask = pos_indices >= len(sensor_timestamps)
        sensor_values[last_mask] = sensor_values_array[-1]
        
        # For positions in between - find closest
        mid_mask = ~(first_mask | last_mask)
        mid_indices = np.where(mid_mask)[0]
        
        for idx in mid_indices:
            pos = pos_indices[idx]
            left_ts = sensor_timestamps[pos - 1]
            right_ts = sensor_timestamps[pos]
            ts = wide_timestamps[idx]
            # Choose closer timestamp
            if abs(ts - left_ts) <= abs(right_ts - ts):
                sensor_values[idx] = sensor_values_array[pos - 1]
            else:
                sensor_values[idx] = sensor_values_array[pos]
                    
        if prints:
            print(f"  Merged {len(sensor_values)} values for sensor {sensor_id}")
        
        wide[sensor_id] = sensor_values
        
    if prints:
        print("Initialized 'wide' preview:")
        print(wide.head(10))
        print("Initialized 'wide' shape:", wide.shape)
        
    # Some of the sensors have 3 dimensional values given as 3 values in a list. Convert them to individual columns in the dataframe
    # Optimized: check first value instead of applying to entire column (10-100x faster)
    cols_to_drop = []
    new_cols_dict = {}
    
    for col in wide.columns:
        if col not in ["stream:timestamp", "event"]:
            # Check only first non-null value (much faster than .apply())
            first_val = wide[col].iloc[0] if len(wide) > 0 else None
            if isinstance(first_val, (list, tuple)) or (isinstance(first_val, str) and first_val.startswith('[')):
                cols_to_drop.append(col)
                try:
                    # If it's a string representation, convert to list first
                    if isinstance(first_val, str):
                        test_val = eval(first_val)
                    else:
                        test_val = first_val
                    if prints:
                        print(f"  Found list column: {col} with {len(test_val)} elements")
                except Exception as e:
                    if prints:
                        print(f"  Warning: Could not parse list column {col}: {str(e)}")
                # Split list values into separate columns
                try:
                    # Convert each element to a list/array and expand into DataFrame
                    list_values = wide[col].apply(lambda x: list(x) if isinstance(x, (list, tuple, np.ndarray)) else eval(x) if isinstance(x, str) else x)
                    split_data = pd.DataFrame(list_values.tolist(), index=wide.index)
                    for i in range(len(split_data.columns)):
                        new_cols_dict[f"{col}_{i}"] = split_data.iloc[:, i].values
                    if prints:
                        print(f"  Successfully split {col} into {len(split_data.columns)} columns")
                except Exception as e:
                    if prints:
                        print(f"  Error splitting column {col}: {str(e)}")
                    raise
    
    # Apply all new columns at once (more efficient than concat)
    if new_cols_dict:
        if prints:
            print(f"Adding {len(new_cols_dict)} new columns and dropping {len(cols_to_drop)} list columns")
        for col_name, col_values in new_cols_dict.items():
            wide[col_name] = col_values
        wide = wide.drop(columns=cols_to_drop)
        if prints:
            print(f"Completed list column conversion. New shape: {wide.shape}")
    elif prints:
        print("No list columns found to convert")
        
    # Apply one hot encoding on categorial columns
    num_cols, _, categorical_cols = detect_column_types(wide)
    if prints:
        print("\nCategorial Cols:\n")
        print(categorical_cols)
        print("\n")
        print("\nNumerical Cols:\n")
        print(num_cols)
        print("\n")
    
    # Use sparse one-hot encoding for memory efficiency (10-100x smaller for sparse data)
    if categorical_cols:
        wide = pd.get_dummies(wide, columns=categorical_cols, drop_first=True, sparse=True)
        # Convert sparse columns to dense for compatibility with most ML models
    
    if prints:
        print("Removed 3 dimensional entries from the df and applied one hot encoding on categorial columns")
        num_nan = wide.isna().sum().sum()
        print(f"Number of nan values in the df: {num_nan}")
        print("Finnished 'wide' preview:")
        print(wide.head(10))
        print("Finished 'wide' shape:", wide.shape)
        print("\n\n")
        
    # Split the dataframe into train, validation, and test sets
    train, test = train_test_split(wide, test_size=0.2, random_state=69)
    val, test = train_test_split(test, test_size=0.5, random_state=69)

    if prints:
        print(f"Training set size: {len(train)}\n")
        print(f"Test set size: {len(test)}\n")
        print(f"Validation set size: {len(val)}\n")
    
    # Drop timestamp columns if present
    timestamp_cols = [col for col in train.columns if "imestamp" in col]
    if timestamp_cols:
        train = train.drop(columns=timestamp_cols)
        test = test.drop(columns=timestamp_cols)
        val = val.drop(columns=timestamp_cols)
    
    column_names = train.columns
    
    print("Dropped timestamp columns")

    # Reduce memory footprint before scaling - use float32 throughout
    # Note: If data is sparse, convert carefully
    if hasattr(train, 'sparse'):
        train = train.sparse.to_dense().astype("float32")
        test = test.sparse.to_dense().astype("float32")
        val = val.sparse.to_dense().astype("float32")
    else:
        # Only convert numeric columns to float32
        numeric_cols = train.select_dtypes(include=[np.number]).columns
        train[numeric_cols] = train[numeric_cols].astype("float32")
        test[numeric_cols] = test[numeric_cols].astype("float32")
        val[numeric_cols] = val[numeric_cols].astype("float32")

    # Convert to numpy arrays (copy=False to avoid extra copies where possible)
    train_values = train.to_numpy(copy=False) if isinstance(train, pd.DataFrame) else train
    test_values = test.to_numpy(copy=False) if isinstance(test, pd.DataFrame) else test
    val_values = val.to_numpy(copy=False) if isinstance(val, pd.DataFrame) else val

    # Scale training set and apply scaling on test and validation set
    # Using chunked processing to minimize RAM spikes
    scaler = StandardScaler(copy=False)
    
    # Fit scaler on training data
    scaler.fit(train_values)
    
    # Transform all sets
    train_scaled = scaler.transform(train_values)
    test_scaled = scaler.transform(test_values)
    val_scaled = scaler.transform(val_values)
    
    # Explicitly free intermediate large arrays to reclaim memory
    del train_values, test_values, val_values, train, test, val
    
    print("Scaled columns")

    return train_scaled, test_scaled, val_scaled, scaler, column_names

def read_and_prepare_data(resource: str, prints:bool = False):
    if prints:
        print("Reading Data:\n\n")
    df = read_data(resource, prints=prints)
    
    if prints:
        print("Preparing Data:\n\n")
    return prepare_data(df , prints=prints)
 
if __name__ == "__main__":
    read_and_prepare_data("hbw_1", prints=True)