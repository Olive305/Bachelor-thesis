# Using code from https://link.springer.com/10.1007/979-8-8688-0008-5 

import tensorflow.keras as keras
from keras import optimizers
from keras import losses
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Dropout, Embedding, LSTM
from keras.optimizers import RMSprop, Adam, Nadam
from keras.preprocessing import sequence
from keras.callbacks import TensorBoard
from keras import regularizers

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report

import seaborn as sns
import pandas as pd
import numpy as np
import duckdb
import os
import matplotlib

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import tensorflow
import sys

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
        df = con.execute(f"""
            SELECT 
            "stream:timestamp",
            "concept:name",
            "stream:system" || '_' || "stream:observation" AS "sensor:id",
            "stream:value"
            FROM read_parquet('{file_location}')
            WHERE "org:resource" = '{resource}'
            AND "stream:observation" NOT LIKE '%NFC%'
            GROUP BY "stream:observation", "stream:system", "stream:value", "stream:timestamp", "concept:name"
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
        
        # For each timestamp in wide, find the closest timestamp in sensor_data
        sensor_values = []
        sensor_timestamps = sensor_data["stream:timestamp"].values
        sensor_values_list = sensor_data["stream:value"].values
        
        if prints:
            print(f"Searching closest timestamps for {len(wide)} entries...")
        
        for ts in wide["stream:timestamp"]:
            # Find index of closest timestamp
            pos = sensor_data["stream:timestamp"].searchsorted(ts)

            if pos == 0:
                sensor_values.append(sensor_values_list[0])
            elif pos >= len(sensor_timestamps):
                sensor_values.append(sensor_values_list[-1])
            else:
                # TODO change to correctly use interpolation instead of just using the closest value
                left_ts = pd.Timestamp(sensor_timestamps[pos - 1])
                right_ts = pd.Timestamp(sensor_timestamps[pos])
                # choose the closer timestamp
                if abs(ts - left_ts) <= abs(right_ts - ts):
                    sensor_values.append(sensor_values_list[pos - 1])
                else:
                    sensor_values.append(sensor_values_list[pos])
                    
        if prints:
            print(f"  Merged {len(sensor_values)} values for sensor {sensor_id}")
        
        wide[sensor_id] = sensor_values
        
    if prints:
        print("Initialized 'wide' preview:")
        print(wide.head(10))
        print("Initialized 'wide' shape:", wide.shape)
        
    # Some of the sensors have 3 dimensional values given as 3 values in a list. Convert them to individual columns in the dataframe
    for col in wide.columns:
        if col not in ["stream:timestamp", "event"]:
            # Check if column contains list values
            if wide[col].apply(lambda x: isinstance(x, list)).any():
                # Split list values into separate columns
                split_data = pd.DataFrame(wide[col].tolist(), index=wide.index)
                split_data.columns = [f"{col}_{i}" for i in range(len(split_data.columns))]
                wide = pd.concat([wide.drop(col, axis=1), split_data], axis=1)
        
    # Apply one hot encoding on categorial columns
    num_cols, _, categorical_cols = detect_column_types(wide)
    if prints:
        print("\nCategorial Cols:\n")
        print(categorical_cols)
        print("\n")
        print("\nNumerical Cols:\n")
        print(num_cols)
        print("\n")
    wide = pd.get_dummies(wide, columns=categorical_cols, drop_first=True)
    
    if prints:
        print("Removed 3 dimensional entries from the df and applied one hot encoding on categorial columns")
        num_nan = df.isna().sum().sum()
        print(f"Number of nan values in the df: {num_nan}")
        print("Finnished 'wide' preview:")
        print(wide.head(10))
        print("Finished 'wide' shape:", wide.shape)
        print("\n\n")
        
    # Split the dataframe into train, validation, and test sets
    train, test = train_test_split(wide, test_size=0.2, random_state=42)

    if prints:
        print(f"Training set size: {len(train)}")
        print(f"Test set size: {len(test)}\n")
        
    train = train.drop(columns=["stream:timestamp"])
    test = test.drop(columns=["stream:timestamp"])
    
    column_names = train.columns
        
    # Scale training set and apply scaling on test and validation set
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train)
    test_scaled = scaler.transform(test)

    return train_scaled, test_scaled, column_names

def train_autoencoder(train_df: pd.DataFrame, test_df: pd.DataFrame, column_names, prints: bool = False):
    """
    Train the autoencoder on the given data
    
    Args: 
        train_df: Dataframe containing the training set
        test_df: Dataframe containing the test set
        
    Returns:
        prints: If True, it prints further information into console
        
    Raises:

    """
    
    logfilename = "AE"
    
    input_dim = train_df.shape[1]
    encoding_dim = int(input_dim / 3)

    inputArray = Input(shape=(input_dim,))
    encoded = Dense(encoding_dim, activation='relu')(inputArray)
    decoded = Dense(input_dim, activation=None)(encoded)

    autoencoder = Model(inputArray, decoded)

        
    autoencoder.compile(optimizer=RMSprop(), loss='mean_squared_error', metrics=['mae', 'mse'])
    
    batch_size = 32
    epochs = 8
    
    history = autoencoder.fit(train_df, train_df, batch_size=batch_size, epochs=epochs, verbose=1, shuffle=True, validation_data=(test_df, test_df), callbacks=[TensorBoard(log_dir='./logs/{0}'.format(logfilename))])
    
    test_score = autoencoder.evaluate(test_df, test_df, verbose=1)
    print('Test loss: ', test_score[0])
    print('Test accuracy: ', test_score[1])
    
    print("\n\n")
    # Select random samples from test set and compare input vs output
    num_samples = 5
    random_indices = np.random.choice(test_df.shape[0], num_samples, replace=False)

    predictions = autoencoder.predict(test_df[random_indices])

    print("Input vs Output Comparison:")
    print("=" * 19)
    for i, idx in enumerate(random_indices):
        input_sample = test_df[idx]
        output_sample = predictions[i]
        
        print(f"Sample {i+1}:")
        print(f"{'Column':<150}{'Input Value':<20}{'Predicted Value':<20}")
        print("=" * 190)
        
        for col, input_val, output_val in zip(range(input_sample.shape[0]), input_sample, output_sample):
            print(f"{column_names[col]:<150}{input_val:<20}{output_val:<20}")
        
        print("-" * 190)
    
if __name__ == "__main__":
    df = read_data("ov_1", prints=True)
    train_scaled, test_scaled, column_names = prepare_data(df, prints=True)
    train_autoencoder(train_scaled, test_scaled, column_names, prints=True)