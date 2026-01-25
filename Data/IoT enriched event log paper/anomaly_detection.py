from pytorch_autoencoder import create_AE
from autoencoder_data_preparation import read_and_prepare_data
import torch
import numpy as np

def detect_anomalies(resource: str, train_model: bool = True):
    train_scaled, test_scaled, val_scaled, scaling, column_names = read_and_prepare_data("ov_1", prints=False)
    
    # TODO implement behaviour to just load model if it has been already trained
    model_location = create_AE(train_scaled, test_scaled, val_scaled, scaling, column_names)
    
    model = torch.load(model_location)
    
    # TODO define the threshold for the loss, from where on it is considered as an anomaly
    treshold = 0
    
    # TODO get the data with synthetic anomalies, perform anomaly detection on it and then display valuable information like precision, false negatives, percentage of anomalies in data
    
    
def add_synthetic_anomalies(val_scaled, column_names):
    num_rows = len(val_scaled)
    num_anomalies = int(np.ceil(num_rows * 0.05))
    anomaly_indices = np.random.choice(num_rows, num_anomalies, replace=False)
    
    non_event_indices = [i for i, col in enumerate(column_names) if "event" not in col]

    for idx in anomaly_indices:
        unchanged_indices = non_event_indices.copy()
        while unchanged_indices:
            # Modify anomaly
            col_idx = np.random.choice(unchanged_indices)
            val_scaled[idx, col_idx] = 0 # TODO use the correct anomaly value here
            unchanged_indices.remove(col_idx)
            
            # with 5% probability change another column
            if np.random.random() < 0.05:
                break

    return val_scaled