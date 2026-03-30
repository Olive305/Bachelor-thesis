from Data.pytorch_autoencoder import create_AE
from Data.autoencoder_data_preparation import read_and_prepare_data
import torch
import torch.nn as nn
import numpy as np
import os
from tabulate import tabulate
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM


RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def detect_anomalies(train_scaled, test_scaled, val_anomalous, scaling, column_names, resource, train_model: bool = False, redo_hyperparameter_tuning:bool = False, prints: bool = False, threshold_percentile: float = 99, val_scaled=None):
    
    # Train the model, if training is required or if the file doesnt exist
    if train_model or not os.path.exists(os.path.join("model", resource + "_autoencoder.pth")):
        if prints:
            print("Training autoencoder model" + (" since model file hasn't been found...\n" if not os.path.exists(os.path.join("model", resource + "autoencoder.pth")) else "...\n"))
        if val_scaled is None:
            val_scaled = test_scaled
        model_location = create_AE(train_scaled, test_scaled, val_scaled, scaling, column_names, resource, tune_hyperparameters=redo_hyperparameter_tuning)
        
    # Load the trained model
    model_location = os.path.join("model", resource + "_autoencoder.pth")
    model = torch.load(model_location, weights_only=False, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    
    # Perform anomaly detection on the modified validation set
    if prints:
        print("Performing anomaly detection on the modified validation set...\n")
        
    # Set device for anomaly detection and set other parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    train_tensor = torch.tensor(train_scaled, dtype=torch.float32).to(device)

    # Set the threshold as a percentile of the per-sample training losses
    with torch.no_grad():
        train_reconstructed = model(train_tensor)
        train_losses = torch.mean((train_reconstructed - train_tensor) ** 2, dim=1).cpu().numpy()
    threshold = np.percentile(train_losses, threshold_percentile)
    
    val_anomalous_tensor = torch.tensor(val_anomalous, dtype=torch.float32).to(device)

    # Calculate the loss on the modified validation set per sample
    model.eval()
    with torch.no_grad():
        reconstructed = model(val_anomalous_tensor)
        losses = torch.mean((reconstructed - val_anomalous_tensor) ** 2, dim=1).cpu().numpy()  # Calculate per-sample loss

    # Detect anomalies and return their indices
    detected_anomaly_indices = np.where(losses > threshold)[0]
    
    return detected_anomaly_indices, reconstructed
    
    
    # Show example anomalous data rows and their output
    # Not used in final code, only for manual validation

    random_indices = np.random.choice(detected_anomaly_indices, min(5, len(detected_anomaly_indices)), replace=False)

    for idx, row_idx in enumerate(random_indices):
        anomalous_data = val_anomalous[row_idx]
        is_anomaly = row_idx in anomaly_indices
        loss = losses[row_idx]
        
        # Reverse the scaling
        anomalous_data_reversed = scaling.inverse_transform([anomalous_data])[0]
        
        # Get model output
        anomalous_tensor = torch.tensor([anomalous_data], dtype=torch.float32).to(device)
        with torch.no_grad():
            anomalous_output_tensor = model(anomalous_tensor)
            anomalous_output = anomalous_output_tensor.cpu().numpy()[0]
        
        anomalous_output_reversed = scaling.inverse_transform([anomalous_output])[0]
        
        # Determine if detected as anomaly
        detected = loss > threshold
        
        print(f"\nRow {idx+1}:")
        print(f"True Anomaly: {is_anomaly}, Detected as Anomaly: {detected}, Loss: {loss:.8f}")
        
        table_data = [[col, f"{anomalous_data_reversed[i]:.4f}", f"{anomalous_output_reversed[i]:.4f}"] 
                      for i, col in enumerate(column_names)]
        print(tabulate(table_data, headers=["Feature", "Initial Unscaled", "Reconstructed Unscaled"], tablefmt="grid"))
    
def add_synthetic_anomalies(val_scaled, train_scaled, column_names):
    # Select 5% of the rows of the validation set, which should be changed to create an anomaly
    
    val_synthetic = val_scaled.copy()
    
    num_rows = len(val_scaled)
    num_anomalies = int(np.ceil(num_rows * 0.05))
    anomaly_indices = np.random.choice(num_rows, num_anomalies, replace=False)
    
    # Exclude the event rows and the elapsed second since start value from the creation of anomalies
    change_columns = [i for i, col in enumerate(column_names) if ("event" not in col and "lapsed" not in col)]
    
    # Calculate standard deviation for each column in train_scaled
    std_devs = np.std(train_scaled, axis=0)
    
    changed_columns = {}
    
    for i in anomaly_indices:
        num_changed_columns = np.random.choice([1, 2, 3])
        selected_columns = np.random.choice(change_columns, num_changed_columns, replace=False)
        
        changed_columns[i] = selected_columns
        
        for col in selected_columns:
            value = val_scaled[i, col]
            border_up = value + np.random.normal(0, std_devs[col] * 3)
            border_down = value - np.random.normal(0, std_devs[col] * 3)
            
            suitable_values = train_scaled[
                (train_scaled[:, col] >= border_down) & 
                (train_scaled[:, col] <= border_up),
                col
            ]
            if len(suitable_values):
                val_synthetic[i, col] = suitable_values[np.random.randint(len(suitable_values))]
            else:
                val_synthetic[i, col] = train_scaled[np.argmax(np.abs(train_scaled[:, col] - value)), col]

    return val_synthetic, changed_columns

def detect_using_isolation_forest(train_scaled, test_scaled, val_anomalous, scaling, column_names, prints: bool = False):
        
    # Train Isolation Forest on training data
    iso_forest = IsolationForest(contamination=0.01, random_state=42)
    iso_forest.fit(train_scaled)
    
    # Get anomaly scores and predictions
    val_scores = iso_forest.decision_function(val_anomalous)
    anomalies = iso_forest.predict(val_anomalous) == -1
    detected_anomaly_indices = np.where(anomalies)[0]
    
    return detected_anomaly_indices
    
    
def detect_using_one_class_support_vector_machine(train_scaled, test_scaled, val_anomalous, scaling, column_names, prints: bool = False):
        
    # Train One-Class SVM on training data
    oc_svm = OneClassSVM(kernel='rbf', gamma='auto', nu=0.1)
    oc_svm.fit(train_scaled)
        
    # Get predictions
    anomalies = oc_svm.predict(val_anomalous) == -1
    detected_anomaly_indices = np.where(anomalies)[0]
        
    return detected_anomaly_indices

if __name__ == "__main__":
    train_scaled, test_scaled, val_scaled, scaling, column_names = read_and_prepare_data("hbw_1", prints=False)
    val_anomalous, anomaly_positions = add_synthetic_anomalies(val_scaled, train_scaled, column_names)
    ae_anomalies = detect_anomalies(train_scaled, test_scaled, val_anomalous, scaling, column_names, "hbw_1", train_model=False, val_scaled=val_scaled)
    if_anomalies = detect_using_isolation_forest(train_scaled, test_scaled, val_anomalous, scaling, column_names)
    oc_svm_anomalies = detect_using_one_class_support_vector_machine(train_scaled, test_scaled, val_anomalous, scaling, column_names)