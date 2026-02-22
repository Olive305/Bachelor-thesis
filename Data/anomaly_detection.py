from pytorch_autoencoder import create_AE
from autoencoder_data_preparation import read_and_prepare_data
import torch
import torch.nn as nn
import numpy as np
import os
from tabulate import tabulate
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM


def detect_anomalies(train_scaled, test_scaled, val_scaled, scaling, column_names, resource, train_model: bool = False, redo_hyperparameter_tuning:bool = False):
    
    if train_model or not os.path.exists(os.path.join("model", "autoencoder.pth")):
        model_location = create_AE(train_scaled, test_scaled, val_scaled, scaling, column_names, resource, tune_hyperparameters=redo_hyperparameter_tuning)
        
    model_location = os.path.join("model", resource + "_autoencoder.pth")
    model = torch.load(model_location, weights_only=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    train_tensor = torch.tensor(train_scaled, dtype=torch.float32).to(device)

    criterion = nn.MSELoss()

    with torch.no_grad():
        train_reconstructed = model(train_tensor)
        train_losses = criterion(train_reconstructed, train_tensor).cpu().numpy()

    treshold = np.percentile(train_losses, 99.9)  # Set threshold as the 99.9th percentile of the training losses
    print("\n Treshold: ", treshold, "\n\n")
    
    #! Metrics calculation maybe false
    val_anomalous, anomaly_indices = add_synthetic_anomalies(val_scaled, train_scaled, column_names)

    val_anomalous_tensor = torch.tensor(val_anomalous, dtype=torch.float32).to(device)

    model.eval()
    with torch.no_grad():
        reconstructed = model(val_anomalous_tensor)
        losses = torch.mean((reconstructed - val_anomalous_tensor) ** 2, dim=1).cpu().numpy()  # Calculate per-sample loss

    anomalies = losses > treshold
    detected_anomaly_indices = np.where(anomalies)[0]
    print(f"Detected {np.sum(anomalies)} anomalies out of {len(val_anomalous)} samples")
    
    # Calculate metrics
    true_positives = np.sum(np.isin(detected_anomaly_indices, anomaly_indices))
    false_positives = np.sum(~np.isin(detected_anomaly_indices, anomaly_indices))
    false_negatives = len(anomaly_indices) - true_positives
    true_negatives = (
    len(val_anomalous)
    - true_positives
    - false_positives
    - false_negatives
    )

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / len(anomaly_indices) if len(anomaly_indices) > 0 else 0
    anomaly_percentage = (len(anomaly_indices) / len(val_anomalous)) * 100
    
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    f2_score = 5 * (precision * recall) / (4 * precision + recall) if (4 * precision + recall) > 0 else 0

    print(f"Precision: {precision:.4f}")
    print(f"True Positives: {true_positives}")
    print(f"False Positives: {false_positives}") 
    print(f"False Negatives: {false_negatives}")
    print(f"True Negatives: {true_negatives}")
    print(f"Recall: {recall:.4f}")
    print(f"Anomaly Percentage in Data: {anomaly_percentage:.2f}%")
    print(f"F1 Score: {f1_score:.4f}")
    print(f"F2 Score: {f2_score:.4f}")
    
    return precision, recall, f1_score, f2_score
    
    # Show example anomalous data rows and their output (skipped)

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
        detected = loss > treshold
        
        print(f"\nRow {idx+1}:")
        print(f"True Anomaly: {is_anomaly}, Detected as Anomaly: {detected}, Loss: {loss:.8f}")
        
        table_data = [[col, f"{anomalous_data_reversed[i]:.4f}", f"{anomalous_output_reversed[i]:.4f}"] 
                      for i, col in enumerate(column_names)]
        print(tabulate(table_data, headers=["Feature", "Initial Unscaled", "Reconstructed Unscaled"], tablefmt="grid"))
    
def add_synthetic_anomalies(val_scaled, train_scaled, column_names):
    num_rows = len(val_scaled)
    num_anomalies = int(np.ceil(num_rows * 0.5))
    anomaly_indices = np.random.choice(num_rows, num_anomalies, replace=False)
    
    non_event_indices = [i for i, col in enumerate(column_names) if "event" not in col]

    for idx in anomaly_indices:
        unchanged_indices = non_event_indices.copy()
        while unchanged_indices:
            # Modify anomaly
            col_idx = np.random.choice(unchanged_indices)
            original_value = val_scaled[idx, col_idx]
            col_min, col_max = train_scaled[:, col_idx].min().item(), train_scaled[:, col_idx].max().item()
            anomalous_value = col_max if original_value < (col_min + col_max) / 2 else col_min
            val_scaled[idx, col_idx] = anomalous_value
            unchanged_indices.remove(col_idx)
            
            # with 5% probability change another column
            if np.random.random() < 0.1:
                break

    return val_scaled, anomaly_indices

def detect_using_isolation_forest(train_scaled, test_scaled, val_scaled, scaling, column_names):
        
    # Train Isolation Forest on training data
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    iso_forest.fit(train_scaled)
    
    # Add synthetic anomalies to validation data
    val_anomalous, anomaly_indices = add_synthetic_anomalies(val_scaled, train_scaled, column_names)
    
    # Get anomaly scores and predictions
    val_scores = iso_forest.decision_function(val_anomalous)
    anomalies = iso_forest.predict(val_anomalous) == -1
    detected_anomaly_indices = np.where(anomalies)[0]
    
    print(f"Detected {np.sum(anomalies)} anomalies out of {len(val_anomalous)} samples")
    
    # Calculate metrics
    true_positives = np.sum(np.isin(detected_anomaly_indices, anomaly_indices))
    false_positives = np.sum(~np.isin(detected_anomaly_indices, anomaly_indices))
    false_negatives = len(anomaly_indices) - true_positives
    true_negatives = len(val_anomalous) - true_positives - false_positives - false_negatives
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / len(anomaly_indices) if len(anomaly_indices) > 0 else 0
    anomaly_percentage = (len(anomaly_indices) / len(val_anomalous)) * 100
    
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    f2_score = 5 * (precision * recall) / (4 * precision + recall) if (4 * precision + recall) > 0 else 0
    
    print(f"Precision: {precision:.4f}")
    print(f"True Positives: {true_positives}")
    print(f"False Positives: {false_positives}")
    print(f"False Negatives: {false_negatives}")
    print(f"True Negatives: {true_negatives}")
    print(f"Recall: {recall:.4f}")
    print(f"Anomaly Percentage in Data: {anomaly_percentage:.2f}%")
    print(f"F1 Score: {f1_score:.4f}")
    print(f"F2 Score: {f2_score:.4f}")
    
    return precision, recall, f1_score, f2_score
    
    
def detect_using_one_class_support_vector_machine(train_scaled, test_scaled, val_scaled, scaling, column_names):
        
    # Train One-Class SVM on training data
    oc_svm = OneClassSVM(kernel='rbf', gamma='auto', nu=0.1)
    oc_svm.fit(train_scaled)
        
    # Add synthetic anomalies to validation data
    val_anomalous, anomaly_indices = add_synthetic_anomalies(val_scaled, train_scaled, column_names)
        
    # Get predictions
    anomalies = oc_svm.predict(val_anomalous) == -1
    detected_anomaly_indices = np.where(anomalies)[0]
        
    print(f"Detected {np.sum(anomalies)} anomalies out of {len(val_anomalous)} samples")
        
    # Calculate metrics
    true_positives = np.sum(np.isin(detected_anomaly_indices, anomaly_indices))
    false_positives = np.sum(~np.isin(detected_anomaly_indices, anomaly_indices))
    false_negatives = len(anomaly_indices) - true_positives
    true_negatives = len(val_anomalous) - true_positives - false_positives - false_negatives
        
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / len(anomaly_indices) if len(anomaly_indices) > 0 else 0
    anomaly_percentage = (len(anomaly_indices) / len(val_anomalous)) * 100
    
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    f2_score = 5 * (precision * recall) / (4 * precision + recall) if (4 * precision + recall) > 0 else 0
        
    print(f"Precision: {precision:.4f}")
    print(f"True Positives: {true_positives}")
    print(f"False Positives: {false_positives}")
    print(f"False Negatives: {false_negatives}")
    print(f"True Negatives: {true_negatives}")
    print(f"Recall: {recall:.4f}")
    print(f"Anomaly Percentage in Data: {anomaly_percentage:.2f}%")
    print(f"F1 Score: {f1_score:.4f}")
    print(f"F2 Score: {f2_score:.4f}")
    
    return precision, recall, f1_score, f2_score

if __name__ == "__main__":
    train_scaled, test_scaled, val_scaled, scaling, column_names = read_and_prepare_data("hbw_1", prints=False)
    detect_anomalies(train_scaled, test_scaled, val_scaled, scaling, column_names, "hbw_1", train_model=False)
    print("\n\n")
    detect_using_isolation_forest(train_scaled, test_scaled, val_scaled, scaling, column_names)
    detect_using_one_class_support_vector_machine(train_scaled, test_scaled, val_scaled, scaling, column_names)