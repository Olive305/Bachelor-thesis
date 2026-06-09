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


def _labels_from_anomalies(n_rows, anomalous_values):
    labels = np.zeros(int(n_rows), dtype=int)
    if not anomalous_values:
        return labels
    for row_id in anomalous_values.keys():
        try:
            idx = int(row_id)
        except (TypeError, ValueError):
            continue
        if 0 <= idx < n_rows:
            labels[idx] = 1
    return labels


def detect_anomalies(
    train_scaled,
    val_scaled,
    test_anomalous,
    scaling,
    column_names,
    resource,
    train_model: bool = False,
    redo_hyperparameter_tuning: bool = False,
    prints: bool = False,
    threshold_percentile: float = 99.5,
    test_scaled=None,
    anomalous_values=None,
    optimize_percentile: bool = False,
    percentile_candidates=None,
    return_percentile: bool = False,
):
    
    # Train the model, if training is required or if the file doesnt exist
    if train_model or not os.path.exists(os.path.join("model", resource + "_autoencoder.pth")):
        if prints:
            print("Training autoencoder model" + (" since model file hasn't been found...\n" if not os.path.exists(os.path.join("model", resource + "autoencoder.pth")) else "...\n"))
        if test_scaled is None:
            test_scaled = val_scaled
        model_location = create_AE(train_scaled, val_scaled, test_scaled, scaling, column_names, resource, tune_hyperparameters=redo_hyperparameter_tuning)
        
    # Load the trained model
    model_location = os.path.join("model", resource + "_autoencoder.pth")
    model = torch.load(model_location, weights_only=False, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    
    # Perform anomaly detection on the modified test set
    if prints:
        print("Performing anomaly detection on the modified test set...\n")
        
    # Set device for anomaly detection and set other parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    train_tensor = torch.tensor(train_scaled, dtype=torch.float32).to(device)

    # Compute per sample training losses for threshold selection.
    with torch.no_grad():
        train_reconstructed = model(train_tensor)
        train_losses = torch.mean((train_reconstructed - train_tensor) ** 2, dim=1).cpu().numpy()
    
    test_anomalous_tensor = torch.tensor(test_anomalous, dtype=torch.float32).to(device)

    # Calculate the loss on the modified test set per sample.
    model.eval()
    with torch.no_grad():
        reconstructed = model(test_anomalous_tensor)
        losses = torch.mean((reconstructed - test_anomalous_tensor) ** 2, dim=1).cpu().numpy()  # Calculate per-sample loss

    # Threshold optimization via percentile
    used_percentile = threshold_percentile
    if optimize_percentile:
        if anomalous_values is None:
            if prints:
                print("Warning: optimize_percentile=True but anomalous_values is None; using default threshold_percentile.")
            threshold = np.percentile(train_losses, threshold_percentile)
        else:
            if percentile_candidates is None:
                percentile_candidates = np.linspace(90.0, 99.9, 20)

            labels = _labels_from_anomalies(len(losses), anomalous_values)
            total_anomalies = int(labels.sum())
            if total_anomalies == 0:
                if prints:
                    print("Warning: No anomalies provided for percentile optimization; using default threshold_percentile.")
                threshold = np.percentile(train_losses, threshold_percentile)
            else:
                best = {
                    "f1": -1.0,
                    "recall": -1.0,
                    "precision": -1.0,
                    "percentile": None,
                    "threshold": None,
                }
                for percentile in percentile_candidates:
                    percentile = float(percentile)
                    candidate_threshold = np.percentile(train_losses, percentile)
                    preds = losses > candidate_threshold
                    tp = int(labels[preds].sum())
                    fp = int(preds.sum()) - tp
                    fn = total_anomalies - tp

                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                    f1 = (
                        2 * precision * recall / (precision + recall)
                        if (precision + recall) > 0
                        else 0.0
                    )

                    if (
                        f1 > best["f1"]
                        or (f1 == best["f1"] and recall > best["recall"])
                        or (f1 == best["f1"] and recall == best["recall"] and precision > best["precision"])
                        or (
                            f1 == best["f1"]
                            and recall == best["recall"]
                            and precision == best["precision"]
                            and (best["percentile"] is None or percentile < best["percentile"])
                        )
                    ):
                        best.update(
                            {
                                "f1": f1,
                                "recall": recall,
                                "precision": precision,
                                "percentile": percentile,
                                "threshold": candidate_threshold,
                            }
                        )

                used_percentile = best["percentile"]
                threshold = best["threshold"]

                if prints:
                    print(
                        "Auto-selected percentile "
                        f"{used_percentile:.3f} (F1={best['f1']:.3f}, "
                        f"P={best['precision']:.3f}, R={best['recall']:.3f})"
                    )
    else:
        threshold = np.percentile(train_losses, threshold_percentile)

    # Detect anomalies and return their indices
    detected_anomaly_indices = np.where(losses > threshold)[0]
    
    if return_percentile:
        return detected_anomaly_indices, reconstructed, threshold, float(used_percentile)
    return detected_anomaly_indices, reconstructed, threshold
    
def add_synthetic_anomalies(test_scaled, train_scaled, val_scaled, column_names):
    # Select 5% of the rows of the testidation set, which should be changed to create an anomaly
    
    test_synthetic = test_scaled.copy()
    
    num_rows = len(test_scaled)
    num_anomalies = int(np.ceil(num_rows * 0.025))
    anomaly_indices = np.random.choice(num_rows, num_anomalies, replace=False)
    
    # Divide anomalies into two types
    num_context = int(np.ceil(len(anomaly_indices) / 2))

    context_indices = anomaly_indices[:num_context]
    data_point_indices = anomaly_indices[num_context:]
    
    changed_columns = {}
    
    # Context Anomalies
    test_synthetic, changed_columns = _add_context_anomalies(test_synthetic, test_scaled, train_scaled, val_scaled, column_names, context_indices, changed_columns)
    
    # Data-Point Anomalies
    test_synthetic, changed_columns = _add_data_point_anomalies(test_synthetic, test_scaled, train_scaled, val_scaled, column_names, data_point_indices, changed_columns)
    
    # Track anomaly types
    anomaly_types = {}
    for idx in context_indices:
        anomaly_types[idx] = "context"
    for idx in data_point_indices:
        anomaly_types[idx] = "data_point"
    
    return test_synthetic, changed_columns, anomaly_types


def _add_context_anomalies(test_synthetic, test_scaled, train_scaled, val_scaled, column_names, anomaly_indices, changed_columns):
    # Exclude the event rows and the elapsed second since start value from the creation of anomalies
    change_columns = [i for i, col in enumerate(column_names) if ("event" not in col and "lapsed" not in col)]
    
    # Calculate standard deviation for each column in train_scaled
    std_devs = np.std(train_scaled, axis=0)
    
    for i in anomaly_indices:
        num_changed_columns = np.random.choice([1, 2, 3])
        # Ensure we don't try to select more columns than available
        num_changed_columns = min(num_changed_columns, len(change_columns))
        selected_columns = np.random.choice(change_columns, num_changed_columns, replace=False)
        
        all_scaled = np.vstack([train_scaled, test_scaled, val_scaled])
        
        changed_columns[i] = selected_columns
        
        for col in selected_columns:
            value = test_scaled[i, col]
            offset = np.abs(np.random.normal(0, std_devs[col] * 1.5))  # always positive

            # Correctly define outer bounds: values far from the original
            border_up = value + offset
            border_down = value - offset
            
            suitable_values = all_scaled[
                (all_scaled[:, col] >= border_up) | 
                (all_scaled[:, col] <= border_down),
                col
            ]

            # Always exclude the exact original value
            suitable_values = suitable_values[suitable_values != value]

            if len(suitable_values):
                test_synthetic[i, col] = suitable_values[np.random.randint(len(suitable_values))]
            else:
                # Fallback: pick the most distant value, ensuring it differs
                candidate = all_scaled[np.argmax(np.abs(all_scaled[:, col] - value)), col]
                if candidate != value:
                    test_synthetic[i, col] = candidate
                # If truly no different value exists, leave unchanged (as per spec)
    
    return test_synthetic, changed_columns


def _add_data_point_anomalies(test_synthetic, test_scaled, train_scaled, val_scaled, column_names, anomaly_indices, changed_columns):
    # Exclude the event rows and elapsed second since start value from the creation of anomalies
    # Exclude event/lapsed columns and columns that are binary (exactly 2 unique values)
    combined_data = np.vstack([train_scaled, test_scaled, val_scaled])
    binary_columns = {
        i for i in range(combined_data.shape[1])
        if np.unique(combined_data[:, i]).size == 2
    }
    change_columns = [
        i for i, col in enumerate(column_names)
        if ("event" not in col and "lapsed" not in col and i not in binary_columns)
    ]
    
    # Calculate min and max for each column across all data (train, val, and test)
    all_scaled = np.vstack([train_scaled, test_scaled, val_scaled])
    mins = np.min(all_scaled, axis=0)
    maxs = np.max(all_scaled, axis=0)
    
    for i in anomaly_indices:
        num_changed_columns = np.random.choice([1, 2, 3])
        # Ensure we don't try to select more columns than available
        num_changed_columns = min(num_changed_columns, len(change_columns))
        selected_columns = np.random.choice(change_columns, num_changed_columns, replace=False)
        
        changed_columns[i] = selected_columns
        
        for col in selected_columns:
            col_min = mins[col]
            col_max = maxs[col]
            col_range = col_max - col_min
            
            # Fallback offset for zero-range (constant) columns to still produce an anomaly
            effective_offset = col_range * 0.5 if col_range > 0 else 1.0

            # Sample uniformly outside the data range
            if np.random.rand() > 0.5:
                # Sample above max
                test_synthetic[i, col] = np.random.uniform(col_max, col_max + effective_offset)
            else:
                # Sample below min
                test_synthetic[i, col] = np.random.uniform(col_min - effective_offset, col_min)
    
    return test_synthetic, changed_columns

def detect_using_isolation_forest(train_scaled, val_scaled, test_anomalous, scaling, column_names, prints: bool = False):
        
    # Train Isolation Forest on training data
    iso_forest = IsolationForest(contamination=0.15,
                                 n_estimators=200,
                                 max_samples='auto',
                                 random_state=42)
    iso_forest.fit(train_scaled)
    
    # Get anomaly scores and predictions
    test_scores = iso_forest.decision_function(test_anomalous)
    anomalies = iso_forest.predict(test_anomalous) == -1
    detected_anomaly_indices = np.where(anomalies)[0]
    
    return detected_anomaly_indices
    
    
def detect_using_one_class_support_vector_machine(train_scaled, val_scaled, test_anomalous, scaling, column_names, prints: bool = False):
        
    # Train One-Class SVM on training data
    oc_svm = OneClassSVM(kernel='rbf', gamma='scale', nu=0.01)
    oc_svm.fit(train_scaled)
        
    # Get predictions
    anomalies = oc_svm.predict(test_anomalous) == -1
    detected_anomaly_indices = np.where(anomalies)[0]
        
    return detected_anomaly_indices

if __name__ == "__main__":
    train_scaled, val_scaled, test_scaled, scaling, column_names, bin_cols = read_and_prepare_data("hbw_1", prints=False)
    test_anomalous, anomalous_values, anomaly_types = add_synthetic_anomalies(test_scaled, train_scaled, val_scaled, column_names)
    ae_anomalies = detect_anomalies(train_scaled, val_scaled, test_anomalous, scaling, column_names, "hbw_1", train_model=False, test_scaled=test_scaled)
    if_anomalies = detect_using_isolation_forest(train_scaled, val_scaled, test_anomalous, scaling, column_names)
    oc_svm_anomalies = detect_using_one_class_support_vector_machine(train_scaled, val_scaled, test_anomalous, scaling, column_names)