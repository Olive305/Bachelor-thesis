from Data.anomaly_detection import detect_anomalies, detect_using_isolation_forest, detect_using_one_class_support_vector_machine
from Data.autoencoder_data_preparation import read_and_prepare_data
from Data.pytorch_autoencoder import create_AE
import tabulate
import os
import pandas as pd

if __name__ == "__main__":
    
    # Ask for which resources anomaly detection should be performed
    resources = []
    all_resources = ["hbw_1"] #! Fill with all possible resources
    user_input = input("Do you want to perform anomaly detection on all resources (workstations)? (y/n):\n")
    if user_input.lower() == 'y':
        resources = all_resources
        print("\n")
    else:
        # Ask for individual resources
        print("Which resources do you want to perform anomaly detection on? Possible choices:")
        for resource in all_resources:
            print(" ", resource)
            
        print("For information about each resource, read README.")
        user_input = input("List all resources you want to use, seperated by commas:\n")
        resources = [resource.strip() for resource in user_input.split(',') if resource.strip() in all_resources]
        print("\n")
    
    # Ask, if the training data should be loaded from all the data again and then prepared
    user_input = input("Do you want to load and prepare training data from scratch? (y/n):\n")
    reread_prepared_data = user_input.lower() == 'y'
    print("\n")
    
    # Ask, if the autoencoder should be retrained
    user_input = input("Do you want to retrain the autoencoder? (y/n):\n")
    retrain_AE = user_input.lower() == 'y'
    print("\n")
    
    # Ask if the hyperparameters should be redone
    redo_hyperparameter_tuning = False
    if retrain_AE:
        user_input = input("Do you want to redo the hyperparameter tuning? (y/n)\nWARNING: Redoing hyperparameter tuning takes a long time.\n")
        redo_hyperparameter_tuning = user_input.lower() == 'y'
        print("\n")

    for resource in resources:
        # Get the data
        print(f"Performing Anomaly detection on resource {resource}...\n\n", "Loading preprepared data...\n" if reread_prepared_data else "Readinga and preparing data...\n")
        train_scaled, test_scaled, val_scaled, scaling, column_names = read_and_prepare_data(resource, load_preprepared=reread_prepared_data, prints=False)
        
        # Perform anomaly detection on the data
        print("Performing anomaly detection on the data...\n")
        precision, recall, f1_score, f2_score, treshold = detect_anomalies(train_scaled, test_scaled, val_scaled, scaling, column_names, resource, train_model=retrain_AE, redo_hyperparameter_tuning=redo_hyperparameter_tuning)
        
        # Perform anomaly detection on the data using baseline models for comparison
        print("Performing anomaly detection using baseline techniques...\n")
        if_precision, if_recall, if_f1_score, if_f2_score = detect_using_isolation_forest(train_scaled, test_scaled, val_scaled, scaling, column_names)
        svm_precision, svm_recall, svm_f1_score, svm_f2_score = detect_using_one_class_support_vector_machine(train_scaled, test_scaled, val_scaled, scaling, column_names)
        
        # Create metrics table
        metrics_data = {
            'Metric': ['Precision', 'Recall', 'F1-Score', 'F2-Score', 'Threshold'],
            'Autoencoder': [precision, recall, f1_score, f2_score, treshold],
            'Isolation Forest': [if_precision, if_recall, if_f1_score, if_f2_score, ''],
            'One-Class SVM': [svm_precision, svm_recall, svm_f1_score, svm_f2_score, '']
        }

        # Display table in console
        print(tabulate.tabulate(metrics_data, headers='keys', tablefmt='grid'))
        print("\n")

        # Export to Excel

        output_dir = "anomaly_detection_metrics"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        df = pd.DataFrame(metrics_data)
        output_file = os.path.join(output_dir, f"{resource}_metrics_table.xlsx")
        df.to_excel(output_file, index=False)
        print(f"Metrics table exported to {output_file}\n")