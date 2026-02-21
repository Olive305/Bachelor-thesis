from Data.anomaly_detection import detect_anomalies, detect_using_isolation_forest, detect_using_one_class_support_vector_machine
from Data.autoencoder_data_preparation import read_and_prepare_data
from Data.pytorch_autoencoder import create_AE

if __name__ == "__main__":
    
    # Ask for which resources anomaly detection should be performed
    resources = []
    all_resources = [] #! Fill with all possible resources
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

    # Get the data
    read_and_prepare_data()