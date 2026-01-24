# Using code from https://docs.pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets
from torchvision.transforms import ToTensor

from autoencoder_data_preparation import read_and_prepare_data
import numpy as np
from tabulate import tabulate

import optuna

# Defining the autoencoder
class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, activation_func):
        super().__init__()
        
        # Calculate sizes of the hidden layers
        h_layer_1_dim = int(0.66 * input_dim + 0.33 * latent_dim) # approximation of input_dim - (input_dim - latent_dim) * 1/3
        h_layer_2_dim = int(0.33 * input_dim + 0.66 * latent_dim) # approximation of input_dim - (input_dim - latent_dim) * 2/3
        
        # The autoencoder is seperated in an encoder and an decoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, h_layer_1_dim),
            nn.LeakyReLU(),
            nn.Linear(h_layer_1_dim, h_layer_2_dim),
            nn.LeakyReLU(),
            nn.Linear(h_layer_2_dim, latent_dim),
            nn.LeakyReLU()
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, h_layer_2_dim),
            nn.LeakyReLU(),
            nn.Linear(h_layer_2_dim, h_layer_1_dim),
            nn.LeakyReLU(),
            nn.Linear(h_layer_1_dim, input_dim)
        )
        
    def forward(self, x):
        x_compressed = self.encoder(x)
        x_decoded = self.decoder(x_compressed)
        return x_decoded
    
def train_batch(dataloader, model, loss_fn, optimizer, device, prints = False):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        
        # X == y in case of autoencoders
        X, y = X.to(device), y.to(device)
        
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        
        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        # Log training informations
        if prints and batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            
def test(dataloader, model, loss_fn, device, prints=False):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
    test_loss /= num_batches
    if prints:
        print(f"Test Error: \n Avg loss: {test_loss:>8f} \n")
        
    return test_loss
    
def objective_initializer(train_scaled, test_scaled, column_names, scaling, prints = False):
    
    def objective(trial):
        # set the device to train on
        device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

        # hyperparameters
        compression_ratio = trial.suggest_categorial("compression_ratio", [0.25, 0.5, 0.75])
        lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
        batch_size = trial.suggest_categorical("batch_size", [2, 4, 8, 16, 32, 64])
        l2 = trial.suggest_categorical("l2", [2**i for i in range(9)]) # TODO implement l2 regularization
        activation_func = trial.suggest_categorical("activation_func", ["relu", "tanh", "sigmoid"]) # TODO implement hyperparameter tuning for activation func
        
        model = Autoencoder(
            len(column_names), 
            min(int(len(column_names) * compression_ratio), 2)
            ).to(device)
        
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr) # TODO think about optimizer choice
        
        # Convert to tensors if not already
        train_scaled = torch.tensor(train_scaled, dtype=torch.float32)
        test_scaled = torch.tensor(test_scaled, dtype=torch.float32)
        
        # Load the dataset
        dataset_train = TensorDataset(train_scaled, train_scaled)
        train_dataloader = DataLoader(
            dataset_train,
            batch_size=batch_size,
            shuffle=True
        )
        
        dataset_test = TensorDataset(test_scaled, test_scaled)
        test_dataloader = DataLoader(
            dataset_test,
            batch_size=batch_size,
            shuffle=False
        )
        
        # Start training
        epochs = 5
        for t in range(epochs):
            train_batch(train_dataloader, model, loss_fn, optimizer, device, prints=False)
            
        test_loss = test(test_dataloader, model, loss_fn, device, prints=False)
        return test_loss
    
    return objective

def train_AE(train_scaled, test_scaled, column_names, scaling, prints = False):
    # set the device to train on
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    
    if prints:
        print(f"Training using {device}\n")
    
    model = Autoencoder(len(column_names), int(len(column_names)/3)).to(device)
    if prints:
        print("Model:\n", model, "\n\n")
        
    # TODO Loss function for reconstruction
    loss_fn = nn.MSELoss()
    
    # TODO Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Convert to tensors if not already
    train_scaled = torch.tensor(train_scaled, dtype=torch.float32)
    test_scaled = torch.tensor(test_scaled, dtype=torch.float32)
    
    # Load the dataset
    dataset_train = TensorDataset(train_scaled, train_scaled)
    train_dataloader = DataLoader(
        dataset_train,
        batch_size=32,  # TODO Set batch size for hyperparameter tuning
        shuffle=True
    )
    
    dataset_test = TensorDataset(test_scaled, test_scaled)
    test_dataloader = DataLoader(
        dataset_test,
        batch_size=32,
        shuffle=False
    )
    
    # Start training
    epochs = 5
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_batch(train_dataloader, model, loss_fn, optimizer, device, prints=prints)
        test(test_dataloader, model, loss_fn, device)
    print("\nDone training!\n")
    
    # Show example data rows and their output
    # Get 5 random rows from the test data
    random_indices = np.random.choice(len(test_scaled), 5, replace=False)
    example_data = test_scaled[random_indices].cpu().numpy()
    
    # Reverse the scaling
    example_data_reversed = scaling.inverse_transform(example_data)
    
    # Get the output of the autoencoder for each of the rows
    with torch.no_grad():
        output = model(torch.tensor(example_data, dtype=torch.float32).to(device)).cpu().numpy()
    
    # Reverse the scaling for the output
    output_reversed = scaling.inverse_transform(output)
    
    # Print the results as a table
    for i in range(5):
        original = dict(zip(column_names, example_data_reversed[i]))
        output_dict = dict(zip(column_names, output_reversed[i]))
        
        table_data = [[k, f"{original[k]:.4f}", f"{output_dict[k]:.4f}"] for k in column_names]
        print(f"\nRow {i+1}:")
        print(tabulate(table_data, headers=["Feature", "Original", "Reconstructed"], tablefmt="grid"))
        
    
    # Show example anamoulous data rows and their output
    # Get 5 random rows from the test data
    random_indices = np.random.choice(len(test_scaled), 5, replace=False)
    example_data = test_scaled[random_indices].cpu().numpy()
    
    # Reverse the scaling
    example_data_reversed = scaling.inverse_transform(example_data)

    # Get the output of the autoencoder for each of the rows
    with torch.no_grad():
        output = model(torch.tensor(example_data, dtype=torch.float32).to(device)).cpu().numpy()

    # Reverse the scaling for the output
    output_reversed = scaling.inverse_transform(output)

    # Print the results as a table
    for i in range(5):
        # Choose a random column to modify
        col_idx = np.random.randint(0, len(column_names))
        original_value = example_data[i, col_idx]
        
        # Find a value far away from the original (use min or max from training data)
        col_min, col_max = train_scaled[:, col_idx].min().item(), train_scaled[:, col_idx].max().item()
        anomalous_value = col_max if original_value < (col_min + col_max) / 2 else col_min
        
        # Create anomalous data
        anomalous_data = example_data[i].copy()
        anomalous_data[col_idx] = anomalous_value
        
        # Get model output and calculate loss
        anomalous_tensor = torch.tensor([anomalous_data], dtype=torch.float32).to(device)
        with torch.no_grad():
            anomalous_output_tensor = model(anomalous_tensor)
            anomalous_output = anomalous_output_tensor.cpu().numpy()[0]
        
        loss = loss_fn(anomalous_output_tensor, anomalous_tensor).item()
        
        # Reverse scaling
        anomalous_data_reversed = scaling.inverse_transform([anomalous_data])[0]
        anomalous_output_reversed = scaling.inverse_transform([anomalous_output])[0]
    
        print(f"\nAnomaly Row {i+1}:")
        print(f"Modified column: {column_names[col_idx]}, Original: {example_data_reversed[i, col_idx]:.4f}, Anomalous: {anomalous_data_reversed[col_idx]:.4f}")
        print(f"Reconstruction Loss: {loss:.8f}")
        
        table_data = [[k, f"{anomalous_data_reversed[j]:.4f}", f"{anomalous_output_reversed[j]:.4f}"] for j, k in enumerate(column_names)]
        print(tabulate(table_data, headers=["Feature", "Anomalous Input", "Reconstructed"], tablefmt="grid"))
    
if __name__ == "__main__":
    train_scaled, test_scaled, scaling, column_names = read_and_prepare_data("ov_1", prints=False)
    
    # Hyperparameter tuning
    # Create and optimize study
    # TODO include missing hyperparameters and implement l2 regularization
    # TODO check for further hyperparameters to include
    study = optuna.create_study(direction="minimize")
    study.optimize(objective_initializer, n_trials=50)
    
    print("Best hyperparameters:", study.best_params)
    
    # TODO use found hyperparameters in final training step
    train_AE(train_scaled, test_scaled, column_names, scaling, prints=True)
    
    # TODO implement synthetic anomaly generation and check for precision on those
