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
import os

# Defining the autoencoder
class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, activation_func="relu", dropout_rate=0.1):
        super().__init__()
        
        # Calculate sizes of the hidden layers
        h_layer_1_dim = int(2/3 * input_dim + 1/3 * latent_dim)
        h_layer_2_dim = int(1/3 * input_dim + 2/3 * latent_dim)
        
        # Map activation function string to torch module
        activation_map = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid()
        }
        activation = activation_map.get(activation_func, nn.ReLU())
        
        # The autoencoder is separated in an encoder and a decoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, h_layer_1_dim),
            activation,
            nn.Dropout(dropout_rate),
            nn.Linear(h_layer_1_dim, h_layer_2_dim),
            activation,
            nn.Dropout(dropout_rate),
            nn.Linear(h_layer_2_dim, latent_dim),
            activation
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, h_layer_2_dim),
            activation,
            nn.Linear(h_layer_2_dim, h_layer_1_dim),
            activation,
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
    
def objective_initializer(train_scaled, test_scaled, column_names, scaling, prints=False):
    def objective(trial):
        # set the device to train on
        device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

        # hyperparameters
        compression_ratio = 0.5
        lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
        l2 = trial.suggest_float("l2", 1e-6, 1e-2, log=True)
        activation_func = trial.suggest_categorical("activation_func", ["relu", "tanh", "sigmoid"])
        
        model = Autoencoder(
            len(column_names), 
            max(int(len(column_names) * compression_ratio), 2),
            activation_func=activation_func
        ).to(device)
        
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.AdamW(model.parameters(), 
                                     lr=lr,
                                     weight_decay=l2)
        
        # Convert to tensors if not already
        train_tensor = torch.tensor(train_scaled, dtype=torch.float32)
        test_tensor = torch.tensor(test_scaled, dtype=torch.float32)
        
        # Load the datasets
        dataset_train = TensorDataset(train_tensor, train_tensor)
        train_dataloader = DataLoader(
            dataset_train,
            batch_size=batch_size,
            shuffle=True
        )
        
        dataset_test = TensorDataset(test_tensor, test_tensor)
        test_dataloader = DataLoader(
            dataset_test,
            batch_size=batch_size,
            shuffle=False
        )
        
        # Start training with early stopping
        epochs = 128
        patience = 4
        delta = 0.0001
        best_loss = None
        no_improvement_count = 0
        
        for epoch in range(epochs):
            train_batch(train_dataloader, model, loss_fn, optimizer, device, prints=False)
            current_loss = test(test_dataloader, model, loss_fn, device, prints=False)
            
            # Early stopping logic
            if best_loss is None or current_loss < best_loss - delta:
                best_loss = current_loss
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            
            if no_improvement_count >= patience:
                break
            
        test_loss = test(test_dataloader, model, loss_fn, device, prints=False)
        return test_loss
        
    return objective

def train_AE(train_scaled, test_scaled, column_names, scaling, parameters, prints = False):
    # set the device to train on
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    
    if prints:
        print(f"Training using {device}\n")
    
    # Set parameters
    compression_ratio = 0.5
    lr = parameters["lr"]
    batch_size = parameters["batch_size"]
    l2 = parameters["l2"]
    activation_func = parameters["activation_func"]
    
    model = Autoencoder(
            len(column_names), 
            max(int(len(column_names) * compression_ratio), 2),
            activation_func=activation_func
        ).to(device)
    if prints:
        print("Model:\n", model, "\n\n")
    
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), 
                                 lr=lr,
                                 weight_decay=l2)
    
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
    
    # Start training with early stopping
    epochs = 128
    patience = 3
    delta = 0.0001
    best_loss = None
    no_improvement_count = 0
    
    for epoch in range(epochs):
        train_batch(train_dataloader, model, loss_fn, optimizer, device, prints=False)
        current_loss = test(test_dataloader, model, loss_fn, device, prints=False)
        
        # Early stopping logic
        if best_loss is None or current_loss < best_loss - delta:
            best_loss = current_loss
            no_improvement_count = 0
        else:
            no_improvement_count += 1
        
        if prints:
            print(f"Epoch {epoch+1:2d}/{epochs} | Validation Loss: {current_loss:.6f}")
        
        if no_improvement_count >= patience:
            if prints:
                print("Early stopping triggered - no improvement observed.")
            break
    
    if prints:
        print("Done training!\n")
    
    # Save the trained model
    model_dir = "model"
    os.makedirs(model_dir, exist_ok=True)
    torch.save(model, os.path.join(model_dir, "autoencoder.pth"))
    
    return os.path.join(model_dir, "autoencoder.pth")
            
def create_AE(train_scaled, test_scaled, val_scaled, scaling, column_names):
    print("Current accelerator:", torch.accelerator.current_accelerator())
    print("Accelerator available:", torch.accelerator.is_available())
    
    # Hyperparameter tuning
    # Create and optimize study
    study = optuna.create_study(direction="minimize")
    study.optimize(objective_initializer(train_scaled, test_scaled, column_names, scaling, prints=False), n_trials=32, n_jobs=2)
    
    print("Best hyperparameters:", study.best_params)
    
    # Return the path to the model file location
    return train_AE(train_scaled, test_scaled, column_names, scaling, study.best_params, prints=True)
    
if __name__ == "__main__":
    pass
