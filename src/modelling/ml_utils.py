import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim

from datetime import datetime
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from typing import List

from src.pscapes import ProteinLandscape

class EarlyStopping:
    def __init__(self, 
                 patience: int = 5, 
                 min_delta: float = 0.0, 
                 path: str = 'best_model_{}.pt'):
        """
        Class for early stopping during model training. 

        Parameters:
        -----------
            patience : int
                How long to wait after last time validation loss
                improved.

            min_delta : float
                Minimum change in the monitored quantity to qualify as
                an improvement.

            path : str
                Path to save the best model. Date and time are appended.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.path = path.format(datetime.now().strftime("%y%m%d-%H%M"))
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(val_loss, model)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decreases."""
        torch.save(model.state_dict(), self.path)


def train_model(model: nn.Module,
                optimizer: optim.Optimizer,
                loss_fn: nn.modules.loss._Loss, 
                train_loader: DataLoader, 
                val_loader: DataLoader,
                n_epochs: int = 30,
                patience: int = 5, 
                min_delta: float = 1e-5,
                device: str='cpu'):
    """
    Function for training a model.

    Parameters:
    -----------
        model : nn.Module
            Instantiated instance of model() with appropriate parameters.

        optimizer : torch.optim.Optimizer
            Instantiated optimizer function instance.

        loss_fn : torch.nn.modules.loss
            Instantiated loss function instance.

        train_loader : torch.utils.data.DataLoader
            DataLoader with training data.

        val_loader : torch.utils.data.DataLoader
            DataLoader with validation data.

        n_epochs : int, default=30
            Number of epochs to train unless early stopping is initiated.

        patience : int, default=5
            Patience value for early stopping.

        min_delta : float, default=1e-5
            Minimum change in the monitored quantity to qualify as an 
            improvement for early stopping.

        device : str, default=cpu
            Device for PyTorch computations, e.g., 'cpu' or 'cuda'.
    """
    early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)
    model = model.to(device)
    val_epoch_losses = []
    train_epoch_losses = []

    assert n_epochs>0, 'n_epochs not strictly greater than 0, ensure n_epochs > 0'
    
    for epoch in range(n_epochs):
        model.train()  # Training mode
        
        train_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = loss_fn(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        epoch_loss = train_loss/len(train_loader)
        train_epoch_losses.append(epoch_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                #print(inputs.shape)
                #print(targets.shape)
                inputs, targets = inputs.to(device), targets.to(device)
                
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        val_epoch_losses.append(val_loss)

        print(
            f"Epoch [{epoch+1}/{n_epochs}], Train Loss: {train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}"
        )

        # Check early stopping
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

    # Load the best model after early stopping
    model.load_state_dict(torch.load(early_stopping.path))

    #delete best model from early stopiig from disk 
    if os.path.exists(early_stopping.path):
      os.remove(early_stopping.path)
    else:
      print("The file does not exist")
    return model, train_epoch_losses, val_epoch_losses


def get_trainable_params(model: nn.Module): 
    """
    Get the number of trainable parameters for a model.
    """
    pytorch_total_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    return pytorch_total_params


def landscape_data_split(landscape_ls: List[ProteinLandscape],
                         test_split: float = 0.2, 
                         val_split: float = 0.2, 
                         random_state: int = 1): 
    """
    Performs train-test-val splitting of data using a list of protein
    landscape class objects. NOTE: validation data proportion is
    proportion of TRAIN data NOT total data. 

    Parameters:
    -----------
    landscape_ls : List[ProteinLandscape]
        List of protein landscapes.

    test_split : float
        Proportion of total data used for testing

    val_split : float
        proportion of TRAIN data used for validation (NOT total data).

    random_state : int
        Random state of sklearn split.
    """
    x_ohe = [np.array(instance.ohe) for instance in landscape_ls]
    y_ohe = [instance.fitnesses.reshape(-1,1).astype(float)
             for instance in landscape_ls]
    # make list of tuples (x, y) for each landscape
    data_tuples = [list(zip(torch.from_numpy(x_ohe[i]).to(torch.float32), 
                            torch.from_numpy(y_ohe[i]).to(torch.float32))) 
                   for i in range(len(x_ohe))]
    
    # iterate through each landscape and collate data splits
    train_data_ls = []
    val_data_Ls = []
    tst_data_ls = []
    for data in data_tuples:
        # make data splits
        x_trn_outer, x_tst, y_trn_outer, y_tst = train_test_split(
            data, 
            test_size=len(data)*test_split,
            random_state=random_state
        )
        x_trn, x_val, y_trn, y_val = train_test_split(
            (x_trn_outer, y_trn_outer),
            test_size=round(len(x_trn_outer)*val_split),
            random_state=random_state
        )
        train_data_ls.append((x_trn, y_trn))
        val_data_Ls.append((x_val, y_val))
        tst_data_ls.append((x_tst, y_tst))

    return x_ohe, train_data_ls, val_data_Ls, tst_data_ls