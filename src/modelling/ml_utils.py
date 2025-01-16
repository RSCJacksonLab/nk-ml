'''
Functions for ML training.
'''
import os
import torch
import torch.nn as nn
import torch.optim as optim

from datetime import datetime
from torch.utils.data import DataLoader
from typing import Optional


def read_CNN_hparams(best_params: dict): 
    """
    Parse hyperparameters from an optuna best_params and return them in
    a form amenable for input into SequenceRegressionCNN instance. 
    """
    num_conv_layers = best_params['num_conv_layers']
    n_kernels = [int(best_params['n_kernels_layer{}'.format(i)]) 
                 for i in range(num_conv_layers)]
    kernel_sizes = [int(best_params['kernel_size_layer{}'.format(i)]) 
                    for i in range(num_conv_layers)]
    # make hparam dict
    hparam_dict = {'num_conv_layers': num_conv_layers, 
                   'n_kernels': n_kernels, 
                   'kernel_sizes':kernel_sizes}
    return hparam_dict

def read_MLP_hparams(best_params: dict): 
    """
    Parse hyperparameters from an optuna best_params and return them in
    a form amenable for input into SequenceRegressionMLP instance. 
    """
    params=best_params
    n_hidden_layers = params['n_hidden_layers']    
    hidden_sizes = [params['hidden{}_size'.format(i)] 
                    for i in range(n_hidden_layers)]
    param_dict  = {'hidden_sizes': hidden_sizes}
    return param_dict

def read_LSTM_hparams(best_params: dict): 
    """
    Parse hyperparameters from an optuna best_params and return them in
    a form amenable for input into SequenceRegressionLSTM instance. 
    """
    params = best_params
    num_layers  = params['num_layers']
    hidden_size = params['hidden_size']
    param_dict = {'num_layers': num_layers, 
                  'hidden_size': hidden_size}
    return param_dict


def read_transformer_hparams(best_params: dict): 
    """
    Parse hyperparameters from an optuna best_params and return them in 
    a form amenable for input into SequenceRegressionTransformer 
    instance.
    """
    params = best_params
    d_model = params['embed_dim_num_heads'][0]
    nheads  = params['embed_dim_num_heads'][1]
    num_layers = params['num_layers']
    dim_feedforward = params['dim_feedforward']
    max_seq_length = params['max_seq_length']
    param_dict = {'d_model': d_model, 
                  'nhead': nheads, 
                  'num_layers': num_layers,
                  'dim_feedforward': dim_feedforward, 
                  'max_seq_length': max_seq_length}
    return param_dict

def get_model_hparams(model_name: str, best_params: dict):
    '''
    Given model name read Optuna best_params and make dict.
    '''
    name_to_param = {
        'cnn': read_CNN_hparams,
        'mlp': read_MLP_hparams,
        'ulstm': read_LSTM_hparams,
        'blstm': read_LSTM_hparams,
        'transformer': read_transformer_hparams,
    }
    parse_fn = name_to_param[model_name]
    param_dict = parse_fn(best_params)
    return param_dict


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
                val_loader: Optional[DataLoader] = None,
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

    # delete best model from early stopiig from disk 
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