# This file contains high-level code for hparam optimisation 
import optuna as opt
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
sys.path.append('/home/ubuntu/nk-paper-2024-1/nk-ml-2024')

from torch.utils.data import DataLoader



from src.architectures.architectures import SequenceRegressionCNN, SequenceRegressionLinear, SequenceRegressionMLP, SequenceRegressionLSTM, SequenceRegressionTransformer 

torch.backends.nnpack.enabled = False


class EarlyStoppingHparamOpt:
    """Class for early stopping during hparam optimisation"""
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0

    def should_stop(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1

        return self.counter >= self.patience


def optimise_hparams(trial, model, loss_fn, optimizer, train_dataloader, val_dataloader, device, n_epochs=30, patience=5, min_delta=1e-5):
    """
    Function to run inner training/validation loop for hparam optimisation. 
    Similar in function to train_model(). 

    Args:
        trial:                                         optuna trial keyword 
        model (src.architectures.architectures):       insantiated instance of model() with appropriate parameters
        loss_fn (torch.nn.modules.loss):               instantiated loss function instance 
        optimizer (torch.optim):                       instantiated optimizer function instance 
        train_dataloader (torch DataLoader):           DataLoader with train data
        val_dataloader (torch DataLoader):             DataLoader with val data 
        n_epochs (int):                                number of epochs to train unless early stopping initiated
        patience (int):                                patience value for early stopping 
        min_delta (float):                             min_delta change value for early stopping 
 
        device (str):                                  device for PyTorch
               
    """
    early_stopping = EarlyStoppingHparamOpt(patience=patience, min_delta=min_delta)
    model.to(device)

    for epoch in range(n_epochs):
        model.train()

        #train model this epoch
        for x_batch, y_batch in train_dataloader: 
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            predictions = model(x_batch)
            loss = loss_fn(predictions, y_batch)
            loss.backward()
            optimizer.step()

        #evaluate model validation loss this epoch 
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x_batch, y_batch in val_dataloader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                predictions = model(x_batch)
                val_loss += loss_fn(predictions, y_batch).item()

        val_loss /= len(val_dataloader)     

        #report to optuna 
        trial.report(val_loss, epoch)
        print('Epoch: {0}, Val loss: {1}'.format(epoch, val_loss))
        
        # Check early stopping
        if early_stopping.should_stop(val_loss):
            print(f"Early stopping at epoch {epoch}")
            break

        #check optuna pruning 
        if trial.should_prune():
            raise opt.TrialPruned()
    print('Best validation loss this trial: {}'.format(val_loss))
    return val_loss
        

def generate_valid_combinations_transformer(embed_dim_options, max_heads):
    valid_combinations = []
    
    for embed_dim in embed_dim_options:
        for num_heads in range(1, max_heads + 1):
            if embed_dim % num_heads == 0:
                valid_combinations.append((embed_dim, num_heads))
    
    return valid_combinations

def objective_NK(trial, h_param_search_space, model, train_data, val_data, n_epochs=30, patience=5, min_delta=1e-5, device='cuda'):
    """
    High-level function to perform hyperparameter optimisation on models. Define the search space in h_param_search_space (dict) 
    by specifying model parameter names as keys, and values as optuna trial samplers. Please also specify model parameters needed 
    for instantiation but that are not being optimised (otherwise model will return error). 

    trial:                           optuna trial object
    h_param_search_space (dict):     dict of hyperparameters {hparam_name: hparam_value}. Specify search space with optuna.trial sampler as value if 
                                     wanting to optimise that hyperparameter. Example: {'learning_rate': trial.suggest_categorical('lr', [0.01, 0.001, 0.0001]), 'sequence_length':5 }
    model (nn.Module):               model to optimise. Do NOT instantiate model with model() on passing.
    train_data:                      train data
    val_data:                        val data
    n_epochs (int):                  number of epochs to train for 
    patience (int):                  patience for early stopping 
    mind_delta(float):               min_delta for early stopping   
    """           

    #define search spaces based on model
    hpss= h_param_search_space
    learning_rate = trial.suggest_categorical('lr', hpss['learning_rate'])
    batch_size    = trial.suggest_categorical('batch_size', hpss['batch_size'])
    
    if model==SequenceRegressionLinear:
        model_instance = model(alphabet_size=hpss['alphabet_size'], sequence_length=hpss['sequence_length'])
        
    elif model==SequenceRegressionMLP:
        n_hidden_layers = trial.suggest_int('n_hidden_layers',1, hpss['max_hidden_layers']) #max_hidden_sizes should be an int
        hidden_sizes    = [int(trial.suggest_categorical("hidden{}_size".format(i), hpss['hidden_sizes_categorical'])) #hidden_sizes_categorical should be a list of hidden sizes
                            for i in range(n_hidden_layers)]
        model_instance = model(alphabet_size=hpss['alphabet_size'], sequence_length=hpss['sequence_length'], hidden_sizes=hidden_sizes)

    elif model==SequenceRegressionCNN:
        num_conv_layers = trial.suggest_int('num_conv_layers', 1, hpss['max_conv_layers']) #max_conv_layers should be an int
        n_kernels = [int(trial.suggest_discrete_uniform("n_kernels", hpss['n_kernels_min'], hpss['n_kernels_max'] , hpss['n_kernels_step']))for i in range(num_conv_layers)]      
        kernel_sizes = [int(trial.suggest_discrete_uniform("kernel_sizes", hpss['kernel_sizes_min'], hpss['kernel_sizes_max'], 1))for i in range(num_conv_layers)]
        model_instance = model(input_channels=hpss['alphabet_size'], sequence_length=hpss['sequence_length'], num_conv_layers=num_conv_layers,
                              n_kernels=n_kernels, kernel_sizes=kernel_sizes)
    elif model==SequenceRegressionLSTM: 
        num_layers     = trial.suggest_int('num_layers', 1, hpss['max_lstm_layers']) #max_lstm_layers should be int
        hidden_size    = trial.suggest_categorical("hidden_size", hpss['hidden_sizes']) #hidden_sizes should be a list of possible hidden layuer sizes
        bidirectional  = hpss['bidirectional']               
        model_instance = model(input_size=hpss['alphabet_size'], hidden_size=hidden_size, num_layers=num_layers, bidirectional=bidirectional)
    
    elif model==SequenceRegressionTransformer:
        embed_dim_options = hpss['embed_dim_options']
        max_heads = hpss['max_heads']
        valid_combinations = generate_valid_combinations_transformer(embed_dim_options, max_heads)
    
        d_model, nhead = trial.suggest_categorical("embed_dim_num_heads", valid_combinations)
            
        num_layers      = trial.suggest_int('num_layers', 1, hpss['max_layers']) #should be int
        dim_feedforward = trial.suggest_categorical('dim_feedforward', hpss['feedforward_dims']) # should be list of ints possible dims 
        max_seq_length  = trial.suggest_categorical("max_seq_length", hpss['max_seq_lengths']) #shold be list of ints of possible max seq lengths                   
        model_instance  = model(input_dim=hpss['alphabet_size'], d_model=d_model, nhead=nhead,dim_feedforward=dim_feedforward,
                               max_seq_length=max_seq_length)
        
    # Initialize model with the trial’s hyperparameters
    # Loss and optimizer
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model_instance.parameters(), lr=learning_rate)


    #train and val loaders 
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)

    #run train/val
    val_loss = optimise_hparams(trial, model_instance, loss_fn, optimizer, train_loader, 
                               val_loader, n_epochs=n_epochs, patience=patience, min_delta=min_delta, device=device)
    return val_loss



