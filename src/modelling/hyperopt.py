'''
High-level code for hyperparameter optimisation
'''
import optuna as opt
import torch
import torch.nn as nn
import torch.optim as optim

from numpy.typing import ArrayLike
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader
from typing import Any, Dict, List

from modelling.architectures import NeuralNetworkRegression
from modelling import make_dataset

# torch.backends.nnpack.enabled = False

class EarlyStoppingHparamOpt:
    """Class for early stopping during hparam optimisation"""
    def __init__(self, patience: int=5, min_delta: float=0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0

    def should_stop(self, val_loss: float):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


def optimise_hparams(trial: opt.Trial, 
                     model: nn.Module, 
                     loss_fn: nn.modules.loss._Loss,
                     optimizer: optim.Optimizer,
                     train_dataloader: DataLoader,
                     val_dataloader: DataLoader,
                     device: str,
                     n_epochs: int = 30,
                     patience: int = 5,
                     min_delta: float = 1e-5):
    """
    Function to run inner training/validation loop for hparam
    optimisation. Similar in function to train_model(). 

    Parameters:
    -----------
    trial : optuna.trial.Trial: 
        Optuna trial keyword for hyperparameter optimization.

    model : nn.Module: 
        An instantiated instance of the model with appropriate 
        parameters.

    loss_fn : torch.nn.modules.loss._Loss
        Instantiated loss function instance.

    optimizer : torch.optim.Optimizer
        Instantiated optimizer instance.

    train_dataloader : torch.utils.data.DataLoader
        DataLoader containing the training data.

    val_dataloader : torch.utils.data.DataLoader
        DataLoader containing the validation data.

    device : str
        Device for PyTorch computations, e.g., 'cpu' or 'cuda'.

    n_epochs : int, default=30
        The number of epochs to train, unless early stopping is
        triggered.

    patience : int, default=5
        Patience value for early stopping (number of epochs to wait for
        improvement).

    min_delta : float, default=1e-5
        Minimum change in validation loss to qualify as an improvement
        for early stopping.

    Returns:
    --------
    val_loss : float
        A float containing the validation loss.
               
    """
    early_stopping = EarlyStoppingHparamOpt(patience=patience,
                                            min_delta=min_delta)
    model.to(device)

    epoch_val_losses = []
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

        epoch_val_losses.append(val_loss)
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
    trial.set_user_attr("epoch_validation_losses", epoch_val_losses)
    trial.set_user_attr("n_epochs", len(epoch_val_losses))
    return val_loss
        

def get_tx_hparam_combinations(embed_dim_options: List[int], 
                               max_heads: int):
    """
    Generate valid combinations of embedding dimension and the number 
    of attention heads. Ensures the embedding dimension (embed_dim) 
    can be evenly divided by the number of attention heads (num_heads).

    Parameters:
    -----------
        embed_dim_options (List[int]): 
            A list of integers representing options for the embedding 
            dimension.
        max_heads (int): 
            The maximum number of attention heads to consider.

    Returns:
    --------
        List[Tuple[int, int]]: 
            A list of tuples, where each tuple contains a valid 
            embedding dimension and the corresponding number of 
            attention heads.
    """

    valid_combinations = []
    
    for embed_dim in embed_dim_options:
        for num_heads in range(1, max_heads + 1):
            if embed_dim % num_heads == 0:
                valid_combinations.append((embed_dim, num_heads))
    
    return valid_combinations
   
def objective_fn(trial: opt.Trial,
                 model_name: str,
                 search_space: Dict[str, Any],
                 alphabet_len: int,
                 sequence_len: int,
                 train_data: Any,
                 val_data: Any,
                 n_epochs: int = 30,
                 patience: int = 5,
                 min_delta: float = 1e-5,
                 device: str = 'cuda') -> float:
    """
    Perform hyperparameter optimization on a given model. The search space 
    is defined in h_param_search_space as a dictionary, where keys are model 
    parameter names and values are Optuna trial samplers. Include all model 
    parameters needed for instantiation but not optimized to avoid errors.

    Parameters:
    -----------
    trial (optuna.trial.Trial): 
        Optuna trial object.

    model_name (str): 
        Name of model to optimize.

    search_space (Dict[str, Any]): 
        Dictionary defining hyperparameters to optimize. Keys are 
        parameter names, and values are either fixed values or Optuna 
        samplers. Example: 
        {'learning_rate': trial.suggest_categorical(
            'lr', [0.01, 0.001, 0.0001]
            ), 'sequence_length': 5}.

    alphabet_len : int
        Number of tokens/classes allowed per site

    sequence_len : int
        Length of training data sequences. 

    train_data (Any): 
        Training data.

    val_data (Any): 
        Validation data.

    n_epochs (int, optional): 
        Number of epochs to train. Defaults to 30.

    patience (int, optional): 
        Patience value for early stopping. Defaults to 5.

    min_delta (float, optional): 
        Minimum delta for early stopping. Defaults to 1e-5.

    device (str, optional): 
        Device for PyTorch computations (e.g., 'cuda' or 'cpu'). 
        Defaults to 'cuda'.

    Returns:
    --------
    val_loss : float 
        Best validation loss achieved during training.
    """

    # define search spaces based on model
    lr = trial.suggest_categorical('lr', search_space['learning_rate'])
    batch_size = trial.suggest_categorical('batch_size',
                                           search_space['batch_size'])
    if model_name=='linear':
        model_instance = NeuralNetworkRegression(
            'linear',
            **{'alphabet_size': alphabet_len, 
               'sequence_length': sequence_len,
               'lr': lr,
               'batch_size': batch_size},
        )     
    elif model_name=='mlp':
        n_hidden_layers = trial.suggest_int('n_hidden_layers',
                                            1,
                                            search_space['max_hidden_layers'])
        hidden_sizes = [
            int(
                trial.suggest_categorical(
                    f"hidden{i}_size", search_space['hidden_sizes_categorical']
                )
            )
            for i in range(n_hidden_layers)
        ]
        model_instance= NeuralNetworkRegression(
            'mlp',
            **{'alphabet_size': alphabet_len, 
               'sequence_length': sequence_len,
               'hidden_sizes': hidden_sizes,
               'lr': lr,
               'batch_size': batch_size},
        )
    elif model_name=='cnn':
        num_conv_layers = trial.suggest_int('num_conv_layers', 
                                            1, 
                                            search_space['max_conv_layers'])
        n_kernels = [
            int(trial.suggest_int("n_kernels_layer{}".format(i), 
                                  search_space['n_kernels_min'], 
                                  search_space['n_kernels_max'], 
                                  search_space['n_kernels_step']))
            for i in range(num_conv_layers)
        ]      
        kernel_sizes = [
            int(trial.suggest_int("kernel_size_layer{}".format(i), 
                                  search_space['kernel_sizes_min'], 
                                  search_space['kernel_sizes_max'], 2))
            for i in range(num_conv_layers)
        ]
        model_instance = NeuralNetworkRegression(
            'cnn',
            **{'input_channels': alphabet_len, 
               'sequence_length': sequence_len,
               'num_conv_layers': num_conv_layers,
               'n_kernels': n_kernels,
               'kernel_sizes': kernel_sizes,
               'lr': lr,
               'batch_size': batch_size},
        )       
    elif model_name=='ulstm':
        num_layers = trial.suggest_int('num_layers', 
                                       1,
                                       search_space['max_lstm_layers'])
        hidden_size = trial.suggest_categorical("hidden_size",
                                                search_space['hidden_sizes'])
        model_instance = NeuralNetworkRegression(
                            'lstm',
                            **{'input_size': alphabet_len, 
                            'hidden_size': hidden_size,
                            'num_layers': num_layers,
                            'bidirectional': False,
                            'lr': lr,
                            'batch_size': batch_size},
                        ) 
    elif model_name=='blstm':
        num_layers = trial.suggest_int('num_layers', 
                                       1,
                                       search_space['max_lstm_layers'])
        hidden_size = trial.suggest_categorical("hidden_size",
                                                search_space['hidden_sizes'])
        model_instance = NeuralNetworkRegression(
                            'lstm',
                            **{'input_size': alphabet_len, 
                            'hidden_size': hidden_size,
                            'num_layers': num_layers,
                            'bidirectional': True,
                            'lr': lr,
                            'batch_size': batch_size},
                        )
    elif model_name=='transformer': 
        embed_dim_options = search_space['embed_dim_options']
        max_heads = search_space['max_heads']
        valid_combinations = get_tx_hparam_combinations(embed_dim_options,
                                                        max_heads)
        d_model, nhead = trial.suggest_categorical(
            "embed_dim_num_heads", 
            valid_combinations
        )
        num_layers = trial.suggest_int(
            'num_layers', 
            1, 
            search_space['max_layers']
        )
        dim_feedforward = trial.suggest_categorical(
            'dim_feedforward',
            search_space['feedforward_dims']
        )
        max_seq_length = trial.suggest_categorical(
            "max_seq_length",
            search_space['max_seq_lengths']
        )                   
        model_instance = NeuralNetworkRegression(
                            'transformer',
                            **{'input_dim': alphabet_len, 
                            'd_model': d_model,
                            'nhead': nhead,
                            'dim_feedforward': dim_feedforward,
                            'max_seq_length': max_seq_length,
                            'lr': lr,
                            'batch_size': batch_size},
        )
    else: 
        raise Exception("Model not recognised.")


    #train and val loaders 
    _, val_res = model_instance.fit(train_data, val_data)

    return val_res


def sklearn_objective_fn(trial, 
                        model_name: str, 
                        x_train: ArrayLike, 
                        y_train: ArrayLike,
                        x_val: ArrayLike, 
                        y_val: ArrayLike) -> float:
    """
    Perform hyperparameter optimization for a scikit-learn model.

    Parameters:
    -----------
        trial (optuna.trial.Trial): 
            An Optuna trial object for hyperparameter sampling.

        model_name (str): 
            Name of the model to optimize, either 'RF' (Random Forest)
            or 'GB' (Gradient Boosting).

        x_train (np.array): 
            Training data features with shape (samples, seq_length * 
            alphabet_size).

        y_train (np.array): 
            Training data labels with shape (samples, ).

        x_val (np.array): 
            Validation data features with shape (samples, seq_length * 
            alphabet_size).

        y_val (np.array): 
            Validation data labels with shape (samples, ).

    Returns:
    --------
        val_loss : float
            Validation score (e.g., accuracy or loss) for the model with the 
            current hyperparameters.
    """
    if model_name=='RF': 
        max_features = trial.suggest_float('max_features', 0.1, 1)
        n_estimators = trial.suggest_int('n_estimators', 10, 1000)
        max_depth = trial.suggest_int('max_depth', 1, 32)
        model = RandomForestRegressor(max_features=max_features, 
                                      n_estimators=n_estimators, 
                                      max_depth=max_depth,
                                      n_jobs=-1)
    elif model_name=='GB':         
        max_depth = trial.suggest_int('max_depth', 1, 32)
        n_estimators = trial.suggest_int('n_estimators', 10, 1000)
        learning_rate = trial.suggest_float('learning_rate', 0.001, 0.2)
        model = GradientBoostingRegressor(max_depth=max_depth, 
                                          n_estimators=n_estimators, 
                                          learning_rate=learning_rate)
    else: 
        raise Exception('Model name must be "RF" or "GB".')
    
    print('Fitting model in trial.')
    model.fit(x_train, y_train)
    y_pred   = model.predict(x_val)
    val_loss =  mean_squared_error(y_val, y_pred)
    
    return val_loss 
