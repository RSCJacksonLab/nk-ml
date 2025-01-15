'''
Functions for training tuned ML models.
'''

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pickle

from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from src.modelling import train_model
from src.modelling.architectures import *
from src.pscapes import ProteinLandscape


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
    
    
# def train_tuned_models(hparams_dict: dict,
#                        datapath: str, 
#                        model_savepath: str,
#                        result_path: str, 
#                        amino_acids: str,
#                        seq_length: int, 
#                        n_replicates: int,
#                        n_epochs: int = 30,
#                        patience: int = 5,
#                        min_delta: float=1e-5):
#     """
#     Trains a given model using a set of hyperparameters on NK landscape
#     replicates.

#     Parameters:
#     -----------
#     hparams_dict : dict
#         Dictionary of hyperparameters to use for training.

#     datapath : str
#         Path to CSV files of NK landscapes, formatted as `k{}_r{}.csv`.

#     model_savepath : str
#         Path to save the trained model.

#     result_path : str
#         Path to store the training results.

#     amino_acids : str
#         Alphabet of amino acids used to create NK landscapes.

#     seq_length : int
#         Length of sequences in NK landscapes.

#     n_replicates : int
#         Number of replicate NK landscapes to load for each K value.

#     n_epochs : int, optional (default=30)
#         Maximum number of epochs for training.

#     patience : int, optional (default=5)
#         Number of epochs with no improvement after which training will
#         stop.

#     min_delta : float, optional (default=1e-5)
#         Minimum change in the monitored quantity to qualify as an 
#         improvement.

#     """
#     results = {}
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
#     # for each model type
#     for model_name in hparams_dict.keys():
#         print('Working on model name {}'.format(model_name))       
#         studies = hparams_dict[model_name]
#         results[model_name] = {x: None for x in range(len(studies))}
        
#         # iterate through each study and corresponding K value
#         for k, study in enumerate(studies):
#             print('Working on training for K = {}'.format(k))
#             # assign hyperparameters
#             params = study.best_params

#             # extract model agnostic hparams.
#             if model_name!='RF' and model_name!='GB':             
#                 lr = params['lr']
#                 batch_size = params['batch_size']
                
#             # extract model hparams 
#             if model_name == 'linear':
#                 model_params = {'alphabet_size': len(amino_acids), 
#                                 'sequence_length':seq_length}
#             elif model_name == 'mlp':
#                 model_params = read_MLP_hparams(params) 
#                 model_params['alphabet_size'] = len(amino_acids)
#                 model_params['sequence_length'] = seq_length
#             elif model_name == 'cnn': 
#                 model_params = read_CNN_hparams(params)
#                 model_params['input_channels'] = len(amino_acids)
#                 model_params['sequence_length'] = seq_length
#             elif model_name == 'ulstm': 
#                 model_params = read_LSTM_hparams(params)
#                 model_params['input_size'] = len(amino_acids)
#                 model_params['bidirectional'] = False 
#             elif model_name == 'blstm': 
#                 model_params = read_LSTM_hparams(params)
#                 model_params['input_size'] = len(amino_acids)
#                 model_params['bidirectional'] = True 
#             elif model_name == 'transformer': 
#                 model_params = read_transformer_hparams(params)
#                 model_params['input_dim']=len(amino_acids)
#             elif (model_name == 'RF') or (model_name == 'GB'):
#                 model_params = params
#             else:
#                 raise ValueError("Unknown model name.")
            
#             print (
#                 f"Hyperparameters: {model_params}, learning_rate: {lr}, "
#                 f"batch_size: {batch_size}"
#             )

#             k_results = {'test_r2': [], 
#                         'test_mse': [], 
#                         'test_pearson_r': [], 
#                         'train_epoch_losses': [], 
#                         'val_epoch_losses': [], 
#                         'predictions': [],
#                         'ground_truth': []}

#             # for each replicate
#             for replicate in range(n_replicates):
#                 # load landscape csv file
#                 landscape_csv_name = datapath + f'/k{k}_r{replicate}.csv'
#                 landscape = ProteinLandscape(csv_path=landscape_csv_name,
#                                             amino_acids=amino_acids)
#                 # load ohe and fitness data
#                 x_ohe = np.array(landscape.ohe)
#                 y = landscape.fitnesses.reshape(-1, 1).astype(float)

#                 # make data splits
#                 x_trn_outer, x_tst, y_trn_outer, y_tst = train_test_split(
#                     x_ohe,
#                     y 
#                     test_size=round(len(x_ohe) * 0.2),
#                     random_state=1,
#                 )
#                 x_trn, x_val, y_trn, y_val = train_test_split(
#                     x_trn_outer,
#                     y_trn_outer,
#                     test_size=round(len(x_trn_outer) * 0.2),
#                     random_state=1
#                 )            

#                 # if pytorch model
#                 if model_name !='RF' and model_name!='GB':

#                     # initialise model
#                     if model_name=='linear': 
#                         model_instance = SequenceRegressionLinear(**model_params)
#                     elif model_name=='mlp': 
#                         model_instance = SequenceRegressionMLP(**model_params)
#                     elif model_name=='cnn': 
#                         model_instance = SequenceRegressionCNN(**model_params)
#                     elif model_name=='ulstm':
#                         model_instance = SequenceRegressionLSTM(**model_params)
#                     elif model_name=='blstm': 
#                         model_instance = SequenceRegressionLSTM(**model_params)
#                     elif model_name=='transformer': 
#                         model_instance = SequenceRegressionTransformer(**model_params)
#                     else:
#                         raise ValueError("Unknown model name provided.")

#                     # initialise loss and optimizer
#                     loss_fn = nn.MSELoss()
#                     optimizer = optim.Adam(model_instance.parameters(), lr=lr)
                
#                     # Prepare dataloaders
#                     # convert to PyTorch tensors
#                     x_trn_pt = torch.from_numpy(x_trn).float()
#                     y_trn_pt = torch.from_numpy(y_trn).float()

#                     x_val_pt = torch.from_numpy(x_val).float()
#                     y_val_pt = torch.from_numpy(y_val).float()

#                     # combine into a list of tuples
#                     trn_dset = list(zip(x_trn_pt, y_trn_pt))
#                     val_dset = list(zip(x_val_pt, y_val_pt))

#                     # make dataloaders
#                     train_dataloader = DataLoader(
#                         trn_dset, 
#                         batch_size=batch_size, 
#                         shuffle=True
#                     )
#                     val_dataloader = DataLoader(
#                         val_dset, 
#                         batch_size=batch_size
#                     )
#                     # train model
#                     _, train_epoch_losses, val_epoch_losses = train_model(
#                         model_instance, 
#                         optimizer, 
#                         loss_fn, 
#                         train_dataloader, 
#                         val_dataloader,
#                         n_epochs=n_epochs,
#                         device=device,
#                         patience=patience,
#                         min_delta=min_delta)
                    
#                     # save model 
#                     model_file_name = (
#                         f'/{model_name}_NK_k{k}_r{replicate}.pt'
#                     )
#                     savepath = model_savepath + model_file_name
#                     torch.save(model_instance.state_dict(), savepath)

#                     # evaluate model on test data
#                     model_instance.eval()
#                     x_tst_pt = torch.from_numpy(x_tst).float().to(device)
#                     y_tst_pt = torch.from_numpy(y_tst).float()
                    
#                     y_pred = model_instance(x_tst_pt)
#                     y_pred = y_pred.detach()
                    
#                     # eval performance
#                     test_mse = loss_fn(y_pred, y_tst_pt)
#                     y_pred, y_test = y_pred.cpu(), y_tst_pt
#                     test_pearson_r = pearsonr(y_pred, y_test)                      
#                     test_r2  = r2_score(y_pred, y_test)

#                     # collect data 
#                     k_results['test_r2'].append(test_r2)
#                     k_results['test_mse'].append(test_mse)
#                     k_results['test_pearson_r'].append(test_pearson_r)
#                     k_results['train_epoch_losses'].append(train_epoch_losses)
#                     k_results['val_epoch_losses'].append(val_epoch_losses)
#                     k_results['predictions'].append(y_pred)
#                     k_results['ground_truth'].append(y_test)
                    
#                 # if sklearn model
#                 else:
#                     if model_name == 'RF': 
#                         model_instance = RandomForestRegressor(
#                             **model_params, 
#                             n_jobs=-1
#                         )
#                     elif model_name == 'GB': 
#                         model_instance = GradientBoostingRegressor(
#                             **model_params
#                         )
#                     else:
#                         raise ValueError("Unknown model name provided.")
                    
#                     model_instance.fit(x_trn, y_trn.ravel())
#                     y_pred = model_instance.predict(x_tst)
#                     y_test = y_tst.ravel()

#                     # eval performance
#                     test_mse = mean_squared_error(y_test, y_pred)
#                     test_r2 =  r2_score(y_test, y_pred)
#                     test_pearson_r = pearsonr(y_test, y_pred)
                    
#                     # collect data 
#                     k_results['test_r2'].append(test_r2)
#                     k_results['test_mse'].append(test_mse)
#                     k_results['test_pearson_r'].append(test_pearson_r)
                        
#             # update results dictionary 
#             results[model_name][k] = k_results 
#              # overwrite results at each model k value
#             with open(result_path+'/NK_train_test_results.pkl', 'wb') as f:
#                 pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
    
#     print('Training and testing finished. Results written to disk.')
#     return results                    