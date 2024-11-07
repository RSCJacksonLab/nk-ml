import torch
import torch.nn as nn
from sklearn.neighbors import kneighbors_graph

import math
import networkx as nx
from scipy.sparse import diags

import sys
sys.path.append('../../pscapes')
sys.path.append('../../nk-ml-2024/')

from src.train_utils import read_CNN_hparams, read_MLP_hparams, read_LSTM_hparams, read_transformer_hparams

def get_latent_representation(model_instance): 
    latent_model = nn.Sequential(*list(model_instance.children())[:-1])
    return latent_model

def instantiate_model_from_study(model_name, study, alphabet_size=6, seq_length=6): 
    if model_name == 'linear':
        model_instance = SequenceRegressionLinear(alphabet_size=alphabet_size, sequence_length=seq_length)
    elif model_name == 'mlp': 
        hparams = read_MLP_hparams(study.best_params)
        model_instance = SequenceRegressionMLP(**hparams, alphabet_size=alphabet_size, sequence_length=seq_length)
    elif model_name == 'cnn': 
        hparams = read_CNN_hparams(study.best_params)
        model_instance = SequenceRegressionCNN(**hparams, input_channels=alphabet_size, sequence_length=seq_length)
    elif model_name == 'ulstm': 
        hparams = read_LSTM_hparams(study.best_params)
        model_instance = SequenceRegressionLSTM(**hparams, input_size=alphabet_size, bidirectional=False)
    elif model_name == 'blstm': 
        hparams = read_LSTM_hparams(study.best_params)
        model_instance = SequenceRegressionLSTM(**hparams, input_size=alphabet_size, bidirectional=True)
    elif model_name == 'transformer': 
        hparams = read_transformer_hparams(study.best_params)
        model_instance = SequenceRegressionTransformer(**hparams, input_dim=alphabet_size)
    elif model_name == 'RF': 
        hparams = study.best_params
        model_instance = RandomForestRegressor(**hparams)
    elif model_name == 'GB': 
        hparams = study.best_params
        model_instance = GradientBoostingRegressor(**hparams)
    return model_instance




def adjacency_to_diag_laplacian(A): 
    """
    Calculates degree and laplacian matrices from an adjacency matrix.  
    
    Args: 
        A (scipy sparse matrix): adjacency of graph
    Returns: 
        D (scipy sparse matrix): degree matrix of graph
        L (scipy sparse matrix) : laplacian matrix of graph
    """
    degrees = A.sum(axis=1).A1
    D = diags(degrees, format='csr')
    L = D-A

    
    return D, L

def sparse_dirichlet(L, f): 
    """
    Calculates the Dirichlet energy of a signal f over a graph. 
    
    Args: 
    L (scipy sparse matrix): graph laplacian
    f (np array): signal over graph
    """

    f = f.astype('float64') 
    Lf  = L.dot(f)
    f_T = f.T
    fLf = f_T.dot(Lf)
    return fLf.item()
    