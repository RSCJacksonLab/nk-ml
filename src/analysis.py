import torch
import torch.nn as nn
from sklearn.neighbors import kneighbors_graph

import math
import networkx as nx
from scipy.sparse import diags

import sys
sys.path.append('../../pscapes')
sys.path.append('../../nk-ml-2024/')
from scipy.sparse.linalg import eigsh

import numpy as np



def get_latent_representation(model, model_name,  x_data):
    # Variable to store the final layer activation
    final_activation = None

    # Define a forward hook callback function to capture the output
    def forward_hook(module, input, output):
        nonlocal final_activation  # Use nonlocal to modify the variable outside the inner function
        final_activation = output

    # Attach the hook to the final layer of the model
    if model_name == 'mlp': 
        final_layer = model.fc_layers[-1]
        hook_handle = final_layer.register_forward_hook(forward_hook)
    elif model_name == 'cnn': 
        final_layer = list(model.children())[-2] #gets the final MaxPool1d layer 
        hook_handle = final_layer.register_forward_hook(forward_hook)
    elif model_name == 'ulstm': 
        hook_handle = model.lstm.register_forward_hook(forward_hook)
    else: 
        raise Exception('Model name not recognised.')

    # Run a forward pass
    _ = model(x_data)

    # Remove the hook to prevent side effects
    hook_handle.remove()

    # Return the captured activation
    return final_activation.detach()

def check_negative_entries(A):
    if A.min() < 0:
        raise ValueError("Adjacency matrix contains negative entries.")
def check_symmetry(A):
    if (A != A.T).nnz != 0:
        raise ValueError("Adjacency matrix is not symmetric.")

def adjacency_to_diag_laplacian(A): 
    """
    Calculates degree and Laplacian matrices from an adjacency matrix.  
    
    Args: 
        A (scipy sparse matrix): adjacency of graph
    Returns: 
        D (scipy sparse matrix): degree matrix of graph
        L (scipy sparse matrix): Laplacian matrix of graph
    """
    # Check for negative entries in A
    if A.min() < 0:
        raise ValueError("Adjacency matrix has negative values.")
    
    # Ensure A is symmetric
    if (A != A.T).nnz != 0:
        raise ValueError("Adjacency matrix is not symmetric for an undirected graph.")
    
    degrees = np.array(A.sum(axis=1)).flatten()
    # Check for negative degrees
    if degrees.min() < 0:
        raise ValueError("Degree matrix has negative values.")
    
    D = diags(degrees, format='csr')
    L = D - A
    return D, L

def sparse_dirichlet(L, f): 
    """
    Calculates the Dirichlet energy of a signal f over a graph. 
    
    Args: 
        L (scipy sparse matrix): graph Laplacian
        f (np array): signal over graph
    """
    f = f.astype('float64')
    L = L.astype('float64')

    #min_eigval = eigsh(L, k=1, which='SA', return_eigenvectors=False)[0]
    #if min_eigval < -1e-10:
        #raise ValueError("Laplacian matrix is not positive semi-definite.")

    
    Lf = L.dot(f)
    fLf = f.T @ Lf  # Ensures matrix-vector product

    # Allow for minor numerical errors
    tol = 1e-10
    if fLf < -tol:
        raise ValueError(f"Dirichlet energy is negative: {fLf}")

    return fLf.item()

    """
    f = f.astype('float64') 
    Lf  = L.dot(f)
    f_T = f.T
    fLf = f_T.dot(Lf)
    return fLf.item()"""
    