import optuna as opt
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
#from twilio.rest import Client
import pickle

import time

import sys
import os 


sys.path.append('../../pscapes')
sys.path.append('./')


from torch.utils.data import DataLoader

from pscapes.landscape_class import ProteinLandscape
from pscapes.utils import dict_to_np_array, np_array_to_dict

from architectures import SequenceRegressionCNN, SequenceRegressionLinear, SequenceRegressionMLP, SequenceRegressionLSTM, SequenceRegressionTransformer

from ml_utils import train_val_test_split_ohe, landscapes_ohe_to_numpy
from hyperopt import objective_NK, sklearn_objective_NK

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor 

from train_utils import train_models_from_hparams_NK


print('Initialising parameters...')

SEQ_LEN = 6
AA_ALPHABET  = 'ACDEFG'
HPARAM_PATH = '../hyperopt/results/NK_hyperopt_results.pkl'
DATA_PATH = '../data/nk_landscapes/'
MODEL_SAVEPATH = '../models/'
RESULT_PATH = '../results/'

N_EPOCHS = 100
N_REPLICATES = 1

PATIENCE = 5
MIN_DELTA = 1e-6


def main(): 
    print('Loading hyperparameter optimisation results')
    with open(HPARAM_PATH, 'rb') as handle: 
        NK_hparams = pickle.load(handle)
    
    
    
    print('Training models on best hyperparameters.')
    res = train_models_from_hparams_NK(HPARAM_PATH, DATA_PATH, model_savepath=MODEL_SAVEPATH, result_path=RESULT_PATH,
                                     amino_acids=AA_ALPHABET, 
                                     seq_length=SEQ_LEN, n_replicates=N_REPLICATES, n_epochs=N_EPOCHS, patience=PATIENCE, min_delta=MIN_DELTA) 
    
    
    print('Training complete. Results saved to disk.')


if __name__ == "__main__":
    main()













