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

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor 

from train_utils import train_models_from_hparams_NK


print('Initialising parameters...')

SEQ_LEN = 6
AA_ALPHABET  = 'ACDEFG'
HPARAM_PATH = '../hyperopt/results/NK_hyperopt_results.pkl'
DATA_PATH = '../data/nk_landscapes/'
MODEL_SAVEPATH = '../models/models_K3'
RESULT_PATH = '../results/results_K3'

N_EPOCHS = 200
N_REPLICATES = 4

PATIENCE = 20
MIN_DELTA = 1e-6


def main(): 
    print('Loading hyperparameter optimisation results')
    with open(HPARAM_PATH, 'rb') as handle: 
        NK_hparams = pickle.load(handle)
    
    NK_hparams_k3 = {x:[NK_hparams[x][3] for _ in range(SEQ_LEN)] for x in NK_hparams.keys()} #train using the K=3 model for each model. We can repeat the same for K=2 later 
    
    print('Training models on best hyperparameters.')
    res = train_models_from_hparams_NK(NK_hparams_k3, DATA_PATH, model_savepath=MODEL_SAVEPATH, result_path=RESULT_PATH,
                                     amino_acids=AA_ALPHABET, 
                                     seq_length=SEQ_LEN, n_replicates=N_REPLICATES, n_epochs=N_EPOCHS, patience=PATIENCE, min_delta=MIN_DELTA) 
    
    
    print('Training complete. Results saved to disk.')


if __name__ == "__main__":
    main()













