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
#sys.path.append('./')
sys.path.append('../../nk-2025')



from torch.utils.data import DataLoader

from pscapes.landscape_class import ProteinLandscape
from pscapes.utils import dict_to_np_array, np_array_to_dict

from src.architectures import SequenceRegressionCNN, SequenceRegressionLinear, SequenceRegressionMLP, SequenceRegressionLSTM, SequenceRegressionTransformer

from ml_utils import train_val_test_split_ohe, landscapes_ohe_to_numpy
from hyperopt import objective_NK, sklearn_objective_NK

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor 
from colorama import Fore

torch.backends.nnpack.enabled = False



def run_hparam_opt_GB1(): 

    print('Initialising parameters...')
    SEQ_LEN = 4
    AA_ALPHABET  = 'ACDEFGHIKLMNPQRSTVWY'
    ALPHABET_LEN = len(AA_ALPHABET)
    #K_VALUES_TO_LOAD = range(SEQ_LEN)       
    REPLICATES = 1 #we only optimise hyperparameters on a single set of replicates for computational efficiency
    N_TRIALS_MULTIPLIER = 10 #15 #we use a multiplier -- the larger the hparam space, the more trials 
    PATIENCE = 15
    MIN_DELTA = 1e-5
    learning_rates = [0.01, 0.001, 0.0001]
    batch_sizes    = [32, 64, 128, 256]
    n_epochs = 150
    
    
    
    LINEAR_HPARAMS_SPACE = {'learning_rate': learning_rates, 'batch_size': batch_sizes, 
                               'alphabet_size':ALPHABET_LEN, 'sequence_length':SEQ_LEN} 
        
    MLP_HPARAM_SPACE     = {'learning_rate': learning_rates, 'batch_size': batch_sizes, 
                            'alphabet_size':ALPHABET_LEN, 'sequence_length':SEQ_LEN, 'max_hidden_layers':6, #added
                            'hidden_sizes_categorical': [32, 64, 128, 256, 512]} 
    
    CNN_HPARAM_SPACE     = {'learning_rate': learning_rates, 'batch_size': batch_sizes, 
                            'alphabet_size':ALPHABET_LEN, 'sequence_length':SEQ_LEN, 'max_conv_layers':2, #important to keep the max_conv_layers small for NK landscapes to avoid pooling error resulting in RuntimeError: max_pool1d() Invalid computed output size: 0
    
                            'n_kernels_min':32, 'n_kernels_max':512, 'n_kernels_step': 32, 'kernel_sizes_min':3, 
                           'kernel_sizes_max':3}
    
    uLSTM_HPARAM_SPACE   = {'learning_rate': learning_rates, 'batch_size': batch_sizes, 
                            'alphabet_size':ALPHABET_LEN, 'max_lstm_layers': 2, 'hidden_sizes': [64, 128, 256, 512], 
                           'bidirectional':False}
    
    bLSTM_HPARAM_SPACE   = {'learning_rate': learning_rates, 'batch_size': batch_sizes, 
                            'alphabet_size':ALPHABET_LEN, 'max_lstm_layers': 2, 'hidden_sizes': [64, 128, 256, 512], 
                            'bidirectional':True}
    
    TRANS_HPARAM_SPACE   = {'learning_rate': learning_rates, 'batch_size': batch_sizes, 
                            'alphabet_size':ALPHABET_LEN,'embed_dim_options':[ 32, 64, 128, 256, 512],                                       'max_heads':8, 'max_layers':3, 'feedforward_dims': [32, 64, 128, 256, 512], 
                            'max_seq_lengths':[4, 6, 8]}
    
    
    print('Loading landscapes...')
    #load GB1 landscape
    landscape = ProteinLandscape(csv_path='../data/experimental_datasets/G_prot_4_mut_seq_space_only.csv', amino_acids=AA_ALPHABET)
    
    LANDSCAPES = [landscape.fit_OHE()]
    
    landscapes_ohe, xy_train, xy_val, xy_test, x_test, y_test = train_val_test_split_ohe(LANDSCAPES, test_split=0.75, val_split=0.2, random_state=1)
    
    model_names = ['linear', 'mlp', 'cnn', 'ulstm', 'blstm', 'transformer', 'RF', 'GB']
    study_list = [[opt.create_study(direction='minimize') for i in LANDSCAPES] for j in model_names]
    
    
    models = [SequenceRegressionLinear, SequenceRegressionMLP, SequenceRegressionCNN, SequenceRegressionLSTM, 
                  SequenceRegressionLSTM, SequenceRegressionTransformer, RandomForestRegressor, GradientBoostingRegressor]

    studies    = {x:y for x,y in zip(model_names, study_list)}
    hparam_list = [LINEAR_HPARAMS_SPACE, MLP_HPARAM_SPACE, CNN_HPARAM_SPACE, uLSTM_HPARAM_SPACE, bLSTM_HPARAM_SPACE, TRANS_HPARAM_SPACE]
    
    x_train_np, y_train_np = landscapes_ohe_to_numpy(xy_train) #intialise flattened np arrays for RF and GB training 
    x_val_np, y_val_np = landscapes_ohe_to_numpy(xy_val)
    
    
    print('Running studies...')
    times = {}
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    for model_index, model_name in enumerate(model_names): 
        print(Fore.GREEN + 'Optimising hyperparameters for model: {}'.format(model_name) + Fore.RESET)
                      
        t1 = time.time()
    
        for study_index, study in enumerate(studies[model_name]):
            #study index loops over landscapes for each model 
    
    
    
            if model_name=='RF' or model_name=='GB': 
                
                n_trials = 3*N_TRIALS_MULTIPLIER
                study.optimize(lambda trial: sklearn_objective_NK(trial, model_name, x_train=x_train_np[study_index], y_train=y_train_np[study_index].ravel(), 
                    x_val=x_val_np[study_index], y_val=y_val_np[study_index].ravel()), n_trials=n_trials )
    
            else:
                n_trials = (len(hparam_list[model_index])-2)*N_TRIALS_MULTIPLIER
                model = models[model_index]
                study.optimize(lambda trial: objective_NK(trial, hparam_list[model_index], model,  
                    train_data= xy_train[study_index], val_data=xy_val[study_index], n_epochs=n_epochs, device=device, patience=PATIENCE, min_delta=MIN_DELTA), n_trials=n_trials)
            with open('../hyperopt/results/GB1_hyperopt_results.pkl', 'wb') as handle: #write file as you go for each study 
                pickle.dump(studies, handle,protocol=pickle.HIGHEST_PROTOCOL )
        
        t2 = time.time()
        times[model_name]=(t2-t1)
    
    for model_name in times.keys(): 
        t = times[model_name]
        mins = t/60
        print('Model:{} took {} mins to optimise hyperparameters'.format(model_name, mins) )
        
    print('Hyperparameter optimisation for GB1 complete.')
    
    
if __name__ == "__main__":
    run_hparam_opt_GB1()
    












