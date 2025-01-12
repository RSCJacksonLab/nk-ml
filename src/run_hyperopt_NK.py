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

from src.architectures import SequenceRegressionCNN, SequenceRegressionLinear, SequenceRegressionMLP, SequenceRegressionLSTM, SequenceRegressionTransformer

from ml_utils import train_val_test_split_ohe, landscapes_ohe_to_numpy
from hyperopt import objective, sklearn_objective

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor 
from colorama import Fore

torch.backends.nnpack.enabled = False

### MAKE SURE YOU DON'T TUNE HYPERPARAMETERS ON THE TEST SET !!!! IE YOU'LL NEED TO SAVE THE TEST/TRAIN/VAL SPLITS 
### --> I solved this problem using random_state in the train_test split function in numpy. Works well, I've tested it. 




    

def run_hparam_opt():

    print('Initialising parameters...')
    SEQ_LEN = 6
    AA_ALPHABET  = 'ACDEFG'
    ALPHABET_LEN = len(AA_ALPHABET)
    K_VALUES_TO_LOAD = range(SEQ_LEN)
    REPLICATES = 1 #we only optimise hyperparameters on a single set of replicates for computational efficiency
    N_TRIALS_MULTIPLIER = 15 #15 #we use a multiplier -- the larger the hparam space, the more trials 
    PATIENCE = 20
    MIN_DELTA = 1e-6
    learning_rates = [0.01, 0.001, 0.0001]
    batch_sizes    = [32, 64, 128, 256]

    #N_TRIALS = 64
    n_epochs = 300
    

    #define hyperparameter search space 
    LINEAR_HPARAMS_SPACE = {'learning_rate': learning_rates, 'batch_size': batch_sizes, 
                           'alphabet_size':ALPHABET_LEN, 'sequence_length':SEQ_LEN} 
    
    MLP_HPARAM_SPACE     = {'learning_rate': learning_rates, 'batch_size': batch_sizes, 
                            'alphabet_size':ALPHABET_LEN, 'sequence_length':SEQ_LEN, 'max_hidden_layers':3,
                            'hidden_sizes_categorical': [32, 64, 128, 256]} 
    
    CNN_HPARAM_SPACE     = {'learning_rate': learning_rates, 'batch_size': batch_sizes, 
                            'alphabet_size':ALPHABET_LEN, 'sequence_length':SEQ_LEN, 'max_conv_layers':2, #important to keep the max_conv_layers small for NK landscapes to avoid pooling error resulting in RuntimeError: max_pool1d() Invalid computed output size: 0

                            'n_kernels_min':32, 'n_kernels_max':256, 'n_kernels_step': 32, 'kernel_sizes_min':3, 
                           'kernel_sizes_max':5}

    uLSTM_HPARAM_SPACE   = {'learning_rate': learning_rates, 'batch_size': batch_sizes, 
                            'alphabet_size':ALPHABET_LEN, 'max_lstm_layers': 2, 'hidden_sizes': [64, 128, 256], 
                           'bidirectional':False}

    bLSTM_HPARAM_SPACE   = {'learning_rate': learning_rates, 'batch_size': batch_sizes, 
                            'alphabet_size':ALPHABET_LEN, 'max_lstm_layers': 2, 'hidden_sizes': [64, 128, 256], 
                            'bidirectional':True}

    TRANS_HPARAM_SPACE   = {'learning_rate': learning_rates, 'batch_size': batch_sizes, 
                            'alphabet_size':ALPHABET_LEN,'embed_dim_options':[ 32, 64, 128, 256],                                       'max_heads':8, 'max_layers':2, 'feedforward_dims': [32, 64, 128, 256], 
                            'max_seq_lengths':[6, 8, 10]}



    print('Loading landscapes...')
    LANDSCAPES = []
    for k in K_VALUES_TO_LOAD: 
        for r in range(REPLICATES): 
            landscape = ProteinLandscape(csv_path='../data/nk_landscapes/k{0}_r{1}.csv'.format(k,r), amino_acids=AA_ALPHABET)
            LANDSCAPES.append(landscape)

    LANDSCAPES = [i.fit_OHE() for i in LANDSCAPES]

    landscapes_ohe, xy_train, xy_val, xy_test, x_test, y_test = train_val_test_split_ohe(LANDSCAPES, random_state=1)


    x_train_np, y_train_np = landscapes_ohe_to_numpy(xy_train) #intialise flattened np arrays for RF and GB training 
    x_val_np, y_val_np = landscapes_ohe_to_numpy(xy_val)




    print('Creating studies...')

    
    model_names = ['linear', 'mlp', 'cnn', 'ulstm', 'blstm', 'transformer', 'RF', 'GB']

    models = [SequenceRegressionLinear, SequenceRegressionMLP, SequenceRegressionCNN, SequenceRegressionLSTM, 
              SequenceRegressionLSTM, SequenceRegressionTransformer, RandomForestRegressor, GradientBoostingRegressor]

    study_list = [[opt.create_study(direction='minimize') for i in LANDSCAPES] for j in model_names]

    studies    = {x:y for x,y in zip(model_names, study_list)}


    
    #note: hparam space for GB and RF are hardcoded 
    hparam_list = [LINEAR_HPARAMS_SPACE, MLP_HPARAM_SPACE, CNN_HPARAM_SPACE, uLSTM_HPARAM_SPACE, bLSTM_HPARAM_SPACE, TRANS_HPARAM_SPACE]


    

    print('Running studies...')
    times = {}
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for model_index, model_name in enumerate(model_names): 
        print(Fore.GREEN + 'Optimising hyperparameters for model: {}'.format(model_name) + Fore.RESET)
                      
        t1 = time.time()

        for study_index, study in enumerate(studies[model_name]):
            #study index loops over K values for each model 
            print('Optimising K={}'.format(study_index))



            if model_name=='RF' or model_name=='GB': 
                
                n_trials = 3*N_TRIALS_MULTIPLIER
                study.optimize(lambda trial: sklearn_objective(trial, model_name, x_train=x_train_np[study_index], y_train=y_train_np[study_index].ravel(), 
                    x_val=x_val_np[study_index], y_val=y_val_np[study_index].ravel()), n_trials=n_trials )

            else:
                n_trials = (len(hparam_list[model_index])-2)*N_TRIALS_MULTIPLIER
                model = models[model_index]
                study.optimize(lambda trial: objective(trial, hparam_list[model_index], model,  
                    train_data= xy_train[study_index], val_data=xy_val[study_index], n_epochs=n_epochs, device=device, patience=PATIENCE, min_delta=MIN_DELTA), n_trials=n_trials)
            with open('../hyperopt/results/NK_hyperopt_results.pkl', 'wb') as handle: #rewrite file as you go for each study 
                pickle.dump(studies, handle,protocol=pickle.HIGHEST_PROTOCOL )
        
        t2 = time.time()
        times[model_name]=(t2-t1)





    with open('../hyperopt/results/NK_hyperopt_results_times.pkl', 'wb') as handle: 
        pickle.dump(times, handle,protocol=pickle.HIGHEST_PROTOCOL )

    for model_name in times.keys(): 
        t = times[model_name]
        mins = t/60
        print('Model:{} took {} mins to optimise hyperparameters'.format(model_name, mins) )
    
    print('Hyperparameter optimisation complete.')


if __name__ == "__main__":
    run_hparam_opt()






    

    
    



    




