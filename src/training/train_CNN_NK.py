import optuna as opt
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

import sys
sys.path.append('/home/ubuntu/nk-paper-2024/pscapes')
sys.path.append('/home/ubuntu/nk-paper-2024/nk-ml-2024')

from torch.utils.data import DataLoader

from pscapes.landscape_class import ProteinLandscape
from pscapes.utils import dict_to_np_array, np_array_to_dict

from src.architectures.architectures import SequenceRegressionCNN
from src.architectures.ml_utils import train_val_test_split_ohe, train_model
import pickle

from sklearn.metrics import r2_score



def train_CNN():
    hyperopt_path = '/home/ubuntu/nk-paper-2024/nk-ml-2024/hyperopt/'
    datapath      = '/home/ubuntu/nk-paper-2024/nk-ml-2024/data/nk_landscapes/'    

    # read hyperopt results 
    with open('hyperopt_CNN_NK_studies.pkl', 'rb') as handle:
        cnn_studies = pickle.load(handle)

    SEQ_LEN = 6
    AA_ALPHABET = 'ACDEFG'

    LANDSCAPES = []
    for k in range(6): 
        REPLICATES = []
        for r in range(8): 
            landscape = ProteinLandscape(csv_path='../data/nk_landscapes/k{0}_r{1}.csv'.format(k,r), amino_acids=AA_ALPHABET)
            REPLICATES.append(landscape)
        LANDSCAPES.append(REPLICATES)
        
    
    LANDSCAPES = [[i.fit_OHE() for i in j] for j in LANDSCAPES]
    
    landscapes_ohe, xy_train, xy_val, xy_test, x_tests, y_tests = [train_val_test_split_ohe(i) for i in LANDSCAPES]

    
    #loop over K values 
    train_epoch_losses_list   = []
    val_epoch_losses_list   = []
    
    r2_scores = []
    
    predict_vs_gt = []
    
    for index,i in enumerate(cnn_studies):
        print('Training model for K={}'.format(index))

        
        #initialise models with correct parameters from hyperopt
        params          = i.best_params
        num_conv_layers = params['num_conv_layers']
        n_kernels       = [params['n_kernels_layer{}'.format(i)] for i in range(num_conv_layers)]
        kernel_sizes    = [params['kernel_size_layer{}'.format(i)] for i in range(num_conv_layers)]
        lr              = params['lr']
        batch_size      = params['batch_size']
    
        device = 'cuda' if torch.cuda.is_available else 'cpu'

        #loop over replicate values
        for 
        model     = SequenceRegressionCNN(input_channels=len(AA_ALPHABET), sequence_length=SEQ_LEN,
                                     num_conv_layers=num_conv_layers, n_kernels=n_kernels, kernel_sizes=kernel_sizes)
        loss_fn   = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
    
        #load dataloaders
        train_dataloader = DataLoader(xy_train[index], batch_size=batch_size, shuffle=True)
        val_dataloader   = DataLoader(xy_val[index], batch_size=batch_size)
    
    
        #train models
        model, train_epoch_losses, val_epoch_losses = train_model(model, optimizer, loss_fn, train_dataloader, val_dataloader, 
                                                                             n_epochs=40, device=device, patience=8, min_delta=1e-5)
        
        train_epoch_losses_list.append(train_epoch_losses)
        val_epoch_losses_list.append(val_epoch_losses)
    
        #save the model
        savepath = './models_test/CNN_K{}.pt'.format(index)
        torch.save(model.state_dict(), savepath)
    
        #evaluate model on test data
        model.eval()
    
        y_pred  = model(x_tests[index].to(device))
        y_pred  = y_pred.cpu().detach()
        y_test  = y_tests[index].detach()
    
    
        r2      = r2_score(y_pred, y_test) 
    
        r2_scores.append(r2)
    
        predict_vs_gt.append([y_pred, y_test])
        
    
    losses = [train_epoch_losses_list, val_epoch_losses_list, r2_scores]
    
    
    with open('CNN_models_epoch_losses_r2.pkl', 'wb') as handle:
        pickle.dump(losses, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open('CNN_models_predict_vs_gt.pkl', 'wb') as handle:
        pickle.dump(predict_vs_gt, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    
    print('All models trained and saved.')
