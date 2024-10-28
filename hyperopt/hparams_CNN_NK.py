import optuna as opt
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from twilio.rest import Client
import pickle

import time

import sys
sys.path.append('/home/ubuntu/nk-paper-2024/pscapes')
sys.path.append('/home/ubuntu/nk-paper-2024/nk-ml-2024')

from torch.utils.data import DataLoader

from pscapes.landscape_class import ProteinLandscape
from pscapes.utils import dict_to_np_array, np_array_to_dict

from src.architectures.architectures import SequenceRegressionCNN
from src.architectures.ml_utils import train_val_test_split_ohe

from src.hyperopt import EarlyStoppingHyperOpt

torch.backends.nnpack.enabled = False


SEQ_LEN = 6
AA_ALPHABET = 'ACDEFG'


#Load NK landscapes -- only a single replicate for hparam tuning 

def cnn_objective(trial, train_data, val_data, epochs):
    # Define the search space
    num_conv_layers = trial.suggest_int('num_conv_layers', 1, 2)
    
    num_kernels = [int(trial.suggest_categorical("n_kernels_layer{}".format(i), [32, 64, 96, 128]))
                   for i in range(num_conv_layers)]  
    
    kernel_sizes = [int(trial.suggest_categorical("kernel_size_layer{}".format(i), [3,5]))
                   for i in range(num_conv_layers)]
    
    learning_rate = trial.suggest_categorical('lr', [0.01, 0.001, 0.0001])

    batch_size    = trial.suggest_categorical('batch_size', [32, 64, 96, 128])

    print(num_kernels)
    print(kernel_sizes)

    
    # Initialize model with the trial’s hyperparameters
    model = SequenceRegressionCNN(input_channels=len(AA_ALPHABET), sequence_length=SEQ_LEN, 
                                  num_conv_layers=num_conv_layers, n_kernels=num_kernels, kernel_sizes=kernel_sizes)
    
    # Loss and optimizer
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)


    # Early stopping setup
    early_stopping = EarlyStoppingHyperOpt(patience=8, min_delta=1e-5)

    # Training loop with validation loss calculation
    for epoch in range(epochs):


        
        model.to(device)
        model.train()
        for x_batch, y_batch in train_loader:

            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            predictions = model(x_batch)
            loss = loss_fn(predictions, y_batch)
            loss.backward()
            optimizer.step()

        # Calculate validation loss`
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                predictions = model(x_batch)
                val_loss += loss_fn(predictions, y_batch).item()

        val_loss /= len(val_loader)
        
        
        trial.report(val_loss, epoch)

        print('Epoch: {0}, Val loss: {1}'.format(epoch, val_loss))
        # Check early stopping
        if early_stopping.should_stop(val_loss):
            print(f"Early stopping at epoch {epoch}")
            break
        
        
        if trial.should_prune():
            raise opt.TrialPruned()
    print('Best Val Loss this Trial: {}'.format(val_loss))
        

    return val_loss







def main():

    print('Running CNN hyperparameter optimisation on NK dataset')

    print('Loading landscapes...')
    LANDSCAPES = []
    for k in range(6): 
        for r in range(1): 
            landscape = ProteinLandscape(csv_path='../data/nk_landscapes/k{0}_r{1}.csv'.format(k,r), amino_acids=AA_ALPHABET)
            LANDSCAPES.append(landscape)
    
    LANDSCAPES = [i.fit_OHE() for i in LANDSCAPES]
    
    landscapes_ohe, xy_train, xy_val, xy_test, x_test, y_test = train_val_test_split_ohe(LANDSCAPES)
    
    
    # Running the study
    
    # creating studies

    print('Creating studies')
    cnn_studies = [opt.create_study(direction='minimize') for i in LANDSCAPES]
    
    #Loop for hparam opt
    times = [time.time()] 


    print('Running studies...')
    for index, study in enumerate(cnn_studies):
        print('Optimising K={}'.format(index))
        study.optimize(lambda trial: cnn_objective(trial, train_data= xy_train[index], val_data=xy_val[index],
                                                   epochs=50), n_trials=30)
        t = time.time()
        times.append(t)
    
    with open('hyperopt_CNN_NK_studies.pkl', 'wb') as handle:
        pickle.dump(cnn_studies, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('hyperopt_CNN_NK_time.pkl', 'wb') as handle: 
        pickle.dump(times, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print('Studies finished. Output data written to disk: hyperopt_CNN_NK_studies.pkl')

    time_taken = (times[-1]-times[0])/60 

    print(time_taken)
    
    account_sid = "AC20fc2432499d2b13cf65b1357fc8e2d4"
    auth_token  = "d7f074f081834ef1578ed0ec4f8e86b9"
    
    client = Client(account_sid, auth_token)
    
    message = client.messages.create(
        to="+61452413597",
        from_="(240) 839-6571",
        body="CNN HyperOpt Finished, took {} minutes".format(time_taken))


if __name__ == "__main__":
    main()