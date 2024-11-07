import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
sys.path.append('../../pscapes')
sys.path.append('../../nk-ml-2024/')

from torch.utils.data import DataLoader
from pscapes.landscape_class import ProteinLandscape
from pscapes.utils import dict_to_np_array, np_array_to_dict
from src.architectures import SequenceRegressionCNN, SequenceRegressionLSTM, SequenceRegressionMLP, SequenceRegressionLinear, SequenceRegressionTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from src.ml_utils import train_val_test_split_ohe, train_model, landscapes_ohe_to_numpy

import pickle
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr


def read_CNN_hparams(best_params): 
    params = best_params
    num_conv_layers = params['num_conv_layers']
    n_kernels       = [int(params['n_kernels_layer{}'.format(i)]) for i in range(num_conv_layers)]
    kernel_sizes    = [int(params['kernel_size_layer{}'.format(i)]) for i in range(num_conv_layers)]

    param_dict = {'num_conv_layers':num_conv_layers, 'n_kernels': n_kernels, 'kernel_sizes':kernel_sizes}
    return param_dict

def read_MLP_hparams(best_params): 
    params=best_params
    n_hidden_layers = params['n_hidden_layers']    
    hidden_sizes = [params['hidden{}_size'.format(i)] for i in range(n_hidden_layers)]
    param_dict  = {'hidden_sizes': hidden_sizes}
    return param_dict

def read_LSTM_hparams(best_params): 
    params = best_params
    num_layers  = params['num_layers']
    hidden_size = params['hidden_size']
    param_dict = {'num_layers': num_layers, 'hidden_size': hidden_size}
    return param_dict
def read_transformer_hparams(best_params): 
    params = best_params
    d_model = params['embed_dim_num_heads'][0]
    nheads  = params['embed_dim_num_heads'][1]
    num_layers = params['num_layers']
    dim_feedforward = params['dim_feedforward']
    max_seq_length = params['max_seq_length']
    param_dict = {'d_model': d_model, 'nhead': nheads, 'num_layers':num_layers,'dim_feedforward':dim_feedforward, 
                 'max_seq_length': max_seq_length}
    return param_dict
    
    
def train_models_from_hparams_NK(hparams_path, datapath, model_savepath, result_path, amino_acids, seq_length, 
                                 n_replicates, n_epochs=30, patience=5, min_delta=1e-5):
    """
    Performs training of a given model with a given set of hyperparameters against NK landscapes replicates. 

    model (architectures.py object): model to train. Do not instantiate model with model() before passing. 
    
    hparams_path (str):              path to pickled list of optuna studies containing results of hyperparameter 
                                     optimisation, with each list element containing the optuna study for the k-th NK landscape
    datapath (str):                  path to csv files of nk landscapes, formatted as k{}_r{}.csv
    amino_acids (str):               alphabet of amino acids used to create NK landscapes 
    seq_length (int):                length of sequences in NK landscapes 
    n_replicates (int):              number of replicate NK landscapes to load for each K value
    
    """
    

    with open(hparams_path, 'rb') as handle:
        hparam_studies = pickle.load(handle)

    

        
    landscapes = []
    print('Loading landscapes.')
    for k in range(seq_length):
        replicate_list = []
        for r in range(n_replicates):
            landscape = ProteinLandscape(csv_path=datapath+'/k{0}_r{1}.csv'.format(k,r), amino_acids=amino_acids)
            replicate_list.append(landscape)
        landscapes.append(replicate_list)
    landscapes = [[i.fit_OHE() for i in j] for j in landscapes]

    print('Calculating train-test-val splits')
    splits = [train_val_test_split_ohe(i, random_state=1) for i in landscapes] #this will split each landscape replicate in each K 
                                                               #sublist into train-test-val splits
    
    #initialise some lists to collect results

    results = {}

    device = 'cuda' if torch.cuda.is_available else 'cpu'

    
    #loop over models 
    for model_name in hparam_studies.keys(): 
        print('Working on model name {}'.format(model_name))
 
                
        studies  = hparam_studies[model_name]
        assert len(landscapes) == len(studies), 'Number of K values does not match number of hyper-parameter studies.'

        results[model_name] = {x:None for x in range(len(studies))}

        for k_index, study in enumerate(studies): #we loop over studies under each model, which should be equal to K values 
            print('Working on training for K = {}'.format(k_index))
            k_results = {'test_r2':[], 'test_mse':[], 'test_pearson_r':[], 'train_epoch_losses':[], 'val_epoch_losses': [], 'predictions': [],'ground_truth':[]}
            params = study.best_params


        
            
            #extract model agnostic hparams. These don't apply to sklearn models 
            if model_name!='RF' and model_name!='GB':             
                lr              = params['lr']
                batch_size      = params['batch_size']
                
            #extract model hparams 
            if model_name == 'linear':
                model_params = {'alphabet_size': len(amino_acids), 'sequence_length':seq_length}
            elif model_name == 'mlp':
                model_params = read_MLP_hparams(params) 
                model_params['alphabet_size']= len(amino_acids)
                model_params['sequence_length']=seq_length
            elif model_name == 'cnn': 
                model_params = read_CNN_hparams(params)
                model_params['input_channels'] = len(amino_acids)
                model_params['sequence_length']=seq_length
            elif model_name == 'ulstm': 
                model_params = read_LSTM_hparams(params)
                model_params['input_size'] = len(amino_acids)
                model_params['bidirectional'] = False 
            elif model_name == 'blstm': 
                model_params = read_LSTM_hparams(params)
                model_params['input_size'] = len(amino_acids)
                model_params['bidirectional'] = True 
            elif model_name == 'transformer': 
                model_params = read_transformer_hparams(params)
                model_params['input_dim']=len(amino_acids)
            elif (model_name == 'RF') or (model_name == 'GB') :
                model_params=params

            #construct Dataloaders by access appropriate data. Note that splits is structured as 
            #[[K1_repl1, ...K1_repli], ..., [Kj_repl1, ..., Kj_repli]]. #Therefore, xy_train etc will be a list 
            #where each element contains data for a replicate landscape
            #such that xy_train = [repl1, repl2, ..., repli]
            print ('Hyperparameters: {}, learning_rate: {}, batch_size: {}'.format(model_params, lr, batch_size))
            
            landscapes_ohe, xy_train, xy_val, xy_test, x_tests, y_tests = splits[k_index] 
            x_train_np, y_train_np = landscapes_ohe_to_numpy(xy_train) #intialise flattened np arrays for RF and GB training 
            x_val_np, y_val_np = landscapes_ohe_to_numpy(xy_val)
            x_test_np, y_test_np = landscapes_ohe_to_numpy(xy_test)

            
            #print('model name: {}, model_params: {}, lr: {}, batch_size:{}'.format(model_name, model_params, lr, batch_size))
            for replicate_index, replicate in enumerate(landscapes[k_index]): 
                print('Training model {},  K= {}, replicate {}'.format(model_name, k_index, replicate_index))
                if model_name !='RF' and model_name!='GB':   #different setup for sklearn models                 
            
                    pass
                    #initialise model
                    if model_name=='linear': 
                        model_instance = SequenceRegressionLinear(**model_params)
                    elif model_name=='mlp': 
                        model_instance = SequenceRegressionMLP(**model_params)
                    elif model_name=='cnn': 
                        model_instance = SequenceRegressionCNN(**model_params)
                    elif model_name=='ulstm':
                        model_instance = SequenceRegressionLSTM(**model_params)
                    elif model_name=='blstm': 
                        model_instance = SequenceRegressionLSTM(**model_params)
                    elif model_name=='transformer': 
                        model_instance = SequenceRegressionTransformer(**model_params)

                    #initialise loss and optimizer
                    loss_fn   = nn.MSELoss()
                    optimizer = optim.Adam(model_instance.parameters(), lr=lr)
                
                    #load dataloaders
                    train_dataloader = DataLoader(xy_train[replicate_index], batch_size=batch_size, shuffle=True)
                    val_dataloader   = DataLoader(xy_val[replicate_index], batch_size=batch_size)
            
                    #train model
                    _, train_epoch_losses, val_epoch_losses = train_model(model_instance, optimizer, loss_fn, 
                                                                          train_dataloader, 
                                                                          val_dataloader,  n_epochs=n_epochs, device=device,
                                                                          patience=patience, min_delta=min_delta)
                    #save model 
                    savepath = model_savepath+'/{0}_NK_k{1}_r{2}.pt'.format(model_name, k_index, replicate_index)
                    torch.save(model_instance.state_dict(), savepath)

                    #evaluate model on test data
                    model_instance.eval()
        
                    x_test   = x_tests[replicate_index].to(device)
                    y_test   = y_tests[replicate_index].to(device) 
                    
                    y_pred   = model_instance(x_test) #get predictions
                    y_pred   = y_pred.detach() #detach tensors 
                    y_test   = y_test.detach()
                    
                    test_mse = loss_fn(y_pred, y_test) #MSE 

                    y_pred, y_test   = y_pred.cpu(), y_test.cpu() #send to cpu for scipy pearson and sklearn r2
                    
                    test_pearson_r = pearsonr(y_pred, y_test) #Pearson rho                       
                                              
                    test_r2  = r2_score(y_pred, y_test) #R^2

                    

                    #collect data 
                    k_results['test_r2'].append(test_r2)
                    k_results['test_mse'].append(test_mse)
                    k_results['test_pearson_r'].append(test_pearson_r)
                    k_results['train_epoch_losses'].append(train_epoch_losses)
                    k_results['val_epoch_losses'].append(val_epoch_losses)
                    k_results['predictions'].append(y_pred)
                    k_results['ground_truth'].append(y_test)
                else: #now we deal with RF and GB 
                    if model_name == 'RF': 
                        model_instance = RandomForestRegressor(**model_params, n_jobs=-1)
                    elif model_name == 'GB': 
                        model_instance = GradientBoostingRegressor(**model_params)
                    model_instance.fit(x_train_np[replicate_index], y_train_np[replicate_index].ravel())
                    y_pred   = model_instance.predict(x_test_np[replicate_index])
                    y_test   = y_test_np[replicate_index].ravel()

                    test_mse  = mean_squared_error(y_test, y_pred)
                    test_r2   =  r2_score(y_test, y_pred)
                    test_pearson_r = pearsonr(y_test, y_pred)

                    
                    k_results['test_r2'].append(test_r2)
                    k_results['test_mse'].append(test_mse)
                    k_results['test_pearson_r'].append(test_pearson_r)
                    
                
                
            #update results dictionary 
            results[model_name][k_index] = k_results 
    with open(result_path+'/NK_train_test_results.pkl', 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    
    print('Training and testing finished. Results written to disk.')
    return results      

                    




            
        
    
"""
    
    for k_val, k_replicates in enumerate(landscapes): 
        print('Training models for K={}'.format(k_val))
        #initialise correct hyperparameters
        params          = hparam_studies[k_val].best_params

        print('Models for this value of K will have the following hyperparameters:{}'.format(params))
        #extract hparams not passed to model 
        lr              = params['lr']
        batch_size      = params['batch_size']

        #extract hparams passed to model 
        if model==SequenceRegressionCNN: #here SequenceRegressionCNN is hardcoded
            model_params = read_CNN_hparams(params) #we need this special function to correctly parse hparams as stored in optuna study to those we can pass to our CNN model
            model_params['input_channels']=len(amino_acids)
            model_params['sequence_length']=seq_length
            model_name = 'CNN'
        else: 
            break #placeholder while developing

        #access appropriate data. Note that splits is structured as 
        #[[K1_repl1, ...K1_repli], ..., [Kj_repl1, ..., Kj_repli]].
        #Therefore, xy_train etc will be a list where each element contains data for a replicate landscape
        #such that xy_train = [repl1, repl2, ..., repli]
        landscapes_ohe, xy_train, xy_val, xy_test, x_tests, y_tests = splits[k_val]  

        train_epoch_losses_list_k = []
        val_epoch_losses_list_k   = []
        r2_scores_k               = []
        predict_vs_gt_k           = []
        
        #now loop over replicates at a given K value
        for r_index, replicate in enumerate(k_replicates):          

            print('Training model K= {}, replicate {}'.format(k_val, r_index))
            #define model, loss fn and optimiser 
            modelr     = model(**model_params)
            loss_fn   = nn.MSELoss()
            optimizer = optim.Adam(modelr.parameters(), lr=lr)

            
            
             #load dataloaders
            train_dataloader = DataLoader(xy_train[r_index], batch_size=batch_size, shuffle=True)
            val_dataloader   = DataLoader(xy_val[r_index], batch_size=batch_size)

            #train model
            _, train_epoch_losses, val_epoch_losses = train_model(modelr, optimizer, loss_fn, train_dataloader, val_dataloader, 
                                                                     n_epochs=n_epochs, device=device, patience=patience, min_delta=min_delta)

            #save model 
            savepath = model_savepath+'/{0}_k{1}_r{2}.pt'.format(model_name, k_val, r_index)
            torch.save(modelr.state_dict(), savepath)

            #evaluate model on test data
            modelr.eval()

            x_test = x_tests[r_index].to(device)
            
            y_pred  = modelr(x_test)
            y_pred  = y_pred.cpu().detach()
            y_test  = y_tests[r_index].detach()

            r2      = r2_score(y_pred, y_test)

            #append relevant lists 
            train_epoch_losses_list_k.append(train_epoch_losses)
            val_epoch_losses_list_k.append(val_epoch_losses)
            r2_scores_k.append(r2)
            predict_vs_gt_k.append([x_test.cpu().detach().numpy(), y_pred.numpy(), y_test.numpy()])
        
        
        #outer loop appends
        train_epoch_losses_list.append(train_epoch_losses_list_k)
        val_epoch_losses_list.append(val_epoch_losses_list_k)
        r2_scores.append(r2_scores_k)
        predict_vs_gt.append(predict_vs_gt_k)

    
    results = (train_epoch_losses_list, val_epoch_losses_list, r2_scores, predict_vs_gt)

    with open(result_path+'/{}_train_test_results.pkl'.format(model_name), 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)    

    print('Training and testing finished. Results written to disk.')
    return results      
"""            

        

        

    
