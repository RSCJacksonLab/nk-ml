import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
sys.path.append('/home/ubuntu/nk-paper-2024/pscapes')
sys.path.append('/home/ubuntu/nk-paper-2024/nk-ml-2024')
from torch.utils.data import DataLoader
from pscapes.landscape_class import ProteinLandscape
from pscapes.utils import dict_to_np_array, np_array_to_dict
from src.architectures.architectures import SequenceRegressionCNN, SequenceRegressionLSTM, SequenceRegressionMLP, SequenceRegressionLinear, SequenceRegressionTransformer
from src.architectures.ml_utils import train_val_test_split_ohe, train_model
import pickle
from sklearn.metrics import r2_score


def read_CNN_hparams(CNN_optuna_study): 
    params = CNN_optuna_study
    num_conv_layers = params['num_conv_layers']
    n_kernels       = [params['n_kernels_layer{}'.format(i)] for i in range(num_conv_layers)]
    kernel_sizes    = [params['kernel_size_layer{}'.format(i)] for i in range(num_conv_layers)]

    param_dict = {'num_conv_layers':num_conv_layers, 'n_kernels': n_kernels, 'kernel_sizes':kernel_sizes}
    return param_dict
    


def train_models_from_hparams_NK(model, hparams_path, datapath, model_savepath, result_path, amino_acids, seq_length, 
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
    splits = [train_val_test_split_ohe(i) for i in landscapes] #this will split each landscape replicate in each K 
                                                               #sublist into train-test-val splits
    

    #initialise some lists to collect results
    train_epoch_losses_list = []
    val_epoch_losses_list   = []
    r2_scores               = []
    predict_vs_gt           = []

    assert len(landscapes)==len(hparam_studies), 'Number of K values does not match number of hyper-parameter studies.'

    device = 'cuda' if torch.cuda.is_available else 'cpu'
    #loop over K values 
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
            model_params = read_CNN_hparams(params) #we need this special function to correctly parse hparams as stored in optuna study to those  we can pass to our CNN model
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
            

        

    
