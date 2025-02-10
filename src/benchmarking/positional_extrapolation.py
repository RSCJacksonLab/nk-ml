''''
Function for positional extrapolation testing
---------------------------------------------
Definition of positional extrapolation: tasking the model with predicting
the effect of a mutation at a position has not been altered in the training set.

In positional extrapolation, the model is tasked with predicting the effect of mutations 
at sequence positions that are never modified in the training data.

Ensure test data includes mutations at sites where no variation is
observed in the training data.

Modification of code from https://github.com/acmater/NK_Benchmarking/
'''

import inspect
import numpy as np
import pandas as pd
import pickle as pkl

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from torch.utils.data import DataLoader
from typing import Optional

from modelling import (
    architectures, 
    make_dataset, 
    score_sklearn_model)

def positional_extrapolation_test(model_dict: dict,
                                  landscape_dict: dict,
                                  sequence_len: int,
                                  alphabet_size: int,
                                  split: float = 0.8,
                                  cross_validation: int = 1,
                                  save: bool = True,
                                  file_name: Optional[str] = None,
                                  directory: str = "results/", 
                                  n_epochs: int = 30, 
                                  patience: int = 5, 
                                  min_delta: float = 1e-5, 
                                  inclusive: bool = True):
    """
    Positional extrapolation function that takes a dictionary of models and a
    landscape dictionary and iterates over all models and landscapes,
    recording results, before finally (saving) and returning them.

    Parameters:
    -----------
    model_dict : dict
        Dictionary of model architectures. Format:
        {sklearn.model : **kwargs}.

    landscape_dict : dict
        Dictionary of protein landscapes. Format:
        {Name : [Protein_Landscape()]}.

    sequence_len : int
        Length of sequences in landscape. 

    alphabet_size : int
        Number of AAs in the alphabet.

    split : float, default=0.8, Allowed values: 0 < split < 1
        The split point used to partition the data.

    cross_validation : int, default=1
        The number of times to randomly resample the dataset, typically
        used with experimental datasets.

    save : Bool, default=True
        Boolean value used to determine whether or not the file will be
        saved.

    file_name : str, default=None
        File name to use if saving file. If none is provided, user will
        be prompted for one.

    directory : str, default="Results/"
        Directory is the directory to which the results will be saved.
    """

    # get the model names 
    first_key = list(model_dict.keys())[0]
    model_names = list(model_dict[first_key].keys())

    complete_results = {
        model: {key: 0 for key in landscape_dict.keys()} 
        for model in model_names
    }
    # iterate over model types
    for model_name in model_names: 
        print(f'Working on model: {model_name}')

        # iterate over each landscape
        for landscape_name in landscape_dict.keys():
            print(f'Working on landscape: {landscape_name}')

            #extract model hparams -- model hparams are landscape-specific 
            model_hparams = model_dict[landscape_name][model_name]

            # add dataset properties to hparams
            model_hparams["input_dim"] = alphabet_size
            model_hparams["sequence_length"] = sequence_len

            results = {}

            # iterate over each instance of the landscape
            for idx, instance in enumerate(landscape_dict[landscape_name]):

                print(
                    f'Working on instance {idx} of landscape {landscape_name}'
                )
                
                # update result dict
                if not instance in results.keys():
                    results[instance] = {}

                landscape_instance = landscape_dict[landscape_name][instance]

                # obtain positions that are mutated with reference to the seed sequence
                # note that positions uses 0 indexing, so the first position will be 0
                positions = landscape_instance.mutated_positions
                wt_sequence = landscape_instance.seed_seq

                # get all data for splitting
                x_data = np.stack(landscape_instance.ohe)
                sequence_data = np.array([
                    list(seq) 
                    for seq in landscape_instance.sequences
                ])
                y_data = landscape_instance.fitnesses.astype(np.float32)

                # for each site with mutations present
                for pos in positions:

                    if not pos.item() in results[instance]:
                        results[instance][pos.item()] = {}

                    # determine WT AA at site
                    wt_aa_at_pos = wt_sequence[pos]
                    
                    # determine alternate AAs at site for testing
                    alt_aas_at_pos = list(
                        set(sequence_data[:, pos]) - {wt_aa_at_pos}
                    )

                    # get data with position fixed to WT
                    trn_idx = np.where(sequence_data[:, pos] == wt_aa_at_pos)[0]

                    x_trn = x_data[trn_idx]
                    y_trn = y_data[trn_idx]

                    # get models and train
                    if model_name not in ["gb", "rf"]:
                        loaded_model = architectures.NeuralNetworkRegression(
                            model_name,
                            **model_hparams
                        )
                        # train model
                        print('Fitting model')
                        loaded_model.fit((x_trn,
                                          y_trn),
                                          n_epochs=n_epochs, 
                                          patience=patience, 
                                          min_delta=min_delta 
                                         )

                        # score model
                        print('Scoring model')
                        train_dset = make_dataset(
                            (x_trn, y_trn)
                        )
                        train_dloader = DataLoader(train_dset, batch_size=2048)
                        score_train = loaded_model.score(
                            train_dloader
                        )
                        results[instance][pos.item()]["train"] = score_train

                        for alt_aa in alt_aas_at_pos:
                            print(f'Testing {model_name} on {alt_aa} at site {pos}')
                            test_idx = np.where(sequence_data[:, pos] == alt_aa)[0]
                            x_tst = x_data[test_idx]
                            y_tst = y_data[test_idx]

                            test_dset = make_dataset(
                                (x_tst, y_tst)
                            )
                            test_dloader = DataLoader(test_dset, batch_size=2048)
                            results[instance][pos.item()][alt_aa.item()] = loaded_model.score(
                                test_dloader,
                            )
                    else:
                        # flatten input data
                        x_trn = [
                            i.flatten().reshape(-1, 1) 
                            for i in x_trn
                            ]
                        x_trn = np.concatenate(
                            x_trn, 
                            axis=1
                        ).T
                        # set model class
                        if model_name == "rf":
                            model_class = RandomForestRegressor

                        elif model_name == "gb":
                            model_class = GradientBoostingRegressor
                        else:
                            print(f"Model {model_name} not known.")
                            continue
                        model_kwargs = inspect.signature(model_class)
                        kwargs_filtered = {
                            hparam: value 
                            for hparam, value in model_hparams.items()
                            if hparam in model_kwargs.parameters
                        }
                        if model_name == "rf": 
                            kwargs_filtered['n_jobs']=-1

                        loaded_model = model_class(
                            **kwargs_filtered
                        )
                        # train model
                        print('Fitting model')
                        loaded_model.fit(x_trn, y_trn)
                        results[instance][pos.item()]["train"] = score_sklearn_model(
                            loaded_model,
                            x_trn,
                            y_trn
                        )
                        
                        for alt_aa in alt_aas_at_pos:
                            print(f'Testing {model_name} on {alt_aa} at site {pos}')
                            test_idx = np.where(sequence_data[:, pos] == alt_aa)
                            x_tst = x_data[test_idx]
                            y_tst = y_data[test_idx]

                            x_tst = [
                                i.flatten().reshape(-1, 1) 
                                for i in x_tst
                                ]
                            x_tst = np.concatenate(
                                x_tst, 
                                axis=1
                            ).T
                            test_dset = make_dataset(
                                (x_tst, y_tst)
                            )
                            results[instance][pos.item()][alt_aa.item()] = score_sklearn_model(
                                loaded_model,
                                x_tst,
                                y_tst
                            )
            complete_results[model_name][landscape_name] = results

                # # cross-fold eval
                # for fold in range(cross_validation):

                #     train_datasets = []
                #     test_datasets = []

                #     # for each position make test/train splits
                #     for pos_idx in range(len(positions)):

                #         actual_pos = int(positions[pos_idx])

                #         if not pos_idx in results[instance].keys():
                #             results[instance][actual_pos] = {}

                #         if not fold in results[instance][actual_pos].keys():
                #             results[instance][actual_pos][fold] = {}
                        

                #         x_trn, y_trn, x_tst, y_tst = landscape_instance.sklearn_data(
                #             split=split,
                #             positions=positions[:pos_idx + 1], 
                #             random_state=fold,
                #             convert_to_ohe=True, 
                #             flatten_ohe=False
                #         )
                #         train_datasets.append([x_trn, y_trn])
                #         test_datasets.append([x_tst, y_tst])

                #     #this loop segregates the test data so x_p3 ONLY includes positions varying at 3 pso
                #     if not inclusive:                         
                #         exclusive_trn_dsets = [] #non-inclusive train datasets
                        
                #         for j_pos_idx, j_pos_tst_dset in enumerate(test_datasets): 
                #             #pos_x_tst = j_pos_dset[0]
                            
                #             if j_pos_idx>0:
                #                 x_tst_prev_all = [i[0] for i in exclusive_trn_dsets]
                #                 x_tst_prev_all = [i for j in x_tst_prev_all for i in j]
                                
                #                 x_tst_current = [i for i in j_pos_tst_dset[0]]
                #                 y_tst_current = j_pos_tst_dset[1]
                
                #                 #find indices of the current position trn_dset that are not in the previous position trn_dset
                #                 indices = [i for i, arr2 in enumerate(x_tst_current) if 
                #                            not any(np.array_equal(arr2, arr1) for arr1 in x_tst_prev_all)]
                                
                #                 x_tst_current_exclusive = j_pos_tst_dset[0][indices]
                #                 y_tst_current_exclusive = j_pos_tst_dset[1][indices]
                #                 continue                                

                #             else: 
                #                 exclusive_trn_dsets.append(j_pos_tst_dset)
                                
                            



                    # train_datasets are composed [[x_p1, y_p1],[x_p2, y_p2], ..., n_positions],
                    # where x_p1 is a list of np arrays consisting of train examples from only 
                    # the first position mutated, x_p2 the first AND second positions mutated, and so on
                    # 
                    # test_datasets are composed [[x_p1, y_p1],[x_p2, y_p2], ..., n_positions],
                    # where x_p1 is a list of np arrays consisting of test examples from only 
                    # the first position mutated, x_p2 the first AND second positions mutated, and so on
                    #
                    # A 80/20 train/test ratio is used at each mutational position by default (can be changed). 
                    # Train and test datasets are inclusive of the 'seed' sequence. 
                    #
                    # in the below train/test loop, models are iteratively trained on each element
                    # of train_datasets i.e. on [x_p1, y_p1], [xp2, y_p2]..etc BUT tested on EVERY
                    # element on test_datasets in each loop, permitting testing of positional 
                    # extrapolation 

            #         # for each test/train split - train and test models
            #         for pos_idx in range(len(positions)):
                        
            #             actual_pos = int(positions[pos_idx]) #get the actual position 
                        
            #             pos_idx += 1 # add 1 for python list slicing below
            #                          # so, for pos_idx 0, the below slice will include the value at 0th index
            #             x_training = collapse_concat(
            #                 [x[0] for x in train_datasets[:pos_idx]]
            #             )
            #             y_training = collapse_concat(
            #                 [x[1] for x in train_datasets[:pos_idx]]
            #             )

            #             if model_name not in ["gb", "rf", "linear", "mlp", 
            #                                   "blstm", "ulstm", "transformer"]:

            #                 loaded_model = architectures.NeuralNetworkRegression(
            #                     model_name,
            #                     **model_hparams
            #                 )


            #                 # train model
            #                 loaded_model.fit((x_training, 
            #                                  y_training), 
            #                                  n_epochs=n_epochs, 
            #                                  patience=patience, 
            #                                  min_delta=min_delta)
            #                 print(
            #                     f"{model_name} trained on dataset "
            #                     f"{landscape_name} positions "
            #                     f"{positions[:pos_idx]}"
            #                 )
    
            #                 # score model
            #                 train_dset = make_dataset(
            #                     (x_training, y_training)
            #                 )
            #                 train_dloader = DataLoader(train_dset, 
            #                                            batch_size=2048)

            #                 score_train = loaded_model.score(train_dloader)
            #                 score_train["train_epochs"] = loaded_model.actual_epochs

            #                 score = {
            #                     'train': score_train,
            #                 }
            #                 # score on different distance test sets
            #                 for t_pos_idx, pos_dset in enumerate(test_datasets):
            #                     x_tst = pos_dset[0]
            #                     y_tst = pos_dset[1]




            #                     test_dset = make_dataset(
            #                         (x_tst, y_tst)
            #                     )
            #                     test_dloader = DataLoader(test_dset, 
            #                                               batch_size=2048)
            #                     score_test = loaded_model.score(
            #                         test_dloader,
            #                     )

            #                     print(f'Model {model_name} trained on pos {positions[:pos_idx]}')
            #                     print('score on test position {positions[t_pos_idx]} is {score_test}')
                                
            #                     score[f"test_pos{positions[t_pos_idx]}"] = score_test


            #             else:

            #                 # flatten input data 
            #                 x_training = [
            #                     i.flatten().reshape(-1, 1) 
            #                     for i in x_training
            #                     ]
            #                 x_training = np.concatenate(
            #                     x_training, 
            #                     axis=1
            #                 ).T

            #                 # set model class
            #                 if model_name == "rf":
            #                     model_class = RandomForestRegressor

            #                 elif model_name == "gb":
            #                     model_class = GradientBoostingRegressor
            #                 else:
            #                     print(f"Model {model_name} not known.")
            #                     continue
                        
            #                 # apply hyperparams 
            #                 model_kwargs = inspect.signature(model_class)
            #                 kwargs_filtered = {
            #                     hparam: value 
            #                     for hparam, value in model_hparams.items()
            #                     if hparam in model_kwargs.parameters
            #                 }

            #                 if model_name == "rf": 
            #                     kwargs_filtered['n_jobs']=-1
                            
            #                 #initialise model with hyperparams
            #                 loaded_model = model_class(
            #                     **kwargs_filtered
            #                 )
            #                 # train model on appropriate positional training data
            #                 loaded_model.fit(x_training, y_training)

            #                 print(
            #                     f"{model_name} trained on dataset"
            #                     f" {landscape_name} positions "
            #                     f"{positions[:pos_idx]}."
            #                 )

            #                 # get model performance
            #                 train_score = score_sklearn_model(
            #                     loaded_model,
            #                     x_training,
            #                     y_training
            #                 )
            #                 score = {
            #                     'train': train_score
            #                 }
                            
            #                 # get model performance on data greater than distance
            #                 for pos_idx, pos_dset in enumerate(test_datasets):
            #                     x_tst = pos_dset[0]
            #                     y_tst = pos_dset[1]

            #                     # flatten x_test
            #                     x_tst = [
            #                         i.flatten().reshape(-1, 1) 
            #                         for i in x_tst
            #                         ]
            #                     x_tst = np.concatenate(
            #                         x_tst, 
            #                         axis=1
            #                     ).T  
                                
            #                     # make dataset and get performance
            #                     score_test = score_sklearn_model(
            #                         loaded_model,
            #                         x_tst,
            #                         y_tst,
            #                     )
            #                     print(f'Model {model_name} score on test position {positions[pos_idx]} is {score_test}')
            #                     score[f"test_pos{positions[pos_idx]}"] = score_test

            #             results[instance][actual_pos][fold] = score

            # complete_results[model_name][landscape_name] = results

    if save:
        if not file_name:
            file_name = input(
                "What name would you like to save results with?"
            )
        file = open(directory + file_name + ".pkl", "wb")
        pkl.dump(complete_results,file)
        file.close()

# save csv
        # Prepare a list to hold rows for the DataFrame
        rows = []

        # Iterate through the nested dictionary structure
        for model, landscapes in complete_results.items():
            for landscape, replicates in landscapes.items():
                for replicate, positions in replicates.items():
                    for pos, splits in positions.items():
                        for data_split, metrics in splits.items():
                            # Append a row with the relevant data
                            rows.append({
                                "model": model,
                                "landscape": landscape,
                                "replicate": replicate,
                                "train position": pos,
                                "data_split": data_split,
                                "pearson_r": metrics.get("pearson_r", None),
                                "r2": metrics.get("r2", None),
                                "mse_loss": metrics.get("mse_loss", None), 
                                "train_epochs": metrics.get("train_epochs", None)
                            })

        # Create a DataFrame from the rows
        df = pd.DataFrame(rows)
        df.to_csv(directory + file_name + ".csv", index=False)

    return complete_results



## debugging 
# import os
# from benchmarking.file_proc import make_landscape_data_dicts

# # load yamls for hparams
# hopt_dir =  os.path.abspath("./hyperopt/ohe/nk_landscape_hparams") # hyperparameter directory
# data_dir =  os.path.abspath("./data/nk_landscapes/") # data directory with NK landscape data



# model_dict, data_dict = make_landscape_data_dicts(
#     data_dir,
#     hopt_dir,
#     alphabet='ACDEFG'
# )

# positional_extrapolation_test(model_dict=model_dict, 
#                  landscape_dict=data_dict,
#                  sequence_len=6,
#                  alphabet_size=len("ACDEFG"),
#                  split=0.8,
#                  cross_validation=5,
#                  )