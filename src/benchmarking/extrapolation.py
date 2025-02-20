''''
Function for extrapolation testing
----------------------------------
Based on distance from seed sequence, holdout distant sequences as a
test set.

Modification of code from https://github.com/acmater/NK_Benchmarking/
* Deterministic splits during cross-fold testing
'''

import inspect
import numpy as np
import pandas as pd
import pickle as pkl

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from typing import Optional
from torch.utils.data import DataLoader

from modelling import (
    architectures, 
    collapse_concat, 
    make_dataset, 
    score_sklearn_model
)


def extrapolation_test(model_dict: dict,
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
                       min_delta: float = 1e-5):
    """
    Extrapolation function that takes a dictionary of models and a
    landscape dictionary and iterates over all models and landscapes,
    recording results, before finally (saving) and returning them.

    Parameters
    ----------
    model_dict : dict
        Dictionary of model architectures. Format: 
        {landscape_name: {model_name : **kwargs}}.

    landscape_dict : dict
        Dictionary of protein landscapes. Format: 
        {landscape_name: [datafile_name: ProteinLandscape]}

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
        print('Working on model: {}'.format(model_name))

        # iterate over each landscape
        for landscape_name in landscape_dict.keys():
            print('Working on landscape: {}'.format(landscape_name))

            #extract model hparams
            model_hparams = model_dict[landscape_name][model_name]

            # add dataset properties to hparams
            model_hparams["input_dim"] = alphabet_size
            model_hparams["sequence_length"] = sequence_len

            results = {}

            # iterate over each instance of each landscape
            for idx, instance in enumerate(landscape_dict[landscape_name]):

                # update result dict
                if not instance in results.keys():
                    results[instance] = {}

                print(
                    f'Working on instance {idx} of landscape {landscape_name}'
                )

                # get distance data from landscape
                landscape_instance = landscape_dict[landscape_name][instance]
                distances = landscape_instance.d_data.keys()

                # deletes zero if it listed as a distance
                distances = [d for d in distances if d] 
                # cross fold eval
                for fold in range(cross_validation):

                    print('Working on cross-validation fold: {}'.format(fold))

                    train_datasets = []
                    test_datasets = []

                    print('Creating distance-based train and test datasets')
                    for d in distances:

                        if not f"train distance {d}" in results[instance].keys():
                            results[instance][f"train distance {d}"] = {} 

                        x_trn, y_trn, x_tst, y_tst = landscape_instance.sklearn_data(
                            split=split,
                            distance=d,
                            random_state=fold,
                            convert_to_ohe=True,
                            flatten_ohe=False
                        )
                        train_datasets.append([x_trn, y_trn])
                        test_datasets.append([x_tst, y_tst])

                    for j, d in enumerate(distances):

                        print(f'Training on distance 1-{d}')
                        # get training and testing data
                        j += 1
                        x_training = collapse_concat(
                            [x[0] for x in train_datasets[:j]]
                        )
                        y_training = collapse_concat(
                            [x[1] for x in train_datasets[:j]]
                        )

                        if model_name not in ["gb", "rf"]:
                            loaded_model = architectures.NeuralNetworkRegression(
                                model_name,
                                **model_hparams
                            )
                            # train model


                            loaded_model.fit((x_training, y_training),
                                             n_epochs=n_epochs, 
                                             patience=patience, 
                                             min_delta=min_delta)
                            print(
                                f"{model_name} trained on Dataset "
                                f"{landscape_name} distances 1-{d}"
                            )

                            # score model
                            train_dset = make_dataset(
                                (x_training, y_training)
                            )
                            train_dloader = DataLoader(train_dset, 
                                                       batch_size=2048)
                            score_train = loaded_model.score(
                                train_dloader
                            )

                            score = {}
                            score["train"] = score_train
                            score["train"]["train_epochs"] = loaded_model.actual_epochs

                            # score on different distance test sets
                            #print(test_datasets[0])
                            for dist_idx, dist_dset in enumerate(test_datasets):
                                print(f'Testing on distance {dist_idx+1}')
                                x_tst = dist_dset[0]
                                y_tst = dist_dset[1]
                              

                                test_dset = make_dataset(
                                    (x_tst, y_tst)
                                )
                                test_dloader = DataLoader(test_dset, 
                                                          batch_size=2048)
                                score_test = loaded_model.score(
                                    test_dloader,
                                )
                                print(f'Score this distance: {score_test}')
                                score[f"test_dist{distances[dist_idx]}"] = score_test
                        else:
                            # flatten input data 
                            x_training = [
                                i.flatten().reshape(-1, 1) 
                                for i in x_training
                                ]
                            x_training = np.concatenate(
                                x_training, 
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

                            # apply hyperparams 
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
                            # train model on data less than distance
                            loaded_model.fit(x_training, y_training)

                            # get model performance
                            train_score = score_sklearn_model(
                                loaded_model,
                                x_training,
                                y_training,
                            )
                            score = {
                                'train': train_score
                            }

                            print(
                                f"{model_name} trained on Dataset "
                                f"{landscape_name} distances 1-{d}"
                            )
                            # get model performance on data greater than distance
                            for dist_idx, dist_dset in enumerate(test_datasets):
                                print(f'Testing on distance {dist_idx+1}')
                                x_tst = dist_dset[0]
                                y_tst = dist_dset[1]

                                # flatten x_test
                                x_tst = [
                                    i.flatten().reshape(-1, 1) 
                                    for i in x_tst
                                    ]
                                x_tst = np.concatenate(
                                    x_tst, 
                                    axis=1
                                ).T  
                                
                                # make dataset and get performance
                                score_test = score_sklearn_model(
                                    loaded_model,
                                    x_tst,
                                    y_tst,
                                )
                                print(f'Score this distance: {score_test}')
                                score[f"test_dist{distances[dist_idx]}"] = score_test

                        results[instance][f"train distance {d}"][fold] = score

            complete_results[model_name][landscape_name] = results

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
                for replicate, distances in replicates.items():
                    for dist_val, cv_folds in distances.items():
                        for cv_fold, splits in cv_folds.items():
                            for data_split, metrics in splits.items():
                                # Append a row with the relevant data
                                rows.append({
                                    "model": model,
                                    "landscape": landscape,
                                    "replicate": replicate,
                                    "train distance": dist_val,
                                    "cv_fold": cv_fold,
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
import os
from benchmarking.file_proc import make_landscape_data_dicts

# load yamls for hparams
hopt_dir =  os.path.abspath("./hyperopt/ohe/nk_landscape_hparams/") # hyperparameter directory
data_dir =  os.path.abspath("./data/nk_landscapes/") # data directory with NK landscape data



model_dict, data_dict = make_landscape_data_dicts(
    data_dir,
    hopt_dir,
    alphabet='ACDEFG'
)

extrapolation_test(model_dict=model_dict, 
                 landscape_dict=data_dict,
                 sequence_len=6,
                 alphabet_size=len("ACDEFG"),
                 split=0.8,
                 cross_validation=5,
                 )