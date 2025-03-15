''''
Function for extrapolation testing
----------------------------------
Based on distance from seed sequence, holdout distant sequences as a
test set.

Modification of code from https://github.com/acmater/NK_Benchmarking/
* Added cross validation
* Deterministic splits during cross-fold testing
* Refactor for new models
* Addition of hyperparameter tuning for each sequence length
'''

import inspect
import numpy as np
import pandas as pd
import pickle as pkl
import torch
import optuna as opt

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from torch.utils.data import DataLoader
from typing import List, Literal, Optional

from modelling import (
    architectures,
    get_model_hparams,
    make_dataset,
    objective_fn,
    score_sklearn_model,
    sklearn_objective_fn,
)
from modelling.hparam_space import hparam_space_GB1, hparam_space_NK

def length_dependency_test_with_tuning(
        model_ls: list,
        landscape_dict: dict,
        alphabet_size: int,
        tuning_landscape_rep: str,
        hparam_reference: Literal['nk', 'gb1'],
        split: float = 0.8,
        cross_validation: int = 1,
        save: bool = True,
        amino_acids: str='ACDEFGHIKLMNPQRSTVWY',
        seq_lens: List[int] = [10, 50, 100, 200, 250, 500],
        file_name: Optional[str] = None,
        directory: str = "results/",
        n_epochs: int = 10, 
        patience: int = 5, 
        min_delta: float = 1e-5,
        n_trials_multiplier: int = 15,
    ):
    """
    Length testing function that takes a dictionary of models, a
    landscape dictionary, and a list of sequence lengths. It iterates 
    over all of these and leverages the Protein Landscape function that
    enables it to randomly inject length into its sequences to train 
    each model on each of these values.

    Parameters
    ----------
    model_ls : list
        List of model architectures.

    landscape_dict : dict
        Dictionary of protein landscapes. Format: 
        {landscape_name: [datafile_name: ProteinLandscape]}

    alphabet_size : int
        Number of AAs in the alphabet.

    tuning_landscape_rep : str
        Name of replicate to use for hyperparameter tuning.

    hparam_reference :

    split : float, default=0.8, Allowed values: 0 < split < 1
        The split point used to partition the data.

    cross_validation : int, default=1
        The number of times to randomly resample the dataset, typically
        used with experimental datasets.

    save : Bool, default=True
        Boolean value used to determine whether or not the file will be
        saved.

    amino_acids : str, default='ACDEFGHIKLMNPQRSTVWY'
        String containing all allowable amino acids.

    seq_lens : list, default=[10, 50, 100, 200, 250, 500]
        List of sequence lengths, determining how long the extended
        sequences will be.

    file_name : str, default=None
        File name to use if saving file. If none is provided, user will
        be prompted for one.

    directory : str, default="Results/"
        Directory is the directory to which the results will be saved.

    n_trials_multiplier : int, default = 15
        Number of trial multiplier - scales the number of 
        hyperparameters to get the number of trials.
    """

    results = {
        'model': [],
        'landscape': [],
        'replicates': [],
        'tuning_landscape': [],
        'cv_fold': [],
        'sequence_length': [],
        'data_split': [],
        'pearson_r': [],
        'r2': [],
        'mse_loss': [],
    }

    # iterate over each landscape
    
    for landscape_name in landscape_dict.keys():
        print('Working on landscape: {}'.format(landscape_name))

        # iterate over model types
        for model_name in model_ls:
            print('Working on model: {}'.format(model_name))

            model_hparam_dict = {}

            # iterate over each instance of each landscape
            ## set instance to use for tuning first
            landscape_instances = list(landscape_dict[landscape_name].keys())
            if tuning_landscape_rep is not None:
                landscape_instances.remove(tuning_landscape_rep)
                landscape_instances = [tuning_landscape_rep] + landscape_instances

            for idx, instance in enumerate(landscape_instances):

                print(
                    f'Working on instance {idx} of landscape {landscape_name}'
                )

                # get distance data from landscape
                landscape_instance = landscape_dict[landscape_name][instance]

                # cross fold eval
                for fold in range(cross_validation):

                    print('Working on cross-validation fold: {}'.format(fold))

                    # Iterate over each INSTANCE of each landscape, 1 for experimental
                    for length in seq_lens:

                        x_trn, y_trn, x_tst, y_tst = landscape_instance.return_lengthened_data(
                            length,
                            amino_acids=amino_acids,
                            split=split,
                            random_state=fold
                        )
                        if model_name not in ["gb", "rf"]:

                            # hyperparameter tuning
                            tuned_rep = False
                            if not length in model_hparam_dict:
                                tuned_rep = True
                                
                                # get search space
                                if hparam_reference == 'nk':
                                    search_space = hparam_space_NK
                                elif hparam_reference == 'gb1':
                                    search_space = hparam_space_GB1
                                else:
                                    raise KeyError(f"Unknown search space {hparam_ref}.")
                                
                                # Optuna study
                                study = opt.create_study(direction='minimize')
                                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                                n_trials = len(search_space[model_name]) * n_trials_multiplier

                                study.optimize(
                                    lambda trial : objective_fn(
                                        trial,
                                        model_name,
                                        search_space[model_name],
                                        alphabet_size,
                                        length,
                                        (x_trn, y_trn),
                                        (x_tst, y_tst),
                                        n_epochs,
                                        patience,
                                        min_delta,
                                        device
                                    ),
                                    n_trials = n_trials
                                )
                                model_hparams = get_model_hparams(
                                    model_name, 
                                    study.best_params
                                )
                                model_hparams["input_dim"] = alphabet_size
                                model_hparams["sequence_length"] = length

                                model_hparam_dict[length] = model_hparams
        
                            # instantiate on determined hyperparameters
                            model_hparams = model_hparam_dict[length]
                            loaded_model = architectures.NeuralNetworkRegression(
                                model_name,
                                **model_hparams
                            )
                        
                            print(f'Loading model hparams: {model_hparams}')

                            # train model
                            print('Fitting model')
                            loaded_model.fit((x_trn, 
                                            y_trn),
                                            n_epochs=n_epochs, 
                                            patience=patience,
                                            min_delta=min_delta)

                            # score model
                            print('Scoring model')
                            train_dset = make_dataset(
                                (x_trn, y_trn)
                            )

                            # dynamic batch size to accomadate variable sequence lengths
                            batch_size = 1024 if length < 200 else 512

                            train_dloader = DataLoader(train_dset, 
                                                    batch_size=batch_size)
                            score_train = loaded_model.score(
                                train_dloader
                            )
                            test_dset = make_dataset((x_tst, y_tst))
                            test_dloader = DataLoader(test_dset, 
                                                    batch_size=batch_size)
                            score_test = loaded_model.score(
                                test_dloader,
                            )
                            score = {
                                'train': score_train,
                                'test': score_test
                            }
                            
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

                            x_tst = [
                                i.flatten().reshape(-1, 1) 
                                for i in x_tst
                                ]
                            x_tst = np.concatenate(
                                x_tst, 
                                axis=1
                            ).T

                            # hyperparameter tuning
                            tuned_rep = False
                            if not length in model_hparam_dict:
                                tuned_rep = True
                                # get search space
                                if hparam_reference == 'nk':
                                    search_space = hparam_space_NK
                                elif hparam_reference == 'gb1':
                                    search_space = hparam_space_GB1
                                else:
                                    raise KeyError(f"Unknown search space {hparam_ref}.")
                                
                                # Optuna study
                                study = opt.create_study(direction='minimize')
                                n_trials = 3 * n_trials_multiplier

                                study.optimize(
                                    lambda trial : sklearn_objective_fn(
                                        trial,
                                        model_name,
                                        x_train=x_trn,
                                        y_train=y_trn,
                                        x_val=x_tst,
                                        y_val=y_tst,
                                    ),
                                    n_trials = n_trials
                                )
                                model_hparams = get_model_hparams(
                                    model_name, 
                                    study.best_params
                                )
                                model_hparams["input_dim"] = alphabet_size
                                model_hparams["sequence_length"] = length

                                model_hparam_dict[length] = model_hparams
        
                            else:
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
                            loaded_model = model_class(
                                **kwargs_filtered
                            )
                            # train model on ablated data
                            print('Fitting model')
                            loaded_model.fit(x_trn, y_trn)

                            # get model performance
                            print('Scoring model')                            
                            train_score = score_sklearn_model(
                                loaded_model,
                                x_trn,
                                y_trn,
                            )
                            test_score = score_sklearn_model(
                                loaded_model,
                                x_tst,
                                y_tst
                            )
                            score = {
                                "train": train_score,
                                "test": test_score
                            }
                    
                        # save results
                        for test_train_split in score.keys():
                            results['model'].append(model_name)
                            results['landscape'].append(landscape_name)
                            results['replicates'].append(instance)
                            results['tuning_landscape'].append(tuned_rep)
                            results['cv_fold'].append(fold)
                            results['sequence_length'].append(length)
                            results['data_split'].append(test_train_split)
                            results['pearson_r'].append(
                                score[test_train_split].get("pearson_r", None)
                            )
                            results['r2'].append(
                                score[test_train_split].get("r2", None)
                            )
                            results['mse_loss'].append(
                                score[test_train_split].get("mse_loss", None)
                            )
            
    if save:
        df = pd.DataFrame(results)
        df.to_csv(directory + file_name + ".csv", index=False)

    return results