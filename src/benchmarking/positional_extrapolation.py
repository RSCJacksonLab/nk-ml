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

from scipy.stats import pearsonr
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import r2_score
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
                                  control_pct: Optional[float] = None,
                                  save: bool = True,
                                  file_name: Optional[str] = None,
                                  directory: str = "results/", 
                                  n_epochs: int = 30, 
                                  patience: int = 5, 
                                  min_delta: float = 1e-5, ):
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

    control_pct : float, default=None
        If running a control, the percent of test data re-added into 
        train for determination of effect prediction capabilities when 
        interpolating.

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
    effect_complete_results = {
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
            effect_results = {}

            # iterate over each instance of the landscape
            for idx, instance in enumerate(landscape_dict[landscape_name]):

                print(
                    f'Working on instance {idx} of landscape {landscape_name}'
                )
                
                # update result dict
                if not instance in results.keys():
                    results[instance] = {}

                if not instance in effect_results.keys():
                    effect_results[instance] = {}

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

                    if not pos.item() in results[instance].keys():
                        results[instance][pos.item()] = {}

                    if not pos.item() in effect_results[instance].keys():
                        effect_results[instance][pos.item()]  = {}

                    # determine WT AA at site
                    wt_aa_at_pos = wt_sequence[pos]
                    
                    # determine alternate AAs at site for testing
                    alt_aas_at_pos = list(
                        set(sequence_data[:, pos]) - {wt_aa_at_pos}
                    )

                    # get data with position fixed to WT
                    trn_idx = np.where(sequence_data[:, pos] == wt_aa_at_pos)[0]

                    # if control run - add some variants back in
                    added_idx = np.array([])
                    if control_pct:
                        print('Note: This run is a control.')
                        for alt_aa in alt_aas_at_pos:
                            alt_idx = np.where(sequence_data[:, pos] == alt_aa)[0]
                            alt_idx = np.random.choice(
                                alt_idx, 
                                size=int(len(alt_idx) * control_pct), 
                                replace=False)
                            added_idx = np.concatenate([added_idx, alt_idx])
                        # add data back into train
                        trn_idx = np.concatenate([added_idx, trn_idx]).astype(np.int32)

                    # make training split
                    x_trn = x_data[trn_idx]
                    y_trn = y_data[trn_idx]  

                    # get models and train
                    if model_name not in ["gb", "rf"]:
                        loaded_model = architectures.NeuralNetworkRegression(
                            model_name,
                            **model_hparams
                        )
                        # train model
                        print(f'Fitting model {model_name}')
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
                            # remove potential control re-added data
                            test_idx = np.setdiff1d(test_idx, added_idx)
                            sequence_tst = sequence_data[test_idx, :]
                            x_tst = x_data[test_idx]
                            y_tst = y_data[test_idx]

                            test_dset = make_dataset(
                                (x_tst, y_tst)
                            )
                            test_dloader = DataLoader(test_dset, batch_size=2048)

                            print(f'Scoring on test for {alt_aa} at {pos}')
                            results[instance][pos.item()][alt_aa.item()] = loaded_model.score(
                                test_dloader,
                            )

                            # get corresponding train data (i.e. data with the same context)
                            comparison_seqs = sequence_tst.copy()
                            comparison_seqs[:, pos] = wt_aa_at_pos
                            
                            seq_to_idx = {
                                tuple(seq): i 
                                for i, seq in enumerate(map(tuple, sequence_data))
                            }
                            comparison_idx = np.array(
                                [seq_to_idx.get(tuple(seq), None) 
                                for seq in map(tuple, comparison_seqs)],
                                dtype=object
                            )

                            valid_mask = (comparison_idx != None).astype(bool)
                            test_idx = test_idx[valid_mask].astype(np.int32)
                            comparison_idx = comparison_idx[valid_mask].astype(np.int32)
                            x_comparison = x_data[comparison_idx]
                            y_comparison = y_data[comparison_idx]

                            comparison_dset = make_dataset(
                                (x_comparison, y_comparison)
                            )
                            comparison_dloader = DataLoader(comparison_dset, 
                                                            batch_size=2048)
                            
                            # get predictions
                            tst_preds, _, _ = loaded_model.predict(test_dloader)
                            tst_preds = tst_preds[valid_mask]
                            y_tst = y_tst[valid_mask]
                            comparison_preds, _, _ = loaded_model.predict(comparison_dloader)
                            
                            # get effects
                            true_effect = (y_comparison - y_tst)
                            pred_effect = (comparison_preds - tst_preds).ravel()
                            
                            # score ability to predict effect
                            mae = np.mean(np.abs(true_effect - pred_effect), dtype=np.float32).item()
                            pearson_r, _ = pearsonr(true_effect, pred_effect)
                            r2 = r2_score(true_effect, pred_effect)

                            effect_results[instance][pos.item()][alt_aa.item()] = {
                                'pearson_r': pearson_r.item(),
                                'r2': r2,
                                'mean_absolute_error': mae
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
                            # remove control train data from test
                            test_idx = np.setdiff1d(test_idx, added_idx)
                            sequence_tst = sequence_data[test_idx, :]
                            x_tst = x_data[test_idx]
                            y_tst = y_data[test_idx]

                            x_tst = [
                                i.ravel().reshape(-1, 1) 
                                for i in x_tst
                                ]
                            x_tst = np.concatenate(
                                x_tst, 
                                axis=1
                            ).T
                            results[instance][pos.item()][alt_aa.item()] = score_sklearn_model(
                                loaded_model,
                                x_tst,
                                y_tst
                            )

                            # get corresponding train data (i.e. data with the same context)
                            comparison_seqs = sequence_tst.copy()
                            comparison_seqs[:, pos] = wt_aa_at_pos
                            
                            seq_to_idx = {
                                tuple(seq): i 
                                for i, seq in enumerate(map(tuple, sequence_data))
                            }
                            comparison_idx = np.array(
                                [seq_to_idx.get(tuple(seq), None) 
                                for seq in map(tuple, comparison_seqs)],
                                dtype=object
                            )

                            valid_mask = (comparison_idx != None).astype(bool)
                            test_idx = test_idx[valid_mask].astype(np.int32)
                            comparison_idx = comparison_idx[valid_mask].astype(np.int32)
                            x_comparison = x_data[comparison_idx]
                            y_comparison = y_data[comparison_idx]
                            
                            x_comparison = [
                                i.ravel().reshape(-1, 1) 
                                for i in x_comparison
                                ]
                            x_comparison = np.concatenate(
                                x_comparison, 
                                axis=1
                            ).T

                            # get predictions
                            tst_preds = loaded_model.predict(x_tst)
                            tst_preds = tst_preds[valid_mask]
                            y_tst = y_tst[valid_mask]
                            comparison_preds = loaded_model.predict(x_comparison)
                            
                            # get effects
                            true_effect = (y_comparison - y_tst)
                            pred_effect = (comparison_preds - tst_preds).ravel()
                            
                            # score ability to predict effect
                            mae = np.mean(np.abs(true_effect - pred_effect), dtype=np.float32).item()
                            pearson_r, _ = pearsonr(true_effect, pred_effect)
                            r2 = r2_score(true_effect, pred_effect)

                            effect_results[instance][pos.item()][alt_aa.item()] = {
                                'pearson_r': pearson_r.item(),
                                'r2': r2,
                                'mean_absolute_error': mae
                            }

            complete_results[model_name][landscape_name] = results
            effect_complete_results[model_name][landscape_name] = effect_results

    if save:
        if not file_name:
            file_name = input(
                "What name would you like to save results with?"
            )
        file = open(directory + file_name + "_Performance" + ".pkl", "wb")
        pkl.dump(complete_results, file)
        file.close()

        file = open(directory + file_name + "_MutationalEffectPrediction" + ".pkl", "wb")
        pkl.dump(effect_complete_results, file)
        file.close()      

        # save csv
        # Prepare a list to hold rows for the DataFrame
        training_rows = []
        # Iterate through the nested dictionary structure
        for model, landscapes in complete_results.items():
            for landscape, replicates in landscapes.items():
                for replicate, sites in replicates.items():
                    for site, splits in sites.items():
                        for data_split, metrics in splits.items():
                            # Append a row with the relevant data
                            training_rows.append({
                                "model": model,
                                "landscape": landscape,
                                "replicate": replicate,
                                "fixed_site": site,
                                "data_split": data_split,
                                "pearson_r": metrics.get("pearson_r", None),
                                "r2": metrics.get("r2", None),
                                "mse_loss": metrics.get("mse_loss", None), 
                                "train_epochs": metrics.get("train_epochs", None)
                            })
        # Create a DataFrame from the rows
        df = pd.DataFrame(training_rows)
        df.to_csv(directory + file_name + "_Performance" + ".csv", index=False)

        # save csv
        # Prepare a list to hold rows for the DataFrame
        effect_rows = []
        # Iterate through the nested dictionary structure
        for model, landscapes in effect_complete_results.items():
            for landscape, replicates in landscapes.items():
                for replicate, sites in replicates.items():
                    for site, amino_acids in sites.items():
                        for aa, metrics in amino_acids.items():
                            # Append a row with the relevant data
                            effect_rows.append({
                                "model": model,
                                "landscape": landscape,
                                "replicate": replicate,
                                "fixed_site": site,
                                "test_aa": aa,
                                "pearson_r": metrics.get("pearson_r", None),
                                "r2": metrics.get("r2", None),
                                "mae": metrics.get("mean_absolute_error", None)
                            })
        # Create a DataFrame from the rows
        df = pd.DataFrame(effect_rows)
        df.to_csv(directory + file_name + "_MutationalEffectPrediction" + ".csv", index=False)

    return complete_results