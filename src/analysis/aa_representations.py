''''
Function for getting amino acid representations
-----------------------------------------------
Based on training data, ensure all test datapoints are closer to a seed
sequence than the training datapoint with the greatest distance.

Similar to interpolation except the representations of individual AAs
is also stored.
'''

import inspect
import numpy as np
import pandas as pd
import pickle as pkl

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from torch.utils.data import DataLoader
from typing import Optional

from modelling import architectures, make_dataset, score_sklearn_model
from utils import aa_to_ohe

def get_aa_reps(model_dict: dict,
                landscape_dict: dict,
                sequence_len: int,
                alphabet_size: int,
                split: float = 0.8,
                save: bool = True,
                file_name: Optional[str] = None,
                directory: str = "results/", 
                n_epochs: int = 10, 
                patience: int = 5, 
                min_delta: float = 1e-5):
    """
    Interpolation function that takes a dictionary of models and a
    landscape dictionary and iterates over all models and landscapes,
    recording results, before finally (saving) and returning them.

    Parameters:
    -----------
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
        Boolean value used to determine whether or not the file will
        be saved.

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

    # Iterate over model types
    for model_name in model_names: 
        print('Working on model: {}'.format(model_name))

        # iterate over each landscape
        for landscape_name in landscape_dict.keys():
            print('Working on landscape: {}'.format(landscape_name))

            model_hparams = model_dict[landscape_name][model_name]
            model_hparams["input_dim"] = alphabet_size
            model_hparams["sequence_length"] = sequence_len

            results = {}

            # iterate over each instanve of the landscape
            for idx, instance in enumerate(landscape_dict[landscape_name]):

                # update result dict
                if not instance in results.keys():
                    results[instance] = {}

                landscape_instance = landscape_dict[landscape_name][instance]

                # prepare AA data
                landscape_aas = landscape_instance.amino_acids
                full_aa_ls = [aa * sequence_len for aa in landscape_aas]
                aa_tokens = np.stack([aa_to_ohe(s, landscape_aas) for s in full_aa_ls])
                ## only take first letter in sequence
                aa_arr = np.zeros_like(aa_tokens)
                aa_arr[:, 0, :] = aa_tokens[:, 0, :]

                print(
                    f'Working on instance {idx} of landscape {landscape_name}'
                )

                x_trn, y_trn, x_tst, y_tst = landscape_instance.sklearn_data(
                    split=split,
                    shuffle=True,
                    random_state=0,
                    convert_to_ohe=True,
                    flatten_ohe=False,
                )
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
                    test_dset = make_dataset(
                        (x_tst, y_tst)
                    )
                    test_dloader = DataLoader(test_dset, batch_size=2048)
                    score_test = loaded_model.score(
                        test_dloader,
                    )
                    score = {
                        'train': score_train,
                        'test': score_test
                    }   

                    # get aa representations
                    aa_dummy_targets = np.random.rand((len(aa_arr)))
                    aa_dset = make_dataset(
                        (aa_arr, aa_dummy_targets)
                    )
                    aa_dloader = DataLoader(aa_dset, batch_size=2048)    
                    
                    _, _, aa_reps, _ = loaded_model.predict(
                        aa_dloader
                    )
                    np.save(
                        directory + file_name + f"{landscape_name}_{model_name}_AA_reps.npy",
                        aa_reps
                    )
                
                else:
                    print(f"Cannot get representations for {model_name}")
                    continue

                results[instance] = score
    
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
                for replicate, cv_folds in replicates.items():
                    for cv_fold, splits in cv_folds.items():
                        for data_split, metrics in splits.items():
                            # Append a row with the relevant data
                            rows.append({
                                "model": model,
                                "landscape": landscape,
                                "replicate": replicate,
                                "cv_fold": cv_fold,
                                "data_split": data_split,
                                "pearson_r": metrics.get("pearson_r", None),
                                "r2": metrics.get("r2", None),
                                "mse": metrics.get("mse", None)
                            })

        # Create a DataFrame from the rows
        df = pd.DataFrame(rows)
        df.to_csv(directory + file_name + ".csv", index=False)


    return complete_results