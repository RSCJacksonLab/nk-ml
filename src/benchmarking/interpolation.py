''''
Function for interpolation testing
----------------------------------
Based on training data, ensure all test datapoints are closer to a seed
sequence than the training datapoint with the greatest distance.

Modification of code from https://github.com/acmater/NK_Benchmarking/
* Added cross-fold evaluation
* Deterministic splits during cross-fold testing
'''

import numpy as np
import pickle as pkl

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from torch.utils.data import DataLoader
from typing import Optional

from modelling import architectures, make_dataset, score_sklearn_model

def interpolation(model_dict: dict,
                  landscape_dict: dict,
                  split: float = 0.8,
                  cross_validation: int = 1,
                  save: bool = True,
                  file_name: Optional[str] = None,
                  directory: str = "Results/"):
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
    complete_results = {
        x: {key :0 for key in landscape_dict.keys()}
        for x in model_dict.keys()
    }
    # Iterate over model types
    for model_name, model_hparams in model_dict.items():
        # iterate over each landscape
        for landscape_name in landscape_dict.keys():
            results = np.zeros((len(landscape_dict[landscape_name]), cross_validation))
            # iterate over each instanve of the landscape
            for idx, instance in enumerate(landscape_dict[landscape_name]):
                # cross fold evalutation
                for fold in range(cross_validation):
                    print()
                    x_trn, y_trn, x_tst, y_tst = instance.sklearn_data(
                        split=split,
                        random_state=fold,
                    )
                    if model_name not in ["gb", "rf"]:

                        loaded_model = architectures.NeuralNetworkRegression(
                            model_name,
                            **model_hparams
                        )
                        # train model
                        loaded_model.fit((x_trn, y_trn))

                        # score model
                        train_dset = make_dataset(
                            (x_trn, y_trn)
                        )
                        train_dloader = DataLoader(train_dset)
                        score_train = loaded_model.score(
                            train_dloader
                        )
                        test_dset = make_dataset(
                            (x_tst, y_tst)
                        )
                        test_dloader = DataLoader(test_dset)
                        score_test = loaded_model.score(
                            test_dloader,
                        )
                        score = {
                            'train': score_train,
                            'test': score_test
                        }   
                    
                    else:
                        if model_name == "rf":
                            loaded_model = RandomForestRegressor(
                                **model_hparams
                            )
                        elif model_name == "gb":
                            loaded_model = GradientBoostingRegressor(
                                **model_hparams
                            )
                        else:
                            print(f"Model {model_name} not known.")
                            continue

                        # train model on ablated data
                        loaded_model.fit(x_trn, y_trn)

                        # get model performance
                        score = score_sklearn_model(
                            loaded_model,
                            x_trn,
                            y_trn,
                            x_tst,
                            y_tst
                        )

                    print(
                        f"{model_name} trained on Dataset {landscape_name} "
                        f"a score of:"
                    )
                    for metric, value in score.items():
                        print(f"{metric}: {value}")

                    results[idx][fold] = score
        
            complete_results[model_name][landscape_name] = results

    if save:
        if not file_name:
            file_name = input(
                "What name would you like to save results with?"
            )
        file = open(directory + file_name + ".pkl", "wb")
        pkl.dump(complete_results,file)
        file.close()

    return complete_results
