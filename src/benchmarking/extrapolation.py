''''
Function for extrapolation testing
----------------------------------
Based on distance from seed sequence, holdout distant sequences as a
test set.

Modification of code from https://github.com/acmater/NK_Benchmarking/
* Deterministic splits during cross-fold testing
'''

import numpy as np
import pickle as pkl

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from typing import Optional
from torch.utils.data import DataLoader

from benchmarking import make_landscape_data_dicts
from modelling import (
    architectures, 
    collapse_concat, 
    make_dataset, 
    score_sklearn_model
)


def extrapolation(model_dict: dict,
                  landscape_dict: dict,
                  split: float = 0.8,
                  cross_validation: int = 1,
                  save: bool = True,
                  file_name: Optional[str] = None,
                  directory: str = "Results/"):
    """
    Extrapolation function that takes a dictionary of models and a
    landscape dictionary and iterates over all models and landscapes,
    recording results, before finally (saving) and returning them.

    Parameters
    ----------
    model_dict : dict
        Dictionary of model architectures. Format: 
        {sklearn.model : **kwargs}.

    landscape_dict : dict
        Dictionary of protein landscapes. Format: 
        {Name : [Protein_Landscape()]}.

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

    complete_results = {
        x: {key :0 for key in landscape_dict.keys()} 
        for x in model_dict.keys()
    }
    # Iterate over model types
    for model_name, model_hparams in model_dict.items():
        # Iterate over each landscape
        for landscape_name in landscape_dict.keys():
            # Iterate over each instance of each landscape
            for idx, instance in enumerate(landscape_dict[landscape_name]):
                # Get distance data from landscape
                distances = instance.d_data.keys()
                # Deletes zero if it listed as a distance
                distances = [d for d in distances if d] 
                results = []
                results = np.zeros((
                    len(distances),
                    len(distances),
                    cross_validation
                ))
                # cross fold eval
                for fold in range(cross_validation):
                    print()
                    train_datasets = []
                    test_datasets = []

                    for d in distances:
                        x_trn, y_trn, x_tst, y_tst = instance.sklearn_data(
                            split=split,
                            distance=d,
                            random_state=fold,
                        )
                        train_datasets.append([x_trn, y_trn])
                        test_datasets.append([x_tst, y_tst])

                    for j, d in enumerate(distances):
                        # get training and testing data
                        j += 1
                        x_training = collapse_concat(
                            [x[0] for x in train_datasets[:j]]
                        )
                        y_training = collapse_concat(
                            [x[1] for x in train_datasets[:j]]
                        )
                        x_testing = collapse_concat(
                            [x[0] for x in test_datasets[:j]]
                        )
                        y_testing = collapse_concat(
                            [x[1] for x in test_datasets[:j]]
                        )
                        if model_name not in ["gb", "rf"]:

                            loaded_model = architectures.NeuralNetworkRegression(
                                model_name,
                                **model_hparams
                            )
                            # train model
                            print(
                                f"{model_name} trained on Dataset "
                                f"{landscape_name} distances 1-{d}"
                            )
                            loaded_model.fit((x_training, y_training))

                            # score model
                            train_dset = make_dataset(
                                (x_training, y_training)
                            )
                            train_dloader = DataLoader(train_dset)
                            score_train = loaded_model.score(
                                train_dloader
                            )
                            test_dset = make_dataset(
                                (x_testing, y_testing)
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
                            loaded_model.fit(x_training, y_training)

                            # get model performance
                            score = score_sklearn_model(
                                loaded_model,
                                x_training,
                                y_training,
                                x_testing,
                                y_testing
                            )

                        results[idx][j][fold] = score
                        print()

            complete_results[model_name][landscape_name] = np.array(results)

    if save:
        if not file_name:
            file_name = input(
                "What name would you like to save results with?"
            )
        file = open(directory + file_name + ".pkl", "wb")
        pkl.dump(complete_results,file)
        file.close()

    return complete_results
