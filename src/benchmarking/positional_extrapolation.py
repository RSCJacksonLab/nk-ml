''''
Function for positional extrapolation testing
---------------------------------------------
Ensure test data includes mutations at sites where no variation is
observed in the training data.

Modification of code from https://github.com/acmater/NK_Benchmarking/
'''

import numpy as np
import pickle as pkl

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from torch.utils.data import DataLoader
from typing import Optional

from modelling import (
    architectures, 
    collapse_concat, 
    make_dataset, 
    score_sklearn_model)


def positional_extrapolation(model_dict: dict,
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
    # iterate over model types
    for model_name, model_hparams in model_dict.items():
        # iterate over each landscape
        for landscape_name in landscape_dict.keys():
            results = []
            # iterate over each instance of the landscape
            for idx, instance in enumerate(landscape_dict[landscape_name]):
                positions = instance.mutated_positions
                instance_results = np.zeros((
                    len(positions),
                    len(positions),
                    cross_validation
                ))
                # cross-fold eval
                for fold in range(cross_validation):
                    train_datasets = []
                    test_datasets = []

                    # for each position make test/train splits
                    for pos_idx in range(positions):
                        x_trn, y_trn, x_tst, y_tst = instance.sklearn_data(
                            split=split,
                            positions=positions[:pos_idx + 1]
                        )
                        train_datasets.append([x_trn, y_trn])
                        test_datasets.append([x_tst, y_tst])

                    # for each test/train split - train and test models
                    for pos_idx in range(positions):
                        pos_idx += 1
                        x_training = collapse_concat(
                            [x[0] for x in train_datasets[:pos_idx]]
                        )
                        y_training = collapse_concat(
                            [x[1] for x in train_datasets[:pos_idx]]
                        )
                        x_testing = collapse_concat(
                            [x[0] for x in test_datasets[:pos_idx]]
                        )
                        y_testing = collapse_concat(
                            [x[1] for x in test_datasets[:pos_idx]]
                        )
                        if model_name not in ["gb", "rf"]:

                            loaded_model = architectures.NeuralNetworkRegression(
                                model_name,
                                **model_hparams
                            )
                            # train model
                            loaded_model.fit(x_training, 
                                             y_training.reshape(-1,1))
                            print(
                                f"{model_name} trained on Dataset "
                                f"{landscape_name} positions"
                                f"{positions[:pos_idx]}"
                            )
    
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
                        
                        # train model on position data
                        loaded_model.fit(x_training, y_training)

                        print(
                            f"{model_name} trained on Dataset"
                            f" {landscape_name} positions "
                            f"{positions[pos_idx]}."
                        )

                        # get model performance
                        score = score_sklearn_model(
                            loaded_model,
                            x_training,
                            y_training,
                            x_testing,
                            y_testing
                        )
    
                        print(
                            f"On dataset {landscape_name}, fold {fold}, for "
                            f"positions {positions[pos_idx]}, "
                            f"{model_name} returned scores of:"
                        )
                        for metric, value in score.items():
                            print(f"{metric}: {value}")

                        results[pos_idx - 1][fold] = score

                # Remove fold dimension if cross_validation = 1
                results.append(instance_results.squeeze())

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
