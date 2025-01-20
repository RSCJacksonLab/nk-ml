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

from typing import Optional

from benchmarking import make_landscape_data_dicts
from modelling import architectures, collapse_concat, make_dataset


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

            # Get distance data from landscape
            landscape = landscape_dict[landscape_name]
            distances = landscape[0].d_data.keys()

            # Deletes zero if it listed as a distance
            distances = [d for d in distances if d] 
            results = []

            # Iterate over each instance of each landscape
            for instance in landscape:
                instance_results = np.zeros((
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

                        # get raining data
                        j+=1
                        x_training = collapse_concat(
                            [x[0] for x in train_datasets[:j]]
                        )
                        y_training = collapse_concat(
                            [x[1] for x in train_datasets[:j]]
                        )
                        
                        # instantiate model with determined hyperparameters
                        loaded_model = architectures.NeuralNetworkRegression(
                            model_name,
                            **model_hparams
                            )
                        if model_class == "NeuralNetRegressor":
                            this_model.fit(x_training,
                                           y_training.reshape(-1,1))
                            print(
                                f"{model_type} trained on Dataset {name} "
                                f"distances 1-{d}"
                            )
                            print()
                            for k, test_dataset in enumerate(test_datasets):
                                score = this_model.score(
                                    test_dataset[0],
                                    test_dataset[1].reshape(-1,1)
                                )
                                print(
                                    f"On dataset {name}, fold {fold}, for "
                                    f"distance {distances[k]}, {model_type} "
                                    f"returned an R-squared of {score}"
                                )
                                instance_results[j-1][k][fold] = score
                            # reset model parameters
                            reset_params_skorch(this_model)

                        else:
                            this_model.fit(x_training,y_training)
                            print(
                                f"{model_type} trained on Dataset {name} "
                                f"distances 1-{d}"
                            )
                            print()
                            for k,test_dataset in enumerate(test_datasets):
                                score = this_model.score(test_dataset[0],
                                                         test_dataset[1])
                                print(f"On dataset {name}, fold {fold}, for"
                                      f" distance {distances[k]}, "
                                      f"{model_type} returned an R-squared "
                                      f"of {score}",
                                )
                                instance_results[j-1][k][fold] = score
                        print()
                # Removes fold dimension if cross_validation = 1
                results.append(instance_results.squeeze()) 

            complete_results[model_type][name] = np.array(results)

    if save:
        if not file_name:
            file_name = input(
                "What name would you like to save results with?"
            )
        file = open(directory + file_name + ".pkl", "wb")
        pkl.dump(complete_results,file)
        file.close()

    return complete_results
