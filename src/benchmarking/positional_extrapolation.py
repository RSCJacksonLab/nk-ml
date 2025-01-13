''''
Function for positional extrapolation testing
---------------------------------------------
Ensure test data includes mutations at sites where no variation is
observed in the training data.

Modification of code from https://github.com/acmater/NK_Benchmarking/
'''

import numpy as np
import pickle as pkl

from typing import Optional

from src.utils.sklearn_utils import collapse_concat, reset_params_skorch


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
    for model_type, model_properties in model_dict.items():
        model, kwargs = model_properties
        # iterate over each landscape
        for name in landscape_dict.keys():
            landscape = landscape_dict[name]
            results = []
            # iterate over each instance of the landscape
            for instance in landscape:
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

                    for pos_idx in range(positions):

                        x_trn, y_trn, x_tst, y_tst = instance.sklearn_data(
                            positions=positions[:pos_idx + 1]
                        )
                        train_datasets.append([x_trn, y_trn])
                        test_datasets.append([x_tst, y_tst])

                    for pos_idx in range(positions):
                        
                        pos_idx += 1
                        x_training = collapse_concat(
                            [x[0] for x in train_datasets[:pos_idx]]
                        )
                        y_training = collapse_concat(
                            [x[1] for x in train_datasets[:pos_idx]]
                        )
                        this_model = model(**kwargs)
                        model_class = this_model.__class__.__name__
                        if model_class == "NeuralNetRegressor":
                            this_model.fit(x_training, 
                                           y_training.reshape(-1,1))
                            print(
                                f"{model_type} trained on Dataset {name} "
                                f"positions {positions[:pos_idx]}"
                            )
                            print()
                            for k, test_dataset in enumerate(test_datasets):
                                score = this_model.score(
                                    test_dataset[0],
                                    test_dataset[1].reshape(-1,1)
                                )
                                print(
                                    f"On dataset {name}, fold {fold}, for "
                                    f"positions {positions[:k + 1]}, {model} "
                                    f"returned an R-squared of {score}."
                                )

                                instance_results[j-1][k][fold] = score
                            # reset the models parameters
                            reset_params_skorch(this_model) 

                        else:
                            this_model.fit(x_training, y_training)
                            print(
                                f"{model_type} trained on Dataset {name} "
                                f"positions {positions[pos_idx]}."
                            )
                            print()
                            for k,test_dataset in enumerate(test_datasets):
                                score = this_model.score(test_dataset[0],
                                                         test_dataset[1])
                                print(
                                    f"On dataset {name}, fold {fold}, for "
                                    f"positions {positions[pos_idx]}, "
                                    f"{model_type} returned an R-squared of "
                                    f"{score}"
                                )
                                instance_results[pos_idx - 1][k][fold] = score
                        print()

                # Remove fold dimension if cross_validation = 1
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
