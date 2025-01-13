''''
Function for ablation testing
-----------------------------
Randomly reducing the train dataset to a fraction of its original size
and test model performance.

Modification of code from https://github.com/acmater/NK_Benchmarking/
* Deterministic fold splits
* Deterministic ablation
'''

import numpy as np
import pickle as pkl

from typing import List, Optional

from src.utils.sklearn_utils import train_test_model

def ablation_testing(model_dict: dict,
                     landscape_dict: dict,
                     split: float = 0.8,
                     cross_validation: int = 1,
                     save: bool = True,
                     file_name: Optional[str] = None,
                     shuffle: bool = True,
                     sample_densities: List[float] = [0.9, 0.7, 0.5, 0.3, 0.1],
                     directory: str = "Results/"):
    """
    Interpolation function that takes a dictionary of models and a
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

    sample_densities : list, default=[0.9, 0.7, 0.5, 0.3, 0.1]
        Split densities that are passed to the sklearn_data function of
        each landscape.

    directory : str, default="Results/"
        Directory is the directory to which the results will be saved.
    """

    complete_results = {
        x: {key :0 for key in landscape_dict.keys()} 
        for x in model_dict.keys()
    }
    # Iterate over model types
    for model_type, model_properties in model_dict.items():
        model, kwargs = model_properties
        
        # Iterate over each landscape
        for name in landscape_dict.keys():
            results = np.zeros((
                len(landscape_dict[name]),
                len(sample_densities),
                cross_validation
            ))
            # Iterate over each INSTANCE of each landscape,
            # 1 for experimental
            for idx, instance in enumerate(landscape_dict[name]):
                # cross fold eval
                for fold in range(cross_validation):
                    print()
                    for j, density in enumerate(sample_densities):

                        temp_model = model(**kwargs)
                        x_trn, y_trn, x_tst, y_tst = instance.sklearn_data(
                            split=split,
                            shuffle=shuffle,
                            random_state=fold
                        )
                        np.random.seed(0)
                        idxs = np.random.choice(
                            len(x_trn),
                            size=int(len(x_trn)*density)
                        )
                        actual_x_train = x_trn[idxs]
                        actual_y_train = y_trn[idxs]

                        score = train_test_model(
                            temp_model,
                            actual_x_train,
                            actual_y_train,
                            x_tst,
                            y_tst
                        )

                        results[idx][j][fold] = score

                        print(
                            f"For sample density {density}, on {name} "
                            f"instance {idx} {model_type} returned an "
                            f"R-squared of {score}."
                        )
            complete_results[model_type][name] = results.squeeze()

    if save:
        if not file_name:
            file_name = input(
                "What name would you like to save results with?"
            )
        file = open(directory + file_name + ".pkl", "wb")
        pkl.dump(complete_results, file)
        file.close()

    return complete_results
