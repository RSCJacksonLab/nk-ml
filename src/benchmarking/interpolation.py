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

from typing import Optional

from utils.sklearn_utils import train_test_model

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
    for model_type, model_properties in model_dict.items():
        model, kwargs = model_properties
        # iterate over each landscape
        for name in landscape_dict.keys():
            results = np.zeros((len(landscape_dict[name]), cross_validation))
            # iterate over each instanve of the landscape
            for idx, instance in enumerate(landscape_dict[name]):
                # cross fold evalutation
                for fold in range(cross_validation):
                    print()
                    temp_model = model(**kwargs)
                    x_trn, y_trn, x_tst, y_tst = instance.sklearn_data(
                        split=split,
                        random_state=fold,
                    )
                    score = train_test_model(temp_model, 
                                             x_trn, 
                                             y_trn, 
                                             x_tst, 
                                             y_tst)
                    print(
                        f"{model_type} trained on Dataset {name} achieved an "
                        f"R^2 of {score}."
                    )
                    results[idx][fold] = score
            
            complete_results[model_type][name] = results

    if save:
        if not file_name:
            file_name = input(
                "What name would you like to save results with?"
            )
        file = open(directory + file_name + ".pkl", "wb")
        pkl.dump(complete_results,file)
        file.close()

    return complete_results
