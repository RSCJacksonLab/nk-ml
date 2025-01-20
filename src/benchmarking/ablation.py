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
import os

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from torch.utils.data import DataLoader
from typing import List, Optional

from benchmarking.file_proc import make_landscape_data_dicts
from modelling import architectures, make_dataset, score_sklearn_model

def ablation_testing(model_dict: dict,
                     landscape_dict: dict,
                     split: float = 0.8,
                     cross_validation: int = 1,
                     save: bool = True,
                     file_name: Optional[str] = None,
                     shuffle: bool = True,
                     sample_densities: List[float] = [0.9, 0.7, 0.5, 0.3, 0.1],
                     directory: str = "results/"):
    """
    Interpolation function that takes a dictionary of models and a
    landscape dictionary and iterates over all models and landscapes,
    recording results, before finally (saving) and returning them.

    Parameters
    ----------
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
        Boolean value used to determine whether or not the file will be
        saved.

    file_name : str, default=None
        File name to use if saving file. If none is provided, user will
        be prompted for one.

    sample_densities : list, default=[0.9, 0.7, 0.5, 0.3, 0.1]
        Split densities that are passed to the sklearn_data function of
        each landscape.

    directory : str, default="results/"
        Directory is the directory to which the results will be saved.
    """

    complete_results = {
        x: {key: 0 for key in landscape_dict.keys()} 
        for x in model_dict.keys()
    }
    
    first_key = list(model_dict.keys())[0]
    model_names = list(model_dict[first_key].keys()) #get the model names 
    print(model_names)

    # Iterate over model types. 
    # model_dict = {'k0':{'model_name':{hparams}}}
    #for model_name, model_hparams in model_dict.items():
    for model_name in model_names: 
        print('Working on model: {}'.format(model_name))

        # Iterate over each landscape
        # landscape_dict = {'k0':{r1: ProteinLandscape, r2: ProteinLandscape...rn:}}
        for landscape_name in landscape_dict.keys():
            print('Working on landscape: {}'.format(landscape_name))

            #extract model hparams
            model_hparams = model_dict[landscape_name][model_name]

            results = np.zeros((
                len(landscape_dict[landscape_name]),
                len(sample_densities),
                cross_validation
            ))
            # Iterate over each instance of each landscape
            for idx, instance in enumerate(landscape_dict[landscape_name]):
                print('Working on instance {} of landscape {}'.format(idx, landscape_name))

                # cross fold eval
                for fold in range(cross_validation):
                    print('Working on cross-validation fold: {}'.format(fold))

                    for j, density in enumerate(sample_densities):
                        print('Working on sampling density: {}'.format(density))
                        
                        landscape_instance = landscape_dict[landscape_name][instance]

                        # get data splits
                        x_trn, y_trn, x_tst, y_tst = landscape_instance.sklearn_data(
                            split=split,
                            shuffle=shuffle,
                            random_state=fold, 
                            convert_to_ohe=True
                        )
                        # remove random fraction of data from train
                        np.random.seed(0)
                        idxs = np.random.choice(
                            len(x_trn),
                            size=int(len(x_trn)*density)
                        )
                        actual_x_train = x_trn[idxs]
                        actual_y_train = y_trn[idxs]

                        if model_name not in ["gb", "rf"]:
                            # instantiate model with determined hyperparameters
                            loaded_model = architectures.NeuralNetworkRegression(
                                model_name,
                                **model_hparams
                            )
                            
                            print('Loading model hparams: {}'.format(model_hparams))


                            # train model and ablated data
                            print('Fitting model')
                            loaded_model.fit((actual_x_train, 
                                              actual_y_train),
                                              n_epochs=10)

                            # score model
                            print('Scoring model')
                            train_dset = make_dataset(
                                (actual_x_train, actual_y_train)
                            )
                            train_dloader = DataLoader(train_dset)
                            score_train = loaded_model.score(
                                train_dloader
                            )
                            test_dset = make_dataset((x_tst, y_tst))
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
                            print('Fitting model')
                            loaded_model.fit(actual_x_train, actual_y_train)

                            # get model performance
                            print('Scoring model')                            
                            score = score_sklearn_model(
                                loaded_model,
                                actual_x_train,
                                actual_y_train,
                                x_tst,
                                y_tst
                            )

                        results[idx][j][fold] = score  # idx is landscape replicate index;
                                                       # j is density index 
                                                       # fold is cross-validation fold

                        print(
                            f"For sample density {density}, on "
                            f"{landscape_name} instance {idx} {model_name} "
                            f"returned an. Score of: ."
                        )
                        for metric, value in score.items():
                            print(f"{metric}: {value}")

            complete_results[model_name][landscape_name] = results.squeeze()

    if save:
        if not file_name:
            file_name = input(
                "What name would you like to save results with?"
            )
        file = open(directory + file_name + ".pkl", "wb")
        pkl.dump(complete_results, file)
        file.close()

    return complete_results




# load yamls for hparams
hopt_dir =  os.path.abspath("./hyperopt/results/nk_landscape/") # hyperparameter directory
data_dir =  os.path.abspath("./data/nk_landscapes/") # data directory with NK landscape data



model_dict, data_dict = make_landscape_data_dicts(
    data_dir,
    hopt_dir,
    alphabet= 'ACDEFG'
)

ablation_testing(model_dict=model_dict, 
                 landscape_dict=data_dict, 
                 split=0.8,
                 cross_validation=5)