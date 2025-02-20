''''
Function for extrapolation testing
----------------------------------
Based on distance from seed sequence, holdout distant sequences as a
test set.

Modification of code from https://github.com/acmater/NK_Benchmarking/
'''

from tarfile import LENGTH_NAME
import numpy as np

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from torch.utils.data import DataLoader
from typing import List, Optional

from modelling import architectures, make_dataset, score_sklearn_model

def length_testing(model_dict: dict,
                   landscape_dict: dict,
                   alphabet_size: int,
                   split: float = 0.8,
                   cross_validation: int = 1,
                   save: bool = True,
                   amino_acids: str='ACDEFGHIKLMNPQRSTVWY',
                   seq_lens: List[int] = [10, 50, 100, 200, 250, 500],
                   file_name: Optional[str] = None,
                   directory: str = "r esults/"):
    """
    Length testing function that takes a dictionary of models, a
    landscape dictionary, and a list of sequence lengths. It iterates 
    over all of these and leverages the Protein Landscape function that
    enables it to randomly inject length into its sequences to train 
    each model on each of these values.

    Parameters
    ----------
    model_dict : dict
        Dictionary of model architectures. Format: 
        {landscape_name: {model_name : **kwargs}}.

    landscape_dict : dict
        Dictionary of protein landscapes. Format: 
        {landscape_name: [datafile_name: ProteinLandscape]}

    alphabet_size : int
        Number of AAs in the alphabet.

    split : float, default=0.8, Allowed values: 0 < split < 1
        The split point used to partition the data.

    cross_validation : int, default=1
        The number of times to randomly resample the dataset, typically
        used with experimental datasets.

    save : Bool, default=True
        Boolean value used to determine whether or not the file will be
        saved.

    amino_acids : str, default='ACDEFGHIKLMNPQRSTVWY'
        String containing all allowable amino acids.

    seq_lens : list, default=[10, 50, 100, 200, 250, 500]
        List of sequence lengths, determining how long the extended
        sequences will be.

    file_name : str, default=None
        File name to use if saving file. If none is provided, user will
        be prompted for one.

    directory : str, default="Results/"
        Directory is the directory to which the results will be saved.
    """
    # get model names 
    first_key = list(model_dict.keys())[0]
    model_names = list(model_dict[first_key].keys())

    complete_results = {
        model: {key: 0 for key in landscape_dict.keys()} 
        for model in model_names
    }
    # iterate over model types
    for model_name in model_names:
        print('Working on model: {}'.format(model_name))

        # iterate over each landscape
        for landscape_name in landscape_dict.keys():
            print('Working on landscape: {}'.format(landscape_name))

            #extract model hparams
            model_hparams = model_dict[landscape_name][model_name]

            # add dataset properties to hparams
            model_hparams["input_dim"] = alphabet_size

            results = {}

            # iterate over each instance of each landscape
            for idx, instance in enumerate(landscape_dict[landscape_name]):

                # update result dict
                if not instance in results.keys():
                    results[instance] = {}

                print(
                    f'Working on instance {idx} of landscape {landscape_name}'
                )

                # get distance data from landscape
                landscape_instance = landscape_dict[landscape_name][instance]

                # cross fold eval
                for fold in range(cross_validation):

                    print('Working on cross-validation fold: {}'.format(fold))

                    # Iterate over each INSTANCE of each landscape, 1 for experimental
                    for length in seq_lens:

                        # add sequence length hparam
                        model_hparams["sequence_length"] = length

                        x_trn, y_trn, x_tst, y_tst = landscape_instance.return_lengthened_data(
                            length,
                            AAs=amino_acids,
                            split=split,
                            random_state=fold
                        )
                        if model_name not in ["gb", "rf"]:

                            # instantiate model with determined hyperparameters
                            loaded_model = architectures.NeuralNetworkRegression(
                                model_name,
                                **model_hparams
                            )
                            
                            print(f'Loading model hparams: {model_hparams}')

                            # train model and ablated data
                            print('Fitting model')
                            loaded_model.fit((actual_x_train, 
                                              actual_y_train),
                                              n_epochs=n_epochs, 
                                              patience=patience,
                                              min_delta=min_delta)

                            # score model
                            print('Scoring model')
                            train_dset = make_dataset(
                                (actual_x_train, actual_y_train)
                            )
                            train_dloader = DataLoader(train_dset, 
                                                       batch_size=2048)
                            score_train = loaded_model.score(
                                train_dloader
                            )
                            test_dset = make_dataset((x_tst, y_tst))
                            test_dloader = DataLoader(test_dset, 
                                                      batch_size=2048)
                            score_test = loaded_model.score(
                                test_dloader,
                            )
                                
                            score = {
                                'train': score_train,
                                'test': score_test
                                
                            }
                        else:
                            # flatten input data
                            actual_x_train = [
                                i.flatten().reshape(-1, 1) 
                                for i in actual_x_train
                                ]
                            actual_x_train = np.concatenate(
                                actual_x_train, 
                                axis=1
                            ).T

                            x_tst = [
                                i.flatten().reshape(-1, 1) 
                                for i in x_tst
                                ]
                            x_tst = np.concatenate(
                                x_tst, 
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
                            
                            # apply hyperparams
                            model_kwargs = inspect.signature(model_class)
                            kwargs_filtered = {
                                hparam: value 
                                for hparam, value in model_hparams.items()
                                if hparam in model_kwargs.parameters
                            }
                            loaded_model = model_class(
                                **kwargs_filtered
                            )
                            # train model on ablated data
                            print('Fitting model')
                            loaded_model.fit(actual_x_train, actual_y_train)

                            # get model performance
                            print('Scoring model')                            
                            train_score = score_sklearn_model(
                                loaded_model,
                                actual_x_train,
                                actual_y_train,
                            )
                            test_score = score_sklearn_model(
                                loaded_model,
                                x_tst,
                                y_tst
                            )
                            score = {
                                "train": train_score,
                                "test": test_score
                            }
            complete_results[model_type][name] = results

    if save:
        if not file_name:
            file_name = input("What name would you like to save results with?")
        file = open(directory + file_name + ".pkl", "wb")
        pkl.dump(complete_results,file)
        file.close()

    return complete_results