import numpy as np
import pandas as pd

from benchmarking.file_proc import make_landscape_data_dicts
from modelling import architectures

NK_ALPHABET = 'ACDEFG'
NK_SEQ_LEN  = 6

GB1_ALPHABET = 'ACDEFGHIKLMNPQRSTVWY'
GB1_SEQ_LEN  = 4

def main():

    model_n_params = {
        "model": [],
        "landscape": [],
        "n_params": []
    }

    # nk landscapes
    model_dict, _ = make_landscape_data_dicts(
            data_dir='./data/nk_landscapes/', 
            model_dir='./hyperopt/nk_landscape_hparams/', 
            alphabet=NK_ALPHABET,
            n_replicates=1,
    )

    first_key = list(model_dict.keys())[0]
    model_names = list(model_dict[first_key].keys())

    for model_name in model_names:

        for landscape_name in model_dict.keys():
            
            if model_name not in ["gb", "rf"]:

                model_hparams = model_dict[landscape_name][model_name]
                model_hparams["input_dim"] = len(NK_ALPHABET)
                model_hparams["sequence_length"] = NK_SEQ_LEN

                loaded_model = architectures.NeuralNetworkRegression(
                    model_name,
                    **model_hparams
                )
                model_parameters = filter(
                    lambda p: p.requires_grad, loaded_model.model.parameters()
                )
                n_params = sum([np.prod(p.size()) for p in model_parameters])

                model_n_params["model"].append(model_name)
                model_n_params["landscape"].append(landscape_name)
                model_n_params["n_params"].append(n_params)

    # gb1
    model_dict, _ = make_landscape_data_dicts(
            data_dir='./data/experimental_datasets/', 
            model_dir='./hyperopt/gb1_hparams/', 
            alphabet=GB1_ALPHABET,
            n_replicates=1,
            experimental=True
    )

    first_key = list(model_dict.keys())[0]
    model_names = list(model_dict[first_key].keys())

    for model_name in model_names:

        for landscape_name in model_dict.keys():
            
            if model_name not in ["gb", "rf"]:

                model_hparams = model_dict[landscape_name][model_name]
                model_hparams["input_dim"] = len(GB1_ALPHABET)
                model_hparams["sequence_length"] = GB1_SEQ_LEN

                loaded_model = architectures.NeuralNetworkRegression(
                    model_name,
                    **model_hparams
                )
                model_parameters = filter(
                    lambda p: p.requires_grad, loaded_model.model.parameters()
                )
                n_params = sum([np.prod(p.size()) for p in model_parameters])

                model_n_params["model"].append(model_name)
                model_n_params["landscape"].append(landscape_name)
                model_n_params["n_params"].append(n_params)

    df = pd.DataFrame(model_n_params)
    df.to_csv("./results/model_parameters.csv", index=False)

if __name__ == "__main__": 
    main()