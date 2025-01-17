import os
import re
import yaml

from pscapes import ProteinLandscape

def make_landscape_data_dicts(
    data_dir: str,
    model_dir: str,
    alphabet: str = 'ACDEFGHIKLMNPQRSTVWY',
):

    # find yaml files in hparam dir
    hparam_set_files = os.listdir(model_dir)
    hparam_set_files = [f for f in hparam_set_files if f.endswith(".yaml")]

    # find csv files in data dir
    data_files = os.listdir(data_dir)
    data_files = [f for f in data_files if f.endswith(".yaml")]

    # get landscape names and model names
    landscape_names = sorted({filename.split('_')[0] 
                              for filename in hparam_set_files})
    model_names = sorted({filename.split('_')[1].replace('.yaml', '') 
                          for filename in hparam_set_files})

    model_dict = {}
    data_dict = {}

    for landscape in landscape_names:

        if not landscape in model_dict.keys():
            model_dict[landscape] = {}
        if not landscape in data_dict.keys():
            data_dict[landscape] = {}

        # get landscapes for each data file
        landscape_dict = {
            f"r{re.search(r'_r(\d+)', f).group(1)}": ProteinLandscape(
                csv_path=data_dir + f,
                amino_acids=alphabet
            )
            for f in data_files
            if f.startswith(landscape)
        }

        if len(landscape_dict.keys()) == 0:
            print(f"No data could be found for {landscape}")
            continue

        # get each model for the landscape
        for model in model_names:
            # parse hyperparameter data
            if f"{landscape}_{model}.yaml" in hparam_set_files:
                with open(model_dir + f"{landscape}_{model}.yaml", "r") as f:
                    hparam_set = yaml.safe_load(f)
            else:
                print("Could not load expected hyperparameter set for "
                      f"{model} on {landscape}.")
                continue

            # update dicts
            model_dict[landscape][model] = hparam_set
        
        if len(model_dict[landscape].keys()) == 0:
            print(f"No models found for {landscape}")
            continue
        
        data_dict[landscape] = landscape_dict
    
    return model_dict, data_dict