import os
import re
import yaml
import random
import numpy as np

from typing import Optional

from pscapes import ProteinLandscape

 

def make_landscape_data_dicts(
    data_dir: str,
    model_dir: str,
    alphabet: str = 'ACDEFGHIKLMNPQRSTVWY', 
    experimental: bool = False,
    n_replicates: int = 4, 
    random_seed: int = 1,
    seed_seqs: Optional[list] = None,
    landscape_names: Optional[list] = None):

    """
    Function to parse hyperparameter and raw landscape data files into format suitable for extrapolation/ablation/
    positional extrapolation functions. 

    Args:
        data_dir (str): directory where raw landscape data files are stored 
        model_dir (str): directory where hyperparameter files are stores 
        alphabet (str): the amino acid alphabet for this protein landscape 
        seed_id (int):  index of the sequence you want to set as the seed sequence
        seed_seqs (list): seed sequences for generating landscapes - only possible for experimental.

    Returns: 
        model_dict (dict): dictionary with model hyperaparameters 
        data_dict (dict): dictionary of ProteinLandscape class objects 
    
    """

    # find yaml files in hparam dir
    hparam_set_files = os.listdir(model_dir)
    hparam_set_files = [f for f in hparam_set_files if f.endswith(".yaml")]

    # find csv files in data dir
    data_files = os.listdir(data_dir)
    data_files = [f for f in data_files if f.endswith(".csv")]

    # get landscape names and model names
    if not landscape_names:
        landscape_names = sorted(
            {filename.split('_')[0] for filename in hparam_set_files}
        )
    
    model_names = sorted(
        {filename.split('_')[1].replace('.yaml', '') 
         for filename in hparam_set_files}
    )
    model_dict = {}
    data_dict = {}
    
    for landscape in landscape_names: 

        if not landscape in model_dict.keys():
            model_dict[landscape] = {}
        if not landscape in data_dict.keys():
            data_dict[landscape] = {}

        print('LANDSCAPE NAME:{}'.format(landscape))

        # get landscapes for each data file

        if not experimental:         

            landscape_dict = {
                "r" + re.search(r'_r(\d+)', f).group(1): ProteinLandscape(
                    csv_path=data_dir + '/' + f,
                    amino_acids=alphabet, 
                )
                for f in data_files
                if f.startswith(landscape)
            }
        

        # for experimental landscapes we have a different workflow for setting up the dict 
        elif experimental: 

            np.random.seed(random_seed)

            filename = next((f for f in data_files if f.startswith(landscape)), None)
            
            assert filename != None, f'No filename found for the landscape {landscape}'
            
            #load a 'test' landscape to get indexes
            test_landscape = ProteinLandscape(
                csv_path=data_dir+ '/' + filename, 
                amino_acids=alphabet) 
            
            n_sequences = test_landscape.num_sequences()
            
            if seed_seqs is None:
                r_seed_ids = np.random.randint(low=0, 
                                            high=n_sequences-1,
                                            size=n_replicates)
                
                landscape_dict = {
                    f'r{r_index}': ProteinLandscape(
                        csv_path=data_dir + '/'+filename, 
                        amino_acids=alphabet, 
                        seed_id=r
                    )
                    for r_index, r in enumerate(r_seed_ids)
                }
            else:
                landscape_dict = {
                    seed_seq: ProteinLandscape(
                        csv_path=data_dir + '/'+filename, 
                        amino_acids=alphabet, 
                        seed_seq=seed_seq
                    )
                    for seed_seq in seed_seqs
                }

# landscape dict {k1: {r1: ProteinLandscape(), r2: PL}, k2: }

 

        if len(landscape_dict.keys()) == 0:

            print(f"No data could be found for {landscape}")

            continue

 

        # get each model for the landscape

        for model in model_names:

            # parse hyperparameter data
            if f"{landscape}_{model}.yaml" in hparam_set_files:
                with open(model_dir + f"/{landscape}_{model}.yaml", "r") as f:                    
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



def sub_dict(data_dict: dict,
             n_replicates: int = 4, 
             random_seed: int = 1,
             replicate_names: Optional[list] = None):
    
    """ Utility function for subsampling replicates from data_dict.
        Useful for reducing the number of replicates fed into experimental scripts. 
    """
    landscape_keys = list(data_dict.keys())
    replicate_keys = list(data_dict[landscape_keys[0]])

    if replicate_names:
        selection = replicate_names
    else:
        random.seed(random_seed)
        selection = random.sample(replicate_keys, n_replicates)

    out_dict = {i:{} for i in landscape_keys}

    for landscape in landscape_keys: 
        for replicate in selection: 
            out_dict[landscape][replicate] = data_dict[landscape][replicate]
    
    return out_dict





    

    

     
