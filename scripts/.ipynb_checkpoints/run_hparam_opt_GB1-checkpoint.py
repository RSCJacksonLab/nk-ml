import yaml
import optuna as opt
import pickle 
import os

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split

from modelling import (
    get_model_hparams, 
    objective_fn, 
    sklearn_objective_fn
)
from modelling.architectures import *
from modelling.hparam_space import hparam_space_NK
from pscapes.landscape_class import ProteinLandscape


# test hyperparam optimization
def main():

    print('Initialising parameters...')

    SEQ_LEN = 4
    AA_ALPHABET = 'ACDEFGHIKLMNPQRSTVWY'
    REPLICATES = 1 
    N_TRIALS_MULTIPLIER = 15
    PATIENCE = 20
    MIN_DELTA = 1e-5
    N_EPOCHS = 150

    # define hyperparameter search space 
    sklearn_mappings = {"RF": RandomForestRegressor, 
                        "GB": GradientBoostingRegressor}
    model_mapping = {**MODEL_MAPPING, **sklearn_mappings}

    model_names, model_class = zip(*model_mapping.items())
    model_names = list(model_names)
    model_class = list(model_class)
    model_hparams = [hparam_space_NK.get(name) for name in model_names]
    
    



# get data
landscape = ProteinLandscape(
    csv_path=f'../data/experimental_datasets/G_prot_4_mut_seq_space_only.csv',
    amino_acids=AA_ALPHABET
)
ohe = landscape.ohe
y = landscape.fitnesses
x_trn, x_val, y_trn, y_val = train_test_split(
    ohe,
    y, 
    test_size=round(len(y)*0.2),
    random_state=0
)


# commence study for landscape
print(
    f"Commencing studies for landscape GB1"
)

# tune for each model
for idx, model_name in enumerate(model_names):
    print(f"Optimising model: {model_name} for K: {K}")
    study = opt.create_study(direction='minimize')
    if model_name in ['RF', 'GB']:
        print('running sklearn')
        n_trials = 3 * N_TRIALS_MULTIPLIER
        # optimisation
        
        x_trn = [i.flatten().reshape(-1, 1) for i in x_trn]
        x_trn = np.concatenate(x_trn, axis=1).T
        x_val = [i.flatten().reshape(-1, 1) for i in x_val]
        x_val = np.concatenate(x_val, axis=1).T

        study.optimize(
            lambda trial: sklearn_objective_fn(
                trial,
                model_name,
                x_train=x_trn,
                y_train=y_trn,
                x_val=x_val, 
                y_val=y_val),
            n_trials=n_trials
        )
    else:
        print('running nn')
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        n_trials = len(model_hparams[idx]) * N_TRIALS_MULTIPLIER
    
        # optimisation
        study.optimize(
            lambda trial: objective_fn(
                trial,
                model_name,
                model_hparams[idx],
                len(AA_ALPHABET),
                SEQ_LEN,
                (x_trn, y_trn.reshape(-1,1)),
                (x_val, y_val.reshape(-1,1)),
                N_EPOCHS,
                PATIENCE,
                MIN_DELTA,
                device
            ),
            n_trials=n_trials
        )
        print('trials done')
    model_hparam_dict = get_model_hparams(model_name, 
                                         study.best_params)
    
    

    output_dir = os.path.abspath(
        "../hyperopt/results/NK_landscape_model_hparams/")

    

    output_path_yaml   = output_dir + f"/{model_name}_k{K}_r{rep}.yaml"
    output_path_pickle = output_dir + f"/{model_name}_k{K}_r{rep}.pkl"

    #output best parameters to yaml
    with open(
        output_path_yaml,
        "w+"
    ) as f:
        yaml.dump(model_hparam_dict, f)
    
    #output study object as pickle file
    with open(
        output_path_pickle,
        "wb"
    ) as f: 
        pickle.dump(study, f,protocol=pickle.HIGHEST_PROTOCOL)

                



if __name__ == "__main__":
    main()
