import yaml
import optuna as opt
import os

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split

from modelling import (
    get_model_hparams, 
    objective_fn, 
    sklearn_objective_fn
)
from modelling.architectures import *
from modelling.hparam_space import hparam_space
from pscapes.landscape_class import ProteinLandscape

# test hyperparam optimization

def main():

    print('Initialising parameters...')

    SEQ_LEN = 6
    AA_ALPHABET = 'ACDEFG'
    REPLICATES = 1 
    N_TRIALS_MULTIPLIER = 2 
    PATIENCE = 20
    MIN_DELTA = 1e-6
    N_EPOCHS = 10

    # define hyperparameter search space 
    sklearn_mappings = {"rf": RandomForestRegressor, 
                        "gb": GradientBoostingRegressor}
    model_mapping = {**MODEL_MAPPING, **sklearn_mappings}

    model_names, model_class = zip(*model_mapping.items())
    model_names = list(model_names)
    model_class = list(model_class)
    model_hparams = [hparam_space.get(name) for name in model_names]
    
    # for each ruggedness value
    for K in range(SEQ_LEN):
        
        # for each replicate
        for rep in range(REPLICATES):

            # get data
            landscape = ProteinLandscape(
                csv_path=f'./data/nk_landscapes/k{K}_r{rep}.csv',
                amino_acids=AA_ALPHABET
            )
            ohe = landscape.ohe
            y = landscape.fitnesses
            x_outer_trn, x_tst, y_outer_trn, y_tst = train_test_split(
                ohe,
                y, 
                test_size=round(len(y)*0.2), 
                random_state=0
            )
            x_trn, x_val, y_trn, y_val = train_test_split(
                x_outer_trn,
                y_outer_trn, 
                test_size=round(len(y)*0.2), 
                random_state=0
            )

            # commence study for landscape
            print(
                f"Commencing studies for landscape K: {K}, replicate: {rep}"
            )
            
            # tune for each model
            for idx, model_name in enumerate(model_names):
                print(f"Optimising model: {model_name} for K: {K}")
                study = opt.create_study(direction='minimize')
                if model_name in ['rf', 'gb']:
                    n_trials = 3 * N_TRIALS_MULTIPLIER
                    # optimisation 
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
                    device = 'cuda' if torch.cuda.is_available() else 'cpu'
                    n_trials = (len(model_hparams[idx]) - 2) * N_TRIALS_MULTIPLIER
                    # optimisation
                    study.optimize(
                        lambda trial: objective_fn(
                            trial,
                            model_name,
                            model_hparams[idx],
                            len(AA_ALPHABET),
                            SEQ_LEN,
                            (x_trn, y_trn),
                            (x_val, y_val),
                            N_EPOCHS,
                            PATIENCE,
                            MIN_DELTA,
                            device
                        )
                    )
                model_hparam_dict = get_model_hparams(model_name, 
                                                     study.best_params)
                os.mkdir("../hyperopt/results/NK_landscape_model_hparams")
                with open(
                    ("../hyperopt/results/NK_landscape_model_hparams/"
                    f"{model_name}_k{K}_r{rep}.yaml"),
                    "w"
                ) as f:
                    yaml.dump(model_hparam_dict, f)

if __name__ == "__main__":
    main()
