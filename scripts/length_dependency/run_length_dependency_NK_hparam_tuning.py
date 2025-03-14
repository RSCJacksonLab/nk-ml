from benchmarking.length_dependency_hparam_tuning import length_dependency_test_with_tuning
from benchmarking.file_proc import make_landscape_data_dicts, sub_dict

import time

ALPHABET = 'ACDEFG'
SEQ_LEN  = 6
N_REPLICATES = 4
N_TRIALS_MULTIPLIER = 15
N_EPOCHS = 1
PATIENCE = 10
MIN_DELTA = 1e-5

def main():
    print('Loading data and hyperparameter optimisation.')

    _, data_dict = make_landscape_data_dicts(
        data_dir='./data/nk_landscapes/', 
        model_dir='./hyperopt/ohe/nk_landscape_hparams/', 
        alphabet=ALPHABET,
        landscape_names=["k1"]
        )

    small_dict = sub_dict(data_dict, 
        n_replicates=N_REPLICATES, 
        random_seed=1,
        replicate_names=['r0', 'r5', 'r4', 'r7', 'r2']
        )

    print('Training and testing models.')
    t1 = time.time()
    length_dependency_test_with_tuning(
        model_ls=["cnn"],
        landscape_dict=small_dict, 
        alphabet_size=len(ALPHABET),
        tuning_landscape_rep="r0",
        hparam_reference="nk",
        amino_acids=ALPHABET,
        file_name='length_dependency_results_wTuning_NK',
        directory= './results/',
        n_epochs=N_EPOCHS, 
        patience=PATIENCE,
        min_delta=MIN_DELTA,
        n_trials_multiplier=N_TRIALS_MULTIPLIER,
        seq_lens=[10, 100, 250]
        )
    t2 = time.time()
    
    with open('./results/length_dependency_NK_wTuning_time.log', 'w') as file: 
        file.write(f"Time taken: {t2-t1} seconds")       

if __name__ == "__main__": 
    main()