from benchmarking.length_dependency import length_dependency_test
from benchmarking.file_proc import make_landscape_data_dicts, sub_dict

import time

ALPHABET = 'ACDEFG'
SEQ_LEN  = 6
N_REPLICATES = 1
N_EPOCHS = 1
PATIENCE = 8
MIN_DELTA = 1e-5

def main():
    print('Loading data and hyperparameter optimisation.')
    model_dict, data_dict = make_landscape_data_dicts(
        data_dir='./data/nk_landscapes/', 
        model_dir='./hyperopt/ohe/nk_landscape_hparams/', 
        alphabet=ALPHABET
        )

    small_dict = sub_dict(data_dict, 
        n_replicates=N_REPLICATES, 
        random_seed=1
        )

    print('Training and testing models.')
    t1 = time.time()
    length_dependency_results = length_dependency_test(
        model_dict=model_dict, 
        landscape_dict=small_dict, 
        alphabet_size=len(ALPHABET),
        amino_acids=ALPHABET,
        file_name='length_dependency_results_NK',
        directory= './results/',
        n_epochs=N_EPOCHS, 
        patience=PATIENCE,
        min_delta=MIN_DELTA, 
        )
    t2 = time.time()
    
    with open('./results/length_dependency_NK_time.log', 'w') as file: 
        file.write(f"Time taken: {t2-t1} seconds")       


if __name__ == "__main__": 
    main()