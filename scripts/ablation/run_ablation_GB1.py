import numpy as np

from benchmarking.ablation import ablation_test

from benchmarking.file_proc import make_landscape_data_dicts, sub_dict

import time
import random

random.seed(10) 


ALPHABET = 'ACDEFGHIKLMNPQRSTVWY'
SEQ_LEN  = 4
N_REPLICATES = 1 
N_EPOCHS = 1
PATIENCE = 10
MIN_DELTA = 1e-5

#ablation GB1
def main(): 
    print('Loading data and hyperparameter optimisation.')
    model_dict, data_dict = make_landscape_data_dicts(
        data_dir='./data/experimental_datasets/', 
        model_dir='./hyperopt/ohe/gb1_hparams/', 
        alphabet=ALPHABET, 
        experimental=True, 
        n_replicates=N_REPLICATES,
        random_seed=1
        )

    d1 = model_dict['gb1']['gb']
    d2 = model_dict['gb1']['rf']

    filt_dict = {'gb1':{'gb':d1, 'rf': d2}}


    #filtered_model_dict = model_dict['gb1']['gb'] + model_dict['gb1']['rf']


    print('Training and testing models.')

    t1 = time.time()
    ablation_results = ablation_test(
        model_dict=filt_dict, 
        landscape_dict=data_dict, 
        sequence_len=SEQ_LEN, 
        alphabet_size=len(ALPHABET), 
<<<<<<< HEAD
        file_name='ablation_results_GB1',
        directory= './results/',
=======
        file_name='ablation_results_GB1_rf_gb',
        directory= '../../results/',
>>>>>>> a8fb5acbf32536687dc3db42b1c14bf3fce9ddc3
        n_epochs=N_EPOCHS, 
        patience=PATIENCE,
        min_delta=MIN_DELTA
        )
    t2 = time.time()
    
    with open('../../results/ablation_time__rf_gb_GB1.log', 'w') as file: 
        file.write(f"Time taken: {t2-t1} seconds")       


if __name__ == "__main__": 
    main()


