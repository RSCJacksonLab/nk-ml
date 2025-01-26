import numpy as np

from benchmarking.extrapolation import extrapolation_test

from benchmarking.file_proc import make_landscape_data_dicts, sub_dict

import time
import random

random.seed(10) 


ALPHABET = 'ACDEFG'
SEQ_LEN  = 6
N_REPLICATES = 1
N_EPOCHS = 1
PATIENCE = 10
MIN_DELTA = 1e-5

#extrapolation NK 
def main(): 
    print('Loading data and hyperparameter optimisation.')
    model_dict, data_dict = make_landscape_data_dicts(data_dir='./data/nk_landscapes/', 
                                                      model_dir='./hyperopt/ohe/nk_landscape_hparams/', 
                                                      alphabet=ALPHABET)
    
    

    small_dict = sub_dict(data_dict, 
                          n_replicates=N_REPLICATES, 
                          random_seed=1)



    print('Training and testing models.')
    t1 = time.time()
    extrapolation_results = extrapolation_test(model_dict=model_dict, 
                                                landscape_dict=small_dict, 
                                                sequence_len=6, 
                                                alphabet_size=len(ALPHABET), 
                                                file_name='extrapolation_results_NK',
                                                directory= './',
                                                n_epochs=N_EPOCHS, 
                                                patience=PATIENCE,
                                                min_delta=MIN_DELTA
                                                )
    t2 = time.time()
    
    with open('./extrapolation_time.log', 'w') as file: 
        file.write("Time taken: {}".format(t2-t1))       


if __name__ == "__main__": 
    main()


