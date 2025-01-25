import numpy as np

from benchmarking.ablation import ablation_test
from benchmarking.extrapolation import extrapolation_test
from benchmarking.positional_extrapolation import positional_extrapolation_test
from benchmarking.interpolation import interpolation_test

from benchmarking.file_proc import make_landscape_data_dicts, sub_dict

import time
import random


ALPHABET = 'ACDEFG'
SEQ_LEN  = 6
random.seed(10) 
N_REPLICATES = 4 

#ablation NK 
def main(): 
    print('Loading data and hyperparameter optimisation.')
    model_dict, data_dict = make_landscape_data_dicts(data_dir='../data/nk_landscapes/', 
                                                      model_dir='../hyperopt/ohe/nk_landscape_hparams/', 
                                                      alphabet=ALPHABET)
    
    

    small_dict = sub_dict(data_dict, 
                          n_replicates=N_REPLICATES)


    print(small_dict)

    print('Training and testing models.')

    t1 = time.time()
    ablation_results = ablation_test(model_dict=model_dict, 
                                    landscape_dict=data_dict, 
                                    sequence_len=6, 
                                    alphabet_size=len(ALPHABET), 
                                    file_name='ablation_results_NK',
                                    directory= '../results/'
                                    n_epochs=100, 
                                    patience=10,
                                    min_delta=1e-5
                                    )
    t2 = time.time()
    
    with open('ablation_time.log', 'w') as file: 
        file.write("Time taken: {}".format(t2-t1))       


if __name__ == "__main__": 
    main()


