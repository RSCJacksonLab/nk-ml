import numpy as np

from benchmarking.ablation import ablation_test
from benchmarking.extrapolation import extrapolation_test
from benchmarking.positional_extrapolation import positional_extrapolation_test
from benchmarking.interpolation import interpolation_test

from benchmarking.file_proc import make_landscape_data_dicts


ALPHABET = 'ACDEFG'
SEQ_LEN  = 6


#ablation NK 
def main(): 
    print('Loading data and hyperparameter optimisation.')
    model_dict, data_dict = make_landscape_data_dicts(data_dir='./data/nk_landscapes/', 
                                                    model_dir='./hyperopt/ohe/nk_landscape_hparams/', 
                                                    alphabet=ALPHABET)
    

    print('Training and testing models.')
    ablation_results = ablation_test(model_dict=model_dict, 
                                    landscape_dict=data_dict, 
                                    sequence_len=6, 
                                    alphabet_size=len(ALPHABET), 
                                    file_name='ablation_results',
                                    n_epochs=100, 
                                    patience=10,
                                    min_delta=1e-5
                                    )

if __name__ == "__main__": 
    main()


