
from benchmarking.positional_extrapolation import positional_extrapolation_test
from benchmarking.file_proc import make_landscape_data_dicts, sub_dict

import time

ALPHABET = 'ACDEFGHIKLMNPQRSTVWY'
SEQ_LEN  = 4
N_REPLICATES = 4
N_EPOCHS = 1
PATIENCE = 10
MIN_DELTA = 1e-5

#extrapolation NK 
def main(): 
    import os
    os.listdir('./data/experimental_datasets/')
    print('Loading data and hyperparameter optimisation.')
    model_dict, data_dict = make_landscape_data_dicts(data_dir='./data/experimental_datasets/', 
                                                      model_dir='./hyperopt/ohe/gb1_hparams/', 
                                                      alphabet=ALPHABET,
                                                      experimental=True,
                                                      n_replicates=N_REPLICATES,
                                                      random_seed=1)
    
    

    small_dict = sub_dict(data_dict, 
                          n_replicates=N_REPLICATES, 
                          random_seed=1)



    print('Training and testing models.')
    t1 = time.time()
    extrapolation_results = positional_extrapolation_test(model_dict=model_dict, 
                                                            landscape_dict=small_dict, 
                                                            sequence_len=SEQ_LEN, 
                                                            alphabet_size=len(ALPHABET), 
                                                            file_name='positional_extrapolation_results_GB1',
                                                            directory= '../../results/',
                                                            n_epochs=N_EPOCHS, 
                                                            patience=PATIENCE,
                                                            min_delta=MIN_DELTA
                                                            )
    t2 = time.time()
    
    with open('./results/positional_extrapolation_time_GB1.log', 'w') as file: 
        file.write(f"Time taken: {t2-t1} seconds")    

    # run control
    print('Training and testing models as controls.')
    extrapolation_control_results = positional_extrapolation_test(model_dict=model_dict, 
                                                            landscape_dict=small_dict, 
                                                            sequence_len=SEQ_LEN, 
                                                            alphabet_size=len(ALPHABET), 
                                                            file_name='positional_extrapolation_results_GB1_CONTROL',
                                                            directory= '../../results/',
                                                            n_epochs=N_EPOCHS, 
                                                            patience=PATIENCE,
                                                            min_delta=MIN_DELTA,
                                                            control_pct=0.8
                                                            )   


if __name__ == "__main__": 
    main()