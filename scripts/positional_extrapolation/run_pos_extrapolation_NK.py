
from benchmarking.positional_extrapolation import positional_extrapolation_test
from benchmarking.file_proc import make_landscape_data_dicts, sub_dict

import time

ALPHABET = 'ACDEFG'
SEQ_LEN  = 6
N_REPLICATES = 4
N_EPOCHS = 150
PATIENCE = 10
MIN_DELTA = 1e-5

#extrapolation NK 
def main(): 
    print('Loading data and hyperparameter optimisation.')
    model_dict, data_dict = make_landscape_data_dicts(
        data_dir='./data/nk_landscapes/', 
        model_dir='./hyperopt/ohe/nk_landscape_hparams/', 
        alphabet=ALPHABET
        )

    small_dict = sub_dict(data_dict, 
        n_replicates=N_REPLICATES, 
        random_seed=1,
        replicate_seeds=["AAAAAA"]
        )

    print('Training and testing models.')
    t1 = time.time()
    extrapolation_results = positional_extrapolation_test(model_dict=model_dict, 
                                                            landscape_dict=small_dict, 
                                                            sequence_len=SEQ_LEN, 
                                                            alphabet_size=len(ALPHABET), 
                                                            file_name='positional_extrapolation_results_CNN_Effects_NK',
                                                            directory= './results/positional_extrapolation/',
                                                            n_epochs=N_EPOCHS, 
                                                            patience=PATIENCE,
                                                            min_delta=MIN_DELTA,
                                                            )
    t2 = time.time()
    
    

    # run control
    # print('Training and testing models as controls.')
    # t1_control = time.time()
    # extrapolation_control_results = positional_extrapolation_test(model_dict=model_dict, 
    #                                                         landscape_dict=small_dict, 
    #                                                         sequence_len=SEQ_LEN, 
    #                                                         alphabet_size=len(ALPHABET), 
    #                                                         file_name='positional_extrapolation_results_NK_CONTROL_',
    #                                                         directory= './results/',
    #                                                         n_epochs=N_EPOCHS, 
    #                                                         patience=PATIENCE,
    #                                                         min_delta=MIN_DELTA,
    #                                                         control_pct=0.8
    #                                                         )
    # t2_control = time.time()

    # with open('../../results/positional_extrapolation_time.log', 'w') as file: 
    #     file.write(f"Time taken for positional extrapolation experiments: {t2-t1} seconds")   
    #     file.write(f"Time taken for positional extrapolation control experiments: {t2_control-t1_control} seconds")  
        
if __name__ == "__main__": 
    main()