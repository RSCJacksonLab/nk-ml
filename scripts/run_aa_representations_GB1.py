from analysis.aa_representations import get_aa_reps
from benchmarking.file_proc import make_landscape_data_dicts, sub_dict

ALPHABET = 'ACDEFGHIKLMNPQRSTVWY'
SEQ_LEN  = 4
N_REPLICATES = 1
N_EPOCHS = 150
PATIENCE = 10
MIN_DELTA = 1e-5

#interpolation NK 
def main(): 

    print('Loading data and hyperparameter optimisation.')
    model_dict, data_dict = make_landscape_data_dicts(
        data_dir='./data/experimental_datasets/', 
        model_dir='./hyperopt/ohe/gb1_hparams/', 
        alphabet=ALPHABET,
        experimental=True,
        n_replicates=N_REPLICATES,
        random_seed=0
    )
    small_dict = sub_dict(data_dict, 
                          n_replicates=N_REPLICATES, 
                          random_seed=1)

    print('Training and testing models.')
    _ = get_aa_reps(model_dict=model_dict, 
                    landscape_dict=small_dict, 
                    sequence_len=6, 
                    alphabet_size=len(ALPHABET), 
                    file_name='GB1',
                    directory= './results/aa_representations/',
                    n_epochs=N_EPOCHS, 
                    patience=PATIENCE,
                    min_delta=MIN_DELTA
                    )

if __name__ == "__main__": 
    main()



