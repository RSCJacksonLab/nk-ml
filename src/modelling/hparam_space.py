'''
Default hyperparameter search space
'''

hparam_space = {
    'linear': {'learning_rate': [0.01, 0.001, 0.0001], 
               'batch_size': [32, 64, 128, 256]},
    'mlp': {'learning_rate': [0.01, 0.001, 0.0001], 
            'batch_size': [32, 64, 128, 256],
            'max_hidden_layers': 3,
            'hidden_sizes_categorical': [32, 64, 128, 256]},
    'cnn': {'learning_rate': [0.01, 0.001, 0.0001],
            'batch_size': [32, 64, 128, 256],
            'max_conv_layers': 2,
            'n_kernels_min': 32, 
            'n_kernels_max': 256, 
            'n_kernels_step': 32, 
            'kernel_sizes_min': 3, 
            'kernel_sizes_max': 5},
    'ulstm': {'learning_rate': [0.01, 0.001, 0.0001], 
              'batch_size': [32, 64, 128, 256],
              'max_lstm_layers': 2, 
              'hidden_sizes': [64, 128, 256]},
    'blstm': {'learning_rate': [0.01, 0.001, 0.0001], 
              'batch_size': [32, 64, 128, 256],
              'max_lstm_layers': 2,
              'hidden_sizes': [64, 128, 256]},
    'transformer': {'learning_rate': [0.01, 0.001, 0.0001], 
                    'batch_size': [32, 64, 128, 256],
                    'embed_dim_options': [ 32, 64, 128, 256],                                       
                    'max_heads': 8, 
                    'max_layers': 2, 
                    'feedforward_dims': [32, 64, 128, 256], 
                    'max_seq_lengths': [6, 8, 10]},
}