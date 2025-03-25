'''
Default hyperparameter search space
'''

hparam_space_NK = {
        'linear': {'learning_rate': [0.01, 0.001, 0.0001], 
                   'batch_size': [32, 64, 128, 256]},

        'mlp': {'learning_rate': [0.01, 0.001, 0.0001], 
                'batch_size': [32, 64, 128, 256],
                'max_hidden_layers': 3,
                'hidden_sizes_categorical': [32, 64, 128, 256]},

        'cnn':          {'learning_rate': [0.01, 0.001, 0.0001],
                        'batch_size': [32, 64, 128, 256],
                        'max_conv_layers': 2,
                        'n_kernels_min': 32, 
                        'n_kernels_max': 256, 
                        'n_kernels_step': 32, 
                        'kernel_sizes_min': 3, 
                        'kernel_sizes_max': 5},

        'ulstm':         {'learning_rate': [0.01, 0.001, 0.0001], 
                        'batch_size': [32, 64, 128, 256],
                        'max_lstm_layers': 2, 
                        'hidden_sizes': [64, 128, 256]},

        'blstm':        {'learning_rate': [0.01, 0.001, 0.0001], 
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

hparam_space_GB1 = {
        'linear': {'learning_rate': [0.01, 0.001, 0.0001], 
                   'batch_size': [32, 64, 128, 256]}, 

        'mlp': {'learning_rate': [0.01, 0.001, 0.0001], 
                'batch_size': [32, 64, 128, 256], 
                'max_hidden_layers': 6,
                'hidden_sizes_categorical': [32, 64, 128, 256, 512]}, 

        'cnn': {'learning_rate': [0.01, 0.001, 0.0001], 
                'batch_size': [32, 64, 128, 256],
                'max_conv_layers': 2, #important to keep the max_conv_layers small for NK landscapes to avoid pooling error resulting in RuntimeError: max_pool1d() Invalid computed output size: 0
                'n_kernels_min': 32,
                'n_kernels_max': 512,
                'n_kernels_step': 32, 
                'kernel_sizes_min':3, #seq len is 4, so the only kernel size that works is 3
                'kernel_sizes_max':3}, 

        'ulstm':        {'learning_rate': [0.01, 0.001, 0.0001], 
                        'batch_size': [32, 64, 128, 256],
                        'max_lstm_layers': 2, 
                        'hidden_sizes': [64, 128, 256, 512], 
                        'bidirectional':False}, 

        'blstm':        {'learning_rate': [0.01, 0.001, 0.0001], 
                        'batch_size': [32, 64, 128, 256], 
                        'max_lstm_layers': 2, 
                        'hidden_sizes': [64, 128, 256, 512], 
                        'bidirectional':True}, 
        
        'transformer':  {'learning_rate': [0.01, 0.001, 0.0001], 
                         'batch_size': [32, 64, 128, 256], 
                        'embed_dim_options':[ 32, 64, 128, 256, 512],                                       
                        'max_heads': 8, 
                        'max_layers': 3, 
                        'feedforward_dims': [32, 64, 128, 256, 512], 
                        'max_seq_lengths':[4, 6, 8]}
}

