'''
Neural network architectures implemented in PyTorch.
'''
import inspect
import math
import numpy as np
from pydantic import NoneStr
import torch
import torch.nn as nn

from numpy.typing import ArrayLike
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader, random_split
from typing import List, Literal, Optional, Tuple

from modelling.data_utils import make_dataset
from modelling.ml_utils import train_model


class NeuralNetworkRegression(nn.Module):
    '''
    Wrapper for Regressor models to manage train and score methods.

    Parameters:
    -----------
    model_name : str
        Name of the model. Options include: `linear`, `mlp`, `cnn`,
        `ulstm`, 'blstm', `transformer`

    **kwargs
        Key word arguments for specific model instantiation.
    '''

    def __init__(self,
                 model_name: Literal[
                     'linear', 'mlp', 'cnn', 'ulstm', 'blstm', 'transformer'
                 ], 
                 **kwargs):
        super().__init__()
        model_class = MODEL_MAPPING[model_name]
        
        # get model agnostic hparams
        self.lr = kwargs.get('lr', 0.001)
        self.batch_size = kwargs.get('batch_size', 16)

        # get relevant kwargs
        model_kwargs = inspect.signature(model_class)
        kwargs_filtered = {
            hparam: value for hparam, value in kwargs.items()
            if hparam in model_kwargs.parameters
        }

        if model_name == 'blstm': 
            kwargs_filtered['bidirectional']=True
        elif model_name == 'ulstm': 
            kwargs_filtered['bidirectional']=False

        # instantiate model
        self.model = model_class(**kwargs_filtered)

    def fit(
        self, 
        train_data: Tuple[ArrayLike, ArrayLike],
        val_data: Optional[Tuple[ArrayLike, ArrayLike]] = None,
        n_epochs: int = 30,
        patience: int = 5,
        min_delta: int = 1e-5
) -> Tuple[dict, dict]:
        '''
        Train model on provided data. Will make validation data
        automatically if not provided for early stopping.

        Parameters:
        -----------
        train_data : Tuple[ArrayLike, ArrayLike]
            Training data containing features (x) as first element and
            target (y) as second.

        val_data : Optional[Tuple[ArrayLike, ArrayLike]]
            Optional validation data containing features (x) as first 
            element and target (y) as second. If not provided, the train
            data will be used to generate a validation dataset.
        '''
        # convert data into dataset
        trn_dset = make_dataset(train_data)

        # get validation set
        if val_data is not None:
            val_dset = make_dataset(val_data)
        else:
            random_state = torch.Generator().manual_seed(0)
            trn_dset, val_dset = random_split(trn_dset, 
                                              [0.8, 0.2], 
                                              generator=random_state)
        
        # make dataloaders
        trn_dloader = DataLoader(trn_dset, self.batch_size, shuffle=True)
        val_dloader = DataLoader(val_dset, self.batch_size, shuffle=False)

        # initialise optimizer and loss fn
        self.model.train()
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), 
                                     lr=self.lr)
        
        model, trn_loss_ls, val_loss_ls = train_model(
            self.model,
            optimizer,
            loss_fn,
            trn_dloader,
            val_dloader,
            n_epochs=n_epochs,
            patience=patience,
            min_delta=min_delta,
            device='cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model
        self.trn_loss_ls = trn_loss_ls
        self.val_loss_ls = val_loss_ls

        # get full scoring for train and validation data
        train_res = self.score(trn_dloader)
        val_res = self.score(val_dloader)

        return train_res, val_res

    def score(
        self,
        dloader: DataLoader
    ) -> dict:
        '''
        Score model performance on provided data.

        Parameters:
        -----------
        dloader : Tuple[ArrayLike, ArrayLike]
            Data containing features (x) as first element and
            target (y) as second.
        '''
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.model.to(device)
        self.model.eval()

        total_loss = 0
        all_preds = []
        all_targets = []
        
        # get loss + predictions 
        loss_fn = nn.MSELoss()
        with torch.no_grad():
            for inputs, targets in dloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = self.model(inputs)
                loss = loss_fn(outputs, targets)
                total_loss += loss.item()
                all_preds.append(outputs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())

        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        # get performance metrics
        pearson_r, _ = pearsonr(all_preds.flatten(),
                                all_targets.flatten())
        r2 = r2_score(all_targets, all_preds)
        avg_loss = total_loss / len(dloader)

        return {
            'pearson_r': pearson_r,
            'r2': r2,
            'loss': avg_loss
        }

class SequenceRegressionLinear(nn.Module): 
    '''
    Linear regression with PyTorch.

    Parameters:
    -----------
    alphabet_size: int, default=5
        Number of unique characters in the input sequence.

    sequence_length: int, default=10
        Length of the input sequence. 
    '''
    def __init__(self, alphabet_size: int = 5, sequence_length: int = 10):
        super(SequenceRegressionLinear, self).__init__()
        self.alphabet_size   = alphabet_size
        self.sequence_length = sequence_length
        input_size = self.alphabet_size * self.sequence_length
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        # flatten the input [batch_size, sequence_length, alphabet_size]
        # to [batch_size, sequence_length * alphabet_size]
        x = x.view(x.size(0), -1)
        # Pass through the linear layer
        x = self.linear(x)  # Shape: (batch_size, 1)
        return x


class SequenceRegressionMLP(nn.Module):
    '''
    Multi-layer perceptron (MLP) in PyTorch.

    Parameters:
    -----------
    alphabet_size: int, default=5
        Number of unique characters in the input sequence.
    
    sequence_length: int, default=10
        Length of the input sequence.

    hidden_sizes: list, default=[128, 64]
        List of hidden layer sizes.
    '''
    def __init__(self, 
                 alphabet_size: int = 5,
                 sequence_length: int = 10,
                 hidden_sizes: list = [128,64]):
        super(SequenceRegressionMLP, self).__init__()
        self.alphabet_size = alphabet_size
        self.sequence_length = sequence_length

        input_size = alphabet_size * sequence_length
        self.fc_layers = nn.ModuleList()
        for i in range(len(hidden_sizes)):
            fc_layer = nn.Linear(input_size, hidden_sizes[i])
            self.fc_layers.append(fc_layer)
            input_size = hidden_sizes[i]
        self.output_layer = nn.Linear(input_size, 1)
        
    def forward(self, x):
        # reshape to (batch_size, sequence_length*alphabet_size)
        x = x.view(x.size(0), -1)
        for fc_layer in self.fc_layers: 
            x = torch.relu(fc_layer(x))
        # Output layer without activation for real value prediction
        x = self.output_layer(x)  
        return x
    

class SequenceRegressionCNN(nn.Module):
    def __init__(self, 
                 input_channels: int = 20, 
                 sequence_length: int = 50, 
                 num_conv_layers: int = 2, 
                 n_kernels: List[int] = [64, 128], 
                 kernel_sizes: List[int] = [3, 3], 
                 pool_every: int = 1, 
                 pool_kernel_size: int = 2):
        """
        CNN for sequence regression in PyTorch.

        Parameters:
        -----------
        input_channels: int, default=20
            Number of input channels, e.g., 20 for one-hot encoded
            amino acids.

        sequence_length: int, default=50
            Length of each input sequence.

        num_conv_layers: int, default=2
            Number of convolutional layers.

        n_kernels: list, default=[64, 128]
            Number of filters in each conv layer.
        
        kernel_sizes: list, default=[3, 3]
            Kernel size for each conv layer.

        pool_every: int, default=1
            Apply pooling after every N layers.

        pool_kernel_size: int, default=2
            Size of the max pooling kernel.
        """
        super(SequenceRegressionCNN, self).__init__()
        
        assert num_conv_layers == len(n_kernels) == len(kernel_sizes), \
            "n_kernels and kernel_sizes must match num_conv_layers"

        # Create convolutional layers dynamically
        self.conv_layers = nn.ModuleList()
        in_channels = input_channels
        
        for i in range(num_conv_layers):
            conv_layer = nn.Conv1d(in_channels=in_channels,
                                   out_channels=n_kernels[i],
                                   kernel_size=kernel_sizes[i],
                                   padding=kernel_sizes[i] // 2)
            self.conv_layers.append(conv_layer)
            # Update in_channels for the next laye
            in_channels = n_kernels[i]
        
        # Pooling layer setup
        self.pool = nn.MaxPool1d(pool_kernel_size)
        self.pool_every = pool_every

    
        # Determine the flattened size by passing a dummy input
        with torch.no_grad():
            # (batch_size, input_channels, sequence_length)
            dummy_input = torch.randn(1, input_channels, sequence_length)
            for i, conv_layer in enumerate(self.conv_layers):
                dummy_input = torch.relu(conv_layer(dummy_input))
                if (i + 1) % pool_every == 0:
                    dummy_input = self.pool(dummy_input)
            # Flattened size after conv layers
            in_features = dummy_input.numel()  
        # Fully connected output layer
        self.fc = nn.Linear(in_features, 1)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_channels)
        # Reshape for CNN input (batch_size, input_channels, sequence_length)
        x = x.permute(0, 2, 1)
        # Apply convolutional layers with optional pooling
        for i, conv_layer in enumerate(self.conv_layers):
            x = torch.relu(conv_layer(x))
            if (i + 1) % self.pool_every == 0:
                x = self.pool(x)
        # Flatten the output for the fully connected layer
        x = x.view(x.size(0), -1)
        # Fully connected layer to produce a single output
        output = self.fc(x)
        return output


class SequenceRegressionLSTM(nn.Module):
    '''
    LSTM for sequence regression in PyTorch.

    Parameters:
    -----------
    input_size: int, default=20
        Number of input features per time step (20 for one-hot encoded amino acids).

    hidden_size: int, default=128
        Number of features in the hidden state.

    num_layers: int, default=2
        Number of recurrent layers.

    bidirectional: bool, default=True
        If True, make the LSTM bidirectional.
    '''
    def __init__(self, 
                 input_size: int = 20,
                 hidden_size: int = 128, 
                 num_layers: int = 2, 
                 bidirectional: bool = False):
        super(SequenceRegressionLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # Define the LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )

        # Define the output layer that maps LSTM output to a real number
        self.output_layer = nn.Linear(hidden_size * self.num_directions, 1)

        
    def forward(self, x):
        # Initialize hidden and cell states for LSTM
        h0 = torch.zeros(self.num_layers * self.num_directions, 
                         x.size(0), 
                         self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * self.num_directions, 
                         x.size(0), 
                         self.hidden_size).to(x.device)

        # LSTM forward pass
        # lstm_out shape: (batch_size, seq_len, hidden_size * num_directions)
        lstm_out, _ = self.lstm(x, (h0, c0))  

        # Take the output from the last time step (assumes many-to-one output)
        # Shape: (batch_size, hidden_size * num_directions)
        last_out = lstm_out[:, -1, :]  

        # Pass through the output layer
        output = self.output_layer(last_out)  # Shape: (batch_size, 1)
        return output


class PositionalEncoding(nn.Module):
    '''Positional encoding fn for tx module.'''
    def __init__(self, d_model, max_len=10):
        super(PositionalEncoding, self).__init__()
        
        # Create a matrix of max_len x d_model for positional encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, 
                                max_len, 
                                dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(
                0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

class SequenceRegressionTransformer(nn.Module):
    '''
    Transformer for sequence regression.

    Parameters:
    -----------
    input_dim: int
        Number of input features per time step.

    d_model: int, default=64
        Number of expected features in the encoder/decoder inputs.

    nhead: int, default=4
        Number of heads in the multiheadattention models.
    
    n_layers: int, default=2
        Number of sub-encoder-layers in the transformer.

    n_dim: int, default=256
        Dimension of the feedforward network model.

    max_seq_length: int, default=10
        Maximum sequence length for positional encoding.
    '''
    def __init__(self, 
                 input_dim: int, 
                 d_model: int = 64,
                 nhead: int = 4,
                 n_layers: int = 2,
                 n_dim: int = 256,
                 max_seq_length: int = 10):
        super(SequenceRegressionTransformer, self).__init__()
        # Embedding layer to convert ohe amino acids to dense vectors
        self.embedding = nn.Linear(input_dim, d_model)
        
        # Positional encoding to add sequence position information
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, 
                                                   nhead=nhead, 
                                                   dim_feedforward=n_dim)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, 
                                                         num_layers=n_layers)
        # Fully connected output layer for regression
        self.fc_out = nn.Linear(d_model, 1)
        

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        # Embed the input
        # Shape: (batch_size, seq_len, d_model)
        x = self.embedding(x) 
        # Apply positional encoding
        # Transformer expects input shape (seq_len, batch_size, d_model)
        x = x.permute(1, 0, 2)  
        x = self.positional_encoding(x)
        # Transformer encoder
        # Shape: (seq_len, batch_size, d_model)
        transformer_out = self.transformer_encoder(x)
        # Take the output of the last sequence element
        # Shape: (batch_size, d_model)
        final_out = transformer_out[-1]
        # Fully connected layer to produce a single output
        # Shape: (batch_size, 1)
        output = self.fc_out(final_out)
        return output
    
# for mapping model names to class
MODEL_MAPPING = {
    'linear': SequenceRegressionLinear,
    'mlp': SequenceRegressionMLP,
    'cnn': SequenceRegressionCNN,
    'ulstm': SequenceRegressionLSTM,
    'blstm': SequenceRegressionLSTM,
    'transformer': SequenceRegressionTransformer,
}