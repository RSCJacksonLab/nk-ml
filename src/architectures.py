import torch
import torch.nn as nn
import torch.optim as optim
import math




    



class SequenceRegressionLinear(nn.Module): 
    def __init__(self, alphabet_size=5, sequence_length=10):
        super(SequenceRegressionLinear, self).__init__()
        self.alphabet_size   = alphabet_size
        self.sequence_length = sequence_length
        self.model_name = 'linear'

        input_size = self.alphabet_size*self.sequence_length

        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        x = x.view(x.size(0), -1) # Flatten the input from (batch_size, sequence_length, alphabet_size) to (batch_size, sequence_length * alphabet_size)
        # Pass through the linear layer
        x = self.linear(x)  # Shape: (batch_size, 1)
        return x



class SequenceRegressionMLP(nn.Module):
    def __init__(self, alphabet_size=5, sequence_length=10, hidden_sizes=[128,64]):
        super(SequenceRegressionMLP, self).__init__()

        self.alphabet_size = alphabet_size
        self.sequence_length = sequence_length
        self.model_name = 'mlp'

        input_size = alphabet_size * sequence_length

       


        #MLP layers
        self.fc_layers = nn.ModuleList()

        for i in range(len(hidden_sizes)):
            fc_layer = nn.Linear(input_size, hidden_sizes[i])
            self.fc_layers.append(fc_layer)
            input_size = hidden_sizes[i]

        self.output_layer = nn.Linear(input_size, 1)

        
    def forward(self, x):
        x = x.view(x.size(0), -1)  #reshape to (batch_size, sequence_length*alphabet_size)

        for index, fc_layer in enumerate(self.fc_layers): 
            x = torch.relu(fc_layer(x))

        x = self.output_layer(x)  # Output layer without activation for real value prediction
        return x



class SequenceRegressionCNN(nn.Module):
    def __init__(self, input_channels=20, sequence_length=50, num_conv_layers=2, 
                 n_kernels=[64, 128], kernel_sizes=[3, 3], pool_every=1, 
                 pool_kernel_size=2):
        """
        Args:
            input_channels (int): Number of input channels, e.g., 20 for one-hot encoded amino acids.
            sequence_length (int): Length of each input sequence.
            num_conv_layers (int): Number of convolutional layers.
            n_kernels (list): Number of filters in each conv layer.
            kernel_sizes (list): Kernel size for each conv layer.
            pool_every (int): Apply pooling after every N layers.
            pool_kernel_size (int): Size of the max pooling kernel.
        """
        super(SequenceRegressionCNN, self).__init__()
        
        assert num_conv_layers == len(n_kernels) == len(kernel_sizes), \
            "n_kernels and kernel_sizes must match num_conv_layers"

        # Create convolutional layers dynamically
        self.conv_layers = nn.ModuleList()
        in_channels = input_channels
        
        for i in range(num_conv_layers):
            conv_layer = nn.Conv1d(in_channels=in_channels, out_channels=n_kernels[i], kernel_size=kernel_sizes[i], padding=kernel_sizes[i] // 2)
            self.conv_layers.append(conv_layer)
            in_channels = n_kernels[i]  # Update in_channels for the next layer
        
        # Pooling layer setup
        self.pool = nn.MaxPool1d(pool_kernel_size)
        self.pool_every = pool_every

        self.model_name = 'cnn'
    



        # Determine the flattened size by passing a dummy input
        with torch.no_grad():
            dummy_input = torch.randn(1, input_channels, sequence_length)  # (batch_size, input_channels, sequence_length)
            for i, conv_layer in enumerate(self.conv_layers):
                dummy_input = torch.relu(conv_layer(dummy_input))
                if (i + 1) % pool_every == 0:
                    dummy_input = self.pool(dummy_input)
            in_features = dummy_input.numel()  # Flattened size after conv layers



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
    def __init__(self, input_size=20, hidden_size=128, num_layers=2, bidirectional=True):
        """
        Args:
            input_size (int): Number of input features per time step (20 for one-hot encoded amino acids).
            hidden_size (int): Number of features in the hidden state.
            num_layers (int): Number of recurrent layers.
            bidirectional (bool): If True, make the LSTM bidirectional.
        """
        super(SequenceRegressionLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1  # Number of directions

        # Define the LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )

        # Define the output layer that maps LSTM output to a single real number
        self.output_layer = nn.Linear(hidden_size * self.num_directions, 1)

        self.model_name = 'ulstm' if self.bidirectional==False else 'blstm'
    def forward(self, x):
        # Initialize hidden and cell states for LSTM
        h0 = torch.zeros(self.num_layers * self.num_directions, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * self.num_directions, x.size(0), self.hidden_size).to(x.device)

        # LSTM forward pass
        lstm_out, _ = self.lstm(x, (h0, c0))  # lstm_out shape: (batch_size, sequence_length, hidden_size * num_directions)

        # Take the output from the last time step (assuming many-to-one output)
        last_out = lstm_out[:, -1, :]  # Shape: (batch_size, hidden_size * num_directions)

        # Pass through the output layer
        output = self.output_layer(last_out)  # Shape: (batch_size, 1)
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=10):
        super(PositionalEncoding, self).__init__()
        
        # Create a matrix of max_len x d_model for positional encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

class SequenceRegressionTransformer(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, dim_feedforward=256, max_seq_length=10):
        super(SequenceRegressionTransformer, self).__init__()
        
        # Embedding layer to convert one-hot encoded amino acids to dense vectors
        self.embedding = nn.Linear(input_dim, d_model)
        
        # Positional encoding to add sequence position information
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, 
                                                   dim_feedforward=dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Fully connected output layer for regression
        self.fc_out = nn.Linear(d_model, 1)
        self.model_name = 'transformer'
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_dim)
        
        # Step 1: Embed the input
        x = self.embedding(x)  # Shape: (batch_size, sequence_length, d_model)
        
        # Step 2: Apply positional encoding
        x = x.permute(1, 0, 2)  # Transformer expects input shape (sequence_length, batch_size, d_model)
        x = self.positional_encoding(x)
        
        # Step 3: Transformer encoder
        transformer_out = self.transformer_encoder(x)  # Shape: (sequence_length, batch_size, d_model)
        
        # Step 4: Take the output of the last sequence element
        final_out = transformer_out[-1]  # Shape: (batch_size, d_model)
        
        # Step 5: Fully connected layer to produce a single output
        output = self.fc_out(final_out)  # Shape: (batch_size, 1)
        return output