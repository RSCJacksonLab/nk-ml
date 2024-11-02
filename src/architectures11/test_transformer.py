import torch
import torch.nn as nn
import torch.optim as optim
import math

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

class AminoAcidTransformer(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, dim_feedforward=256, max_seq_length=10):
        super(AminoAcidTransformer, self).__init__()
        
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