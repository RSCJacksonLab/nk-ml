import torch
import torch.nn as nn
import torch.optim as optim

class AminoAcidLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2):
        super(AminoAcidLSTM, self).__init__()
        
        # Define the LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # Fully connected layer for the final output
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        # LSTM layer output
        lstm_out, _ = self.lstm(x)
        
        # Get the output from the last time step
        last_out = lstm_out[:, -1, :]  # Shape: (batch_size, hidden_size)
        
        # Fully connected layer
        output = self.fc(last_out)  # Shape: (batch_size, 1)
        return output