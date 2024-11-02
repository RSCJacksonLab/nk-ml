import torch
import torch.nn as nn
import torch.optim as optim

class AminoAcidMLP(nn.Module):
    def __init__(self, input_size, hidden_size1=128, hidden_size2=64):
        super(AminoAcidMLP, self).__init__()
        
        # Define the layers of the MLP
        self.fc1 = nn.Linear(input_size, hidden_size1)  # First hidden layer
        self.fc2 = nn.Linear(hidden_size1, hidden_size2) # Second hidden layer
        self.output = nn.Linear(hidden_size2, 1)  # Output layer for single real value
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))  # Activation after first layer
        x = torch.relu(self.fc2(x))  # Activation after second layer
        x = self.output(x)  # Output layer without activation for real value prediction
        return x