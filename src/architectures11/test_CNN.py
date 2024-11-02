import torch
import torch.nn as nn
import torch.optim as optim

class AminoAcidCNN(nn.Module):
    def __init__(self, input_channels=20, sequence_length=50, num_filters=64, kernel_size=3):
        super(AminoAcidCNN, self).__init__()
        
        # Define the first convolutional layer
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=num_filters, kernel_size=kernel_size, padding=1)
        self.pool = nn.MaxPool1d(2)  # Pooling layer to reduce the sequence length by half
        self.conv2 = nn.Conv1d(in_channels=num_filters, out_channels=num_filters * 2, kernel_size=kernel_size, padding=1)
        
        # Compute the size of the output after convolution and pooling layers
        reduced_length = sequence_length // 2 // 2  # Assuming two max poolings
        self.fc = nn.Linear(num_filters * 2 * reduced_length, 1)  # Fully connected layer

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_channels)
        
        # Reshape for CNN input (batch_size, input_channels, sequence_length)
        x = x.permute(0, 2, 1)
        
        # First convolutional layer + ReLU + Pooling
        x = self.pool(torch.relu(self.conv1(x)))
        
        # Second convolutional layer + ReLU + Pooling
        x = self.pool(torch.relu(self.conv2(x)))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layer
        output = self.fc(x)
        return output

