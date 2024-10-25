import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader




import torch


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0, path='best_model.pt'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            path (str): Path to save the best model.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.path = path
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(val_loss, model)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decreases."""
        torch.save(model.state_dict(), self.path)

def train_model(model, optimizer, loss_fn, train_loader, val_loader, n_epochs=30, device='cpu', patience=5):
    early_stopping = EarlyStopping(patience=patience)
    model = model.to(device)
    epoch_losses = []
    
    for epoch in range(n_epochs):
        model.train()  # Training mode
        train_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = loss_fn(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        epoch_loss = train_loss/len(train_loader)
        epoch_losses.append(epoch_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                #print(inputs.shape)
                #print(targets.shape)
                inputs, targets = inputs.to(device), targets.to(device)
                
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        print(f"Epoch [{epoch+1}/{n_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Check early stopping
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

    # Load the best model after early stopping
    model.load_state_dict(torch.load(early_stopping.path))
    return model, epoch_losses



def get_trainable_params(model): 
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return pytorch_total_params



def train_model_no_early_stopping(model, optimizer, loss_fn, train_loader, n_epochs=10, device='cpu'): 
    model = model.to(device)  # Move model to the specified device (CPU or GPU)
    model.train()  # Set model to training mode
    
    
    epoch_losses = []
    for epoch in range(n_epochs):
        running_loss=0.0
        for i, (x_batch, y_batch) in enumerate(train_loader):
            
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            y_pred = model(x_batch)
            loss   = loss_fn(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Accumulate loss for the batch
            running_loss += loss.item()
            
             # Print stats every 10 batches
            #if (i + 1) % 10 == 0:
                #print(f"Epoch [{epoch + 1}/{n_epochs}], Batch [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        # Print average loss per epoch
        epoch_loss = running_loss / len(train_loader)
        epoch_losses.append(epoch_loss)
        print(f"Epoch [{epoch + 1}/{n_epochs}] Average Loss: {epoch_loss:.4f}")
        
    print("Training completed")
    return epoch_losses