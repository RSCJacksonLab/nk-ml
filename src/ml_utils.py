import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np
from sklearn.model_selection import train_test_split
import os




class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0, path='best_model_{}.pt'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            path (str): Path to save the best model.
        """
        self.patience = patience
        self.min_delta = min_delta
        
        
        self.path = path.format(np.random.randint(10000000,100000000 ))
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

def train_model(model, optimizer, loss_fn, train_loader, val_loader, n_epochs=30, device='cpu', patience=5, min_delta=1e-5):
    early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)
    model = model.to(device)
    val_epoch_losses = []
    train_epoch_losses = []

    assert n_epochs>0, 'n_epochs not strictly greater than 0, ensure n_epochs > 0'
    
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
        train_epoch_losses.append(epoch_loss)

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
        val_epoch_losses.append(val_loss)

        
            
            
        

        print(f"Epoch [{epoch+1}/{n_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Check early stopping
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

    # Load the best model after early stopping
    model.load_state_dict(torch.load(early_stopping.path))

    #delete best model from early stopiig from disk 
    if os.path.exists(early_stopping.path):
      os.remove(early_stopping.path)
    else:
      print("The file does not exist")
        
    return model, train_epoch_losses, val_epoch_losses



def get_trainable_params(model): 
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return pytorch_total_params



def train_model_get_latent_rep(model, model_name, optimizer, loss_fn, train_loader, val_loader, n_epochs=30, device='cpu', patience=5, min_delta=1e-5, x_data=None):
    early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)
    model = model.to(device)
    val_epoch_losses = []
    train_epoch_losses = []
    epoch_latent_reps  = []
    
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
        train_epoch_losses.append(epoch_loss)

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
        val_epoch_losses.append(val_loss)

        assert x_data!=None, 'No landscape x_data provided for latent representation calculation.'
        latent_rep  = get_latent_representation(model, model_name, x_data)
        epoch_latent_reps.append(latent_rep)

        
            
            
        

        print(f"Epoch [{epoch+1}/{n_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Check early stopping
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

    # Load the best model after early stopping
    model.load_state_dict(torch.load(early_stopping.path))
    return model, train_epoch_losses, val_epoch_losses, epoch_latent_reps





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
    

def train_val_test_split_ohe(landscapes, test_split=0.2, val_split=0.2, random_state=1): 
    """Performs train-test-val splitting of data using a list of protein landscape class objects. NOTE: validation data 
        proportion is proportion of TRAIN data NOT total data. 
    
    Args: FIX THIS FUNCTION TO MAKE FASTER AND NEATER BEFORE PUBLICATION
            landscapes (list): List of Protein_Landscape class objects
            test_split (float): proportion of total data used for testing
            val_split (float): proportion of TRAIN data used for validation (NOT total data)
            random_state (int):      controls random state of sklearn train_test_split for reproducible splits. Default 1. """
    LANDSCAPES_OHE = [np.array(i.one_hot_encodings) for i in landscapes]
    X_OHE = LANDSCAPES_OHE
    Y_OHE = [i.fitnesses.reshape(-1,1).astype(float) for i in landscapes]
    XY_OHE = [list(zip(torch.from_numpy(X_OHE[i]).to(torch.float32), torch.from_numpy(Y_OHE[i]).to(torch.float32))) for i in range(len(X_OHE))]
    XY_OHE_TRAIN_TEST_SPLIT = [train_test_split(i, test_size=round(len(i)*test_split), random_state=random_state) for i in XY_OHE]
    
    XY_TRAIN = [i[0] for i in XY_OHE_TRAIN_TEST_SPLIT]
    XY_TEST  = [i[1] for i in XY_OHE_TRAIN_TEST_SPLIT]
    XY_TRAIN_VAL_SPLIT = [train_test_split(i, test_size=round(len(i)*val_split), random_state=random_state) for i in XY_TRAIN]
    
    XY_TRAINING = [i[0] for i in XY_TRAIN_VAL_SPLIT]
    XY_VAL      = [i[1] for i in XY_TRAIN_VAL_SPLIT]


    X_TEST = []
    Y_TEST = []

    for ind, i in enumerate(XY_OHE_TRAIN_TEST_SPLIT):
        
        x_test = []
        y_test = []
        for x, y in XY_OHE_TRAIN_TEST_SPLIT[ind][1]: 
            x_test.append(x.numpy())
            y_test.append(y.numpy())

        x_test = torch.from_numpy(np.array(x_test))
        y_test = torch.from_numpy(np.array(y_test))
        
        X_TEST.append(x_test)
        Y_TEST.append(y_test)
    
    return LANDSCAPES_OHE, XY_TRAINING, XY_VAL, XY_TEST, X_TEST, Y_TEST




def landscapes_ohe_to_numpy(OHE_landscape_list):
    """ Function to convert OHE arrays output by train_val_test_split_ohe() into scikit-amenable flattened numpy arrays. 

        Arguments: 
            OHE_landscape_list (list):     list of landscapes formatted output by train_val_test_split_ohe

        Returns: 
            x_flat_array (np.array):        np.array of shape (len(OHE_landscape_list), n_samples, len(AA_alphabet)*sequence_length)
            y_flat_array (np.array):        np.array of shape (len(OHE_landscape_list), n_samples, 1)
    """

    xy_np_flattened = [[(i[0].numpy().flatten(), i[1].numpy()) for i in j] for j in OHE_landscape_list]
    x_flat_array = np.array([[i[0] for i in j] for j in xy_np_flattened])
    y_flat_array = np.array([[i[1] for i in j] for j in xy_np_flattened])
    
    return x_flat_array, y_flat_array