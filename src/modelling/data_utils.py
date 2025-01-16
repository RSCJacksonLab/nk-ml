from numpy.typing import ArrayLike

from torch.utils.data import Dataset, TensorDataset
from typing import Tuple

def make_dataset(
        data: Tuple[ArrayLike, ArrayLike]) -> Dataset:
    '''
    Parse ArrayLike data into dataset. Expects to the first element
    to be features (X) and second to be targets (y).

    Parameters:
    -----------
    data : Tuple[ArrayLike, ArrayLike])
        Data to make dataset with.
    
    Returns:
    --------
    dset : Dataset
        Dataset containing tensor forms of provided inputs.
    '''
    X, y = data
    
    # convert the data into PyTorch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    
    # make dataset
    dset = TensorDataset(X_tensor, y_tensor)

    return dset
