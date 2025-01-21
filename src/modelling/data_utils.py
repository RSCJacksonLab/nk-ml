import numpy as np
import torch

from numpy.typing import ArrayLike
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
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

    # in case of lists etc.
    X = np.asarray(X).astype(float)
    y = np.asarray(y).astype(float)

    # convert the data into PyTorch tensors
    X_tensor = torch.tensor(np.array(X), dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    
    # make dataset
    dset = TensorDataset(X_tensor, y_tensor)

    return dset


def collapse_concat(arrays, dim=0):
    """
    Takes an iterable of arrays and recursively concatenates them. 
    Functions similarly to the reduce operation from python's functools
    library.

    (from https://github.com/acmater/NK_Benchmarking/)

    Parameters:
    -----------
    arrays : iterable(np.array)
        Arrays contains an iterable of np.arrays

    dim : int, default=0
        The dimension on which to concatenate the arrays.

    Returns:
    --------
        arr_concat : ArrayLike
            Asingle np array representing the concatenation of all arrays
            provided.
    """
    if len(arrays) == 1:
        return arrays[0]
    else:
        arr_concat = np.concatenate((arrays[0],
                                     collapse_concat(arrays[1:])),
                                     axis=dim)
        return arr_concat
    

def score_sklearn_model(
    sklearn_model,
    x: ArrayLike,
    y: ArrayLike,
):
    '''
    Given a trained Scikit Learn model, determine its
    performance on training.

    Parameters:
    -----------
    sklearn_model
        Trained SciKit learn model.

    x : ArrayLike
        Input data.

    y : ArrayLike
        Target data.
    '''
    # get predictions
    preds = sklearn_model.predict(x)

    # assess performance
    mse = mean_squared_error(y, preds)
    r2 = r2_score(y, preds)
    corr = pearsonr(y, preds)[0]


    # result dict
    return {
        'pearson_r': corr,
        'r2': r2,
        'mse': mse
    }
