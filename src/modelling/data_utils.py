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
    x_trn: ArrayLike,
    y_trn: ArrayLike,
    x_tst: ArrayLike,
    y_tst: ArrayLike,
):
    '''
    Given a trained Scikit Learn model, determine its
    performance on training.

    Parameters:
    -----------
    sklearn_model
        Trained SciKit learn model.

    x_trn : ArrayLike
        Training input data.

    y_trn : ArrayLike
        Training target data.

    x_tst : ArrayLike
        Testing input data.

    y_tst : ArrayLike
        Testing target data.
    '''
    # get train predictions
    trn_preds = sklearn_model.predict(x_trn)
    trn_mse = mean_squared_error(y_trn, trn_preds)
    trn_r2 = r2_score(y_trn, trn_preds)
    trn_corr = pearsonr(y_trn, trn_preds)[0]

    # get rest predictions
    tst_preds = sklearn_model.predict(x_tst)
    tst_mse = mean_squared_error(y_tst, tst_preds)
    tst_r2 = r2_score(y_tst, tst_preds)
    tst_corr = pearsonr(y_tst, tst_preds)[0]

    # result dict
    score_dict = {
        'train': {
            'pearson_r': trn_corr,
            'r2': trn_r2,
            'mse': trn_mse
        },
        'test': {
            'pearson_r': tst_corr,
            'r2': tst_r2,
            'mse': tst_mse
        }
    }
