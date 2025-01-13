import numpy as np

from numpy.typing import ArrayLike

def aa_to_ohe(sequence: str, amino_acids: str='ACDEFGHIKLMNPQRSTVWY'):
        """
        Convert amino acid sequence to one-hot vector representation.

        Parameters 
        -----------
        sequence : str 
            Amino acid sequence.

        Returns:
        --------
        NumPy array: One-hot vector representation of the input 
        sequence.
        """

        # Define the mapping of amino acids to indices
        amino_acid_to_index = {aa: i for i, aa in enumerate(amino_acids)}

        # Initialize an array of zeros with shape (len(sequence), len(aas))
        one_hot = np.zeros((len(sequence), len(amino_acids)))

        # Convert each amino acid in the sequence to a one-hot vector
        for i, aa in enumerate(sequence):
            if aa in amino_acid_to_index:
                one_hot[i, amino_acid_to_index[aa]] = 1

        return one_hot

def token_data_to_ohe(tokens: ArrayLike, 
                      amino_acids: str='ACDEFGHIKLMNPQRSTVWY'):
    '''
    Convert an array of tokens and labels into individual target and
    OHE components. 

    Parameters:
    -----------
    tokens : ArrayLike
        Array of tokens - expects shape of (N, L)

    amino_acids : str
            The amino acid alphabet to use for the both the token and OHE
            representation.

    Returns:
    --------
    ohe_arr : ArrayLike
        Array of one-hot encoded sequences.
    '''
    n_seqs, seq_len = tokens.shape
    n_tokens = len(amino_acids)

    # make ohe for all seqs
    ohe_arr = np.zeros((n_seqs, seq_len, n_tokens), dtype=int)
    for idx, seq in enumerate(tokens):
         ohe_arr[idx, np.arange(seq_len), seq] = 1
    ohe_arr_flattened = ohe_arr.reshape(n_seqs, seq_len * n_tokens)
    return ohe_arr_flattened