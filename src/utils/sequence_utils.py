import numpy as np

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