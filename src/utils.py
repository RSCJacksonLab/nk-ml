import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm.notebook import tqdm
from sklearn.utils import shuffle
from scipy.stats import rv_discrete


import scipy
from scipy import sparse
from itertools import chain, combinations, product
from scipy.special import comb
import itertools

def hamming_circle(s, n, alphabet):
    """Generate strings over alphabet whose Hamming distance from s is
    exactly n. 
    (Function taken direct from StackExchange -- https://codereview.stackexchange.com/questions/88912/create-a-list-of-all-strings-within-hamming-distance-of-a-reference-string-with)
    >>> sorted(hamming_circle('abc', 0, 'abc'))
    ['abc']
    >>> sorted(hamming_circle('abc', 1, 'abc'))
    ['aac', 'aba', 'abb', 'acc', 'bbc', 'cbc']
    >>> sorted(hamming_circle('aaa', 2, 'ab'))
    ['abb', 'bab', 'bba']
    """
    for positions in combinations(range(len(s)), n):
        for replacements in product(range(len(alphabet) - 1), repeat=n):
            cousin = list(s)
            for p, r in zip(positions, replacements):
                if cousin[p] == alphabet[r]:
                    cousin[p] = alphabet[-1]
                else:
                    cousin[p] = alphabet[r]
            yield ''.join(cousin)




def all_genotypes(N, AAs):
    """Generates all possible genotypes of length N over alphabet AAs ."""
    return np.array(list(itertools.product(AAs, repeat=N)))

def custom_neighbors(sequence, sequence_space, d):
    """Search algorithm for finding sequences in sequence_space that are exactly Hamming distance d from sequence.
    This is a possibly a slow implementation -- it might be possible to obtain speed-ups."""
    hammings = []
    for i in sequence_space:
        hammings.append((i, hamming_distance(sequence,i)))
    return [sequence  for  sequence, distance in hammings if distance==d]



def get_graph(sequenceSpace, AAs):
    """Get adjacency and degree matrices for a sequence space. This creates a Hamming graph by connecting all sequences 
    in sequence space that are 1 Hamming distance apart. Returns a sparse adjacency and degree matrix (which can be used
    for downstream applications e.g. Laplacian construction).
    sequenceSpace:      iterable of sequences
    returns:            tuple(adjacency matrix, degree matrix), where each matrix is a scipy sparse matrix """
  
    seq_space  = [''.join(i) for i in sequenceSpace]
    nodes      = {x:y for x,y in zip(seq_space, range(len(seq_space)))}
    adjacency  = sparse.lil_matrix((len(seq_space), len(seq_space)), dtype='int8') 
    
    for ind in tqdm(range(len(seq_space))):        
        seq = seq_space[ind]     

        for neighbor in hamming_circle(seq, 1,AAs): 
            adjacency[ind,nodes[neighbor]]=1 #array indexing and replacing can be quite slow; consider using lists

       # degree_matrix = (l*(a-1))*sparse.eye(len(seq_space)) #this definition comes from Zhou & McCandlish 2020, pp. 11
    return adjacency #returns adjacency




