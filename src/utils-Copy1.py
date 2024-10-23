import numpy as np 
import pandas as pd




from scipy.stats import uniform, randint

#import sklearn_utils as sku

# utility code

import numpy as np
import itertools

def collapse_single(protein):
    """
    Takes any iterable form of a single amino acid character sequence and returns a string representing that sequence.
    """
    return "".join([str(i) for i in protein])

def hamming(str1, str2):
    """Calculates the Hamming distance between 2 strings"""
    return sum(c1 != c2 for c1, c2 in zip(str1, str2))

def all_genotypes(N, AAs):
    """Fills the sequence space with all possible genotypes."""
    return np.array(list(itertools.product(AAs, repeat=N)))

def neighbors(sequence, sequence_space):
    """Gets neighbours of a sequence in sequence space based on Hamming
    distance."""
    hammings = []
    for i in sequence_space:
        hammings.append((i, hamming(sequence,i)))
    return [sequence  for  sequence, distance in hammings if distance==1]

def custom_neighbors(sequence, sequence_space, d):
    """Gets neighbours of a sequence in sequence space based on Hamming
    distance."""
    hammings = []
    for i in sequence_space:
        hammings.append((i, hamming(sequence,i)))
    return [sequence  for  sequence, distance in hammings if distance==d]

def genEpiNet(N, K):
    """Generates a random epistatic network for a sequence of length
    N with, on average, K connections"""
    return {
        i: sorted(np.random.choice(
            [n for n in range(N) if n != i],
            K,
            replace=False
        ).tolist() + [i])
        for i in range(N)
    }

def fitness_i(sequence, i, epi, mem):
    """Assigns a (random) fitness value to the ith amino acid that
    interacts with K other positions in a sequence, """
    #we use the epistasis network to work out what the relation is
    key = tuple(zip(epi[i], sequence[epi[i]]))
    #then, we assign a random number to this interaction
    if key not in mem:
        mem[key] = np.random.uniform(0, 1)
    return mem[key]


def fitness(sequence, epi, mem):
    """Obtains a fitness value for the entire sequence by summing
    over individual amino acids"""
    sequence
    epi
    mem
    return np.mean([
        fitness_i(sequence, i, epi, mem) # ω_i
        for i in range(len(sequence))
    ])

def makeNK(N, K, AAs):
    """Make NK landscape with above parameters"""
    f_mem = {}
    epi_net = genEpiNet(N, K)
    sequenceSpace = all_genotypes(N,AAs)
    sequences = np.array([''.join(j) for j in [list(i) for i in sequenceSpace]])
    
    seqspace = [list(i) for i in list(sequenceSpace)]
    land = np.array([(x,y) for x, y in zip(sequenceSpace, [fitness(i, epi=epi_net, mem=f_mem) for i in sequenceSpace])])
    
    sequences = sequences.reshape(land.shape[0],1)
    fitnesses = np.array([float(i) for i in land[:,1]]).reshape(land.shape[0],1)
    
    return np.concatenate((sequences, fitnesses), axis=1, dtype='object')
    


def sklearn_split(data, split=0.8):
    """
    Takes a dataset array of two layers, sequences as the [:,0] dimension and fitnesses
    as the [:,1] dimension, shuffles, and returns the tokenized sequences arrays
    and retyped fitness arraysself.

    Parameters
    ----------
    data : np.array (N x 2)

        The sequence and fitness data with sequences provided as single amino acid strings

    split : float, default=0.8, range (0-1)

        The split point for the training - validation data

    returns : x_train, y_train, x_test, y_test

        All Nx1 arrays with train as the first 80% of the shuffled data and test
        as the latter 20% of the shuffled data.
    """

    assert (0 < split < 1), "Split must be between 0 and 1"

    np.random.shuffle(data)

    split_point = int(len(data)*split)

    train = data[:split_point]
    test  = data[split_point:]

    x_train = train[:,0]
    y_train = train[:,1]
    x_test  = test[:,0]
    y_test  = test[:,1]

    return x_train, y_train, x_test, y_test
import numpy as np

def amino_acid_to_one_hot(sequence, aminos='ACDEFGHIKLMNPQRSTVWY'):
    """
    Convert amino acid sequence to one-hot vector representation.

    Parameters:
    - sequence (str): Amino acid sequence.

    Returns:
    - numpy array: One-hot vector representation of the input sequence.
    """

    # Define the mapping of amino acids to indices
    amino_acids = aminos
    amino_acid_to_index = {aa: i for i, aa in enumerate(amino_acids)}

    # Initialize an array of zeros with shape (len(sequence), len(amino_acids))
    one_hot = np.zeros((len(sequence), len(amino_acids)))

    # Convert each amino acid in the sequence to a one-hot vector
    for i, aa in enumerate(sequence):
        if aa in amino_acid_to_index:
            one_hot[i, amino_acid_to_index[aa]] = 1

    return one_hot


def landscape_to_split(landscape_array, split=0.8, aminos='ACDEFGHIKLMNPQRSTVWY'):
    x_train, y_train, x_test, y_test = sklearn_split(landscape_array, split=split)
    #print(x_train.shape)
    #print(x_test.shape)
    x_train_tokenised = sku.sklearn_tokenize(x_train, aminos)
    x_test_tokenised  = sku.sklearn_tokenize(x_test, aminos)
    x_train_one_hot   = np.array([amino_acid_to_one_hot(i, aminos) for i in x_train])
    x_test_one_hot    = np.array([amino_acid_to_one_hot(i, aminos) for i in x_test])
    
    x_train_one_hot_shape = x_train_one_hot.shape
    x_test_one_hot_shape  = x_test_one_hot.shape
    
    #print(x_train_one_hot_shape)
    x_train_one_hot_flattened = x_train_one_hot.reshape(x_train_one_hot_shape[0], x_train_one_hot_shape[1]*x_train_one_hot_shape[2])
    x_test_one_hot_flattened  = x_test_one_hot.reshape(x_test_one_hot_shape[0], x_test_one_hot_shape[1]*x_test_one_hot_shape[2])
    
    train = {'x_train':x_train, 'x_train_tokenised':x_train_tokenised, 'x_train_one_hot_flattened':x_train_one_hot_flattened, 'y_train':y_train}
    test  = {'x_test':x_test, 'x_test_tokenised':x_test_tokenised, 'x_test_one_hot_flattened':x_test_one_hot_flattened, 'y_test':y_test}
    
    return train, test




