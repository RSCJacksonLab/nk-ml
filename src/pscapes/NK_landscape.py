'''
NK Landscape Generation.
'''
import itertools
import numpy as np

from typing import Callable

def generate_epistatic_network(N: int, K: int):
    """
    Generates a random epistatic network that defines the NK landscape
    (for a sequence of length N with on average, K connections).

    Parameters:
    -----------
    N : int
        Number of positions

    K : int
        K parameter of NK landscape - "ruggedness" or number of
        connections per site.
    """
    return {
        i: sorted(np.random.choice(
            [n for n in range(N) if n != i],
            K,
            replace=False
        ).tolist() + [i])
        for i in range(N)
    }

def site_fitness_contribution(sequence: str,
                              site: int, 
                              epistatic_net: dict,
                              fitness_mem: dict,
                              distribution: Callable[..., float],
                              **kwargs):
    """
    Assigns a (random) fitness value to the amino acid at a given site
    that interacts with K other positions in a sequence.

    Parameters:
    -----------
    sequence : str
        Amino acid sequence

    site : int
        Index of amino acid in sequence

    epistatic_net : dict
        Epistatic network.

    fitness_mem : dict
        Memoized fitness values.

    distribution : 
        Distribution from which to draw random number. For use with, 
        NumPy distributions (e.g. np.random.normal).

    **kwargs
        kwargs associated with the provided distribution.
    """
    #we use the epistasis network to work out what the relation is
    first_key = tuple(zip(epistatic_net[site], 
                          sequence[epistatic_net[site]]))
    key = (site, 0) + first_key
    #then, we assign a random number to this interaction
    if key not in fitness_mem:
        fitness_mem[key] = distribution(**kwargs)
    return fitness_mem[key]


def get_fitness(sequence: str,
                epistatic_net: dict,
                fitness_mem: dict,
                distribution: Callable[..., float], 
                residue_fitnesses=False, 
                **kwargs):
    """
    Obtains a fitness value for the entire sequence by summing
    over individual amino acid contributions.

    Parameters:
    -----------
    sequence : str
        Amino acid sequence

    epistatic_net : dict
        Epistatic network.
    
    fitness_mem : dict
        Memoized fitness values for efficiency.

    distribution : 
        Distribution from which to draw random number. For use with, 
        NumPy distributions (e.g. np.random.normal).

    **kwargs
        kwargs associated with the provided distribution.
    """
    per_residue = [
        site_fitness_contribution(sequence, 
                                  site,
                                  epistatic_net,
                                  fitness_mem,
                                  distribution, 
                                  **kwargs)
        for site in range(len(sequence))
    ]
    if residue_fitnesses: 
        return np.mean(per_residue), per_residue
    else: 
        return np.mean(per_residue)

def min_max_scaler(x): 
    """Scales data in 1D array x to the interval [0,1]."""
    return (x - min(x))/(max(x) - min(x))

def make_NK(N: int,
            K: int,
            amino_acids: str, 
            distribution: Callable[..., float], 
            epistasis_network=None, 
            normalise_fitness: bool = True, 
            return_site_fitnesses = False, 
            **kwargs):
    '''
    Create an NK landscape given a sequence length N, epistatic
    parameter K, an amino acid alphabet AAs and a NumPy distribution
    from which to sample fitnesses.

    Parameters:
    -----------
    N : int
        Number of positions

    K : int
        K parameter of NK landscape

    amino_acids : str
        Alphabet of accessible amino acids

    distribution : function
        Distribution from which to draw random number. For use with, 
        NumPy distributions (e.g. np.random.normal).

    epistasis_network : dict, default=None
        Pre-defined epistasis network, if applicable.

    normalise_fitness : bool, default=True
        If True, fitnesses are scaled to interval [0,1].

    return_site_fitnesses : bool, default=False
        If True, returns individual amino acid contributions to fitness.

    Returns:
    --------
    landscape : Array
        Numpy array of shape (n, 2), where first column contains 
        sequences.

    '''
    
    assert N > K, "N must be greater than K"

    fitness_mem_dict = {}
    if epistasis_network is None: 
        epistasis_network = generate_epistatic_network(N, K)

    seq_space = np.array(list(itertools.product(amino_acids, repeat=N)))

    # calculate fitness values
    fitness_results = [
        get_fitness(
            site,
            epistasis_network=epistasis_network,
            mem=fitness_mem_dict,
            distribution=distribution,
            residue_fitnesses=return_site_fitnesses,
            **kwargs
        )
        for site in seq_space
    ]

    # convert formatting
    if return_site_fitnesses:
        fitness_tuple = np.array(fitness_results)
        fitness_vals = fitness_tuple[:, 0].astype(float)
        site_fitness_vals = fitness_tuple[:, 1]
    else:
        fitness_vals = np.array(fitness_results, dtype=float)

    # normalisation
    if normalise_fitness: 
        fitness_vals = min_max_scaler(fitness_vals)

    # recalculate seq_space so its easier to concat
    seq_space = np.array([''.join(list(i)) 
                          for i in itertools.product(amino_acids, repeat=N)])
    landscape = {x: y for x, y in zip(seq_space, fitness_vals)}
    
    if return_site_fitnesses: 
        return (landscape, 
                epistasis_network, 
                site_fitness_vals, 
                fitness_mem_dict)
    else: 
        return landscape


# def gen_distance_subsets(ruggedness,seq_len=5,library="ACDEFGHIKL",seed=None):
#     """
#     Takes a ruggedness, sequence length, and library and produces an NK landscape then separates it
#     into distances from a seed sequence.

#     ruggedness [int | 0-(seq_len-1)]  : Determines the ruggedness of the landscape
#     seq_len : length of all of the sequences
#     library : list of possible characters in strings
#     seed    : the seed sequence for which distances will be calculated

#     returns ->  {distance : [(sequence,fitness)]}
#     """

#     land_K2, seq, _ = makeNK(seq_len,ruggedness,library)

#     if not seed:
#         seed = np.array([x for x in "".join([library[0] for x in range(seq_len)])])

#     subsets = {x : [] for x in range(seq_len+1)}
#     for seq in land_K2:
#         subsets[hamming(seq[0],seed)].append(seq)

#     return subsets

# def dataset_generation(directory="../Data",seq_len=5):
#     """
#     Generates five instances of each possible ruggedness value for the NK landscape

#     seq_len
#     """

#     if not os.path.exists(directory):
#         os.mkdir(directory)

#     datasets = {x : [] for x in range(seq_len)}

#     for ruggedness in range(0,seq_len):
#         for instance in range(5):
#             print("Generating data for K={} V={}".format(ruggedness,instance))

#             subsets = gen_distance_subsets(ruggedness,seq_len)

#             hold = []

#             for i in subsets.values():
#                 for j in i:
#                     hold.append([collapse_single(j[0]),j[1]])

#             saved = np.array(hold)
#             df = pd.DataFrame({"Sequence" : saved[:,0], "Fitness" : saved[:,1]})
#             df.to_csv("{0}/K{1}/V{2}.csv".format(directory,ruggedness,instance))

#     print ("All data generated. Data is stored in: {}".format(directory))