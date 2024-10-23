import numpy as np
#import networkx as nx
from   tqdm.notebook import tqdm
from   itertools import chain, combinations, product
from   scipy import sparse
import itertools
import matplotlib.pyplot as plt
 
"""
TO DO: 
    1. Fix adjacency and graph functions to get rid of redundancies
    2. Fix NK code so it is consistent and functional as designed
        a. transition to use of landscape as dictionary. Fix data types (especially float issue and incompatibilities in fitnesses)
    3. Fix nmaxima code so it uses a single type 

"""



#####################################################HELPER FUNCTIONS FOR GENERATING LANDSCAPES##############################################

def hamming(str1, str2):
    """Calculates the Hamming distance between 2 strings."""
    return sum(c1 != c2 for c1, c2 in zip(str1, str2))

def all_genotypes(N, AAs):
    """Returns sequence space of alphabet AAs for N positions.
    N:          number of positions (int)
    AAs         alphabet of amino acids to pick from (str)"""
    return np.array(list(itertools.product(AAs, repeat=N)))

def hamming_circle(s, n, alphabet):
    """Generate strings over alphabet whose Hamming distance from s is
    exactly n.
    
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


def genEpiNet(N, K): 
    """Generates a random epistatic network for a sequence of length
    N (int) with, on average, K connections (int)"""
    return {
        i: sorted(np.random.choice(
            [n for n in range(N) if n != i], 
            K, 
            replace=False
        ).tolist() + [i])
        for i in range(N)
    }

def fitness_i(sequence, i, epi, mem, distribution, **kwargs):
    """Assigns a random fitness value to the ith amino acid that interacts with K other positions in a sequence.
    Fitnesses are drawn from a specified distribution (e.g. numpy.random.normal), with **kwargs passed to the 
    distribution sampling function. """

    key_0 = tuple(zip(epi[i], sequence[epi[i]]))    #we use the epistasis network to work out what the relation is  
    key   = (i, 0) + key_0

    if key not in mem:
        mem[key] = distribution(**kwargs)           #then, we assign a random number from distribution to this interaction
    return mem[key]


def fitness(sequence, epi, mem, distribution, residue_fitnesses=False, **kwargs):
    """Obtains a fitness value for the entire sequence by obtaining the mean over individual amino acids contributions."""
    per_residue = [fitness_i(sequence, i, epi, mem, distribution, **kwargs) for i in range(len(sequence))]
    if residue_fitnesses: 
        return np.mean(per_residue), per_residue
    else: 
        return np.mean(per_residue)

def min_max_scaler(x): 
    """Scales data in 1D array x to the interval [0,1]."""
    return (x-min(x))/(max(x)-min(x))

#####################################################LANDSCAPE FUNCTIONS###########################################################

def RMF(G, seq_space, distribution,c_weight=0.5, wildtype=0, **kwargs):
    """Make vanilla RMF with networkx (returns nx.Graph)"""
    for node in G.nodes():
        fitness = distribution(**kwargs) - (c_weight*hamming(seq_space[wildtype], seq_space[node]))
        G.nodes[node]['fitness']  = float(fitness)
        G.nodes[node]['sequence'] = ''.join(list(seq_space[node]))
    return G


def make_RMF(N, AAs, distribution, wildtype_index=0, c_weight=0.5, minmax=True, **kwargs): 

    """Computes a fitness landscape over a sequence space of size (len(AAs))**N using the simplified Rough Mount Fuji model 
    (reference: 10.1088/1742-5468/2013/01/P01005). AAs is the accessible amino acids and N is the number of positions. Fitness is given 
    by a random number (R) from a specified distribution, from which a penalty term c_weight*hamming(WT, seq_i) is subtracted
    (i.e. fitness = R - c_weight*hamming(WT,seq_i). Under this definition of fitness, Hamming distance from the wildtype sequence
    is penalized, with the penalty weight given by c_weight. If minmax=True (recommended), all calculated fitnesses are scaled to
    the interval [0,1]. **kwargs gives the keyword arguments for the distribution being used. 

    Arguments: 
         N:                     number of positions (int)

         AAs:                   alphabet of accessible amino acids (str)

         distribution:          distribution from which to draw random number R. Typically, Numpy distributions (e.g. np.random.normal) is used. If 
                                supplying own distribution function, ensure that it behaves like Numpy distributions. (function)

         wildtype_index:        index of sequence in sequence space which will be assigned as the wild-type sequence (int)

         c_weight:              penalty term weight (float)

         minmax:                if True, fitnesses are scaled to interval [0,1]. (bool)

         **kwargs:              keyword arguments for distribution function. 

    Returns: 
         landscape:             numpy array of shape (n, 2), where first column contains sequences, and second column contains fitnesses"""


    seq_space = all_genotypes(N, AAs)
    
    seqs = []
    fits = []
    
    for index, sequence in enumerate(seq_space): 
        fitness = distribution(**kwargs)-(c_weight*hamming(seq_space[wildtype_index], seq_space[index]))
        seqs.append(''.join(list(seq_space[index])))
        fits.append(float(fitness))
        
    seqs = np.array(seqs).reshape((len(seqs), 1))
    fits = np.array(fits).reshape((len(fits), 1))
    
    if minmax: 
        fits  = min_max_scaler(fits)
    
    landscape = np.concatenate((seqs, fits), axis=1)  
    
        
    return landscape



def make_NK(N, K, AAs, distribution, epi_net=None, minmax=True, residue_fitnesses=False, **kwargs): 
    """Make NK landscape with above parameters"""
    
    assert N>K, "K is greater than or equal to N. Please ensure K is strictly less than N."
    f_mem             = {}
    if epi_net is not None: 
        epistasis_network = epi_net
    else: 
        epistasis_network = genEpiNet(N, K)
    seq_space         = np.array(list(itertools.product(AAs, repeat=N)))

    if residue_fitnesses:
        fitness_tuple     = np.array([fitness(i, epi=epistasis_network, mem=f_mem, distribution=distribution, residue_fitnesses=residue_fitnesses, **kwargs) for i in seq_space])
        fitnesses         = np.array([float(i) for i in fitness_tuple[:,0]])
        fitnesses_i       = fitness_tuple[:,1]

    else: 
        fitnesses         = np.array([fitness(i, epi=epistasis_network, mem=f_mem, distribution=distribution, **kwargs) for i in seq_space])
        

    if minmax: 
        fitnesses     = min_max_scaler(fitnesses)


    seq_space         = np.array([''.join(list(i)) for i in itertools.product(AAs, repeat=N)]) #recalculate seq_space so its easier to concat
    
    fitnesses         = fitnesses.reshape((len(fitnesses),1))
    seq_space         = seq_space.reshape((len(seq_space),1))

    landscape         = (seq_space, fitnesses)
    
    if residue_fitnesses: 
        return landscape, epistasis_network, fitnesses_i, f_mem
    else: 
        return landscape






###############################################################GRAPH TOOLS###############################################################

### THE FOLLOWING FUNCTIONS ARE DEPRECATED; LOOK INTO UTILS.PY; THESE FUNCTIONS WILL BE REMOVED

def get_adjacency(sequenceSpace, AAs):
    """Get adjacency matrix for a sequence space. This creates a Hamming graph by connecting all sequences 
    in sequence space that are 1 Hamming distance apart. Returns a sparse adjacency matrix (which can be used
    for downstream applications e.g. Laplacian construction).

    sequenceSpace:      iterable of sequences
    returns:            adjacency matrix, a scipy sparse CSC matrix """
  
    seq_space  = [''.join(list(i)) for i in sequenceSpace]
    members    = set(seq_space)
    nodes      = {x:y for x,y in zip(seq_space, range(len(seq_space)))}
    connect    = sparse.lil_matrix((len(seq_space), len(seq_space)), dtype='int8') 
    
    for ind in tqdm(range(len(sequenceSpace))):        
        seq = sequenceSpace[ind]     

        for neighbor in hamming_circle(seq, 1,AAs): 
            connect[ind,nodes[neighbor]]=1 

        #degree_matrix = (l*(a-1))*sparse.eye(len(seq_space)) #this definition comes from Zhou & McCandlish 2020, pp. 11
    return connect.tocsc()



def make_graph(adjacency, landscape):
    G         = nx.convert_matrix.from_numpy_matrix(adjacency)
    fitnesses = [float(i) for i in landscape[:,1]]
    sequences = landscape[:,0]
    print(len(fitnesses))
    for node in G.nodes():
        G.nodes[node]['fitness']  = fitnesses[node]
        G.nodes[node]['sequence'] = sequences[node]
    return G

def draw_G(G, **kwargs): 
    plt.figure(figsize=(7,7))
    nx.draw(G, node_color = list(nx.get_node_attributes(G,'fitness').values()),
            cmap = plt.cm.get_cmap('Blues'),
            labels = nx.get_node_attributes(G, 'sequence'), **kwargs)




#####################################################FUNCTIONS FOR RUGGEDNESS CALCULATIONS###################################################




def neighbors(sequence, landscape_dict, AAs='ACDEFGHIKLMNPQRSTVWY'):
    """Gets neighbours of a sequence in sequence space based on Hamming 
    distance."""
    neighbors = hamming_circle(sequence,1, AAs)
    fits = []
    for i in neighbors:
        fits.append([i,landscape_dict[i]])
    return np.array(fits)
    

        
    
def is_maximum(seed, landscape_dict, experimental=False, AAs='ACDEFGHIKLMNPQRSTVWY'): 

    if experimental:
        h_neighbors   = generate_mutations(seed[0], AAs=AAs)
        #exp_neighbors = [seq for seq in h_neighbors if seq in landscape]
        n_data        = np.array([[x,landscape_dict[x]] for x in h_neighbors if x in landscape_dict])        
    else:    
        n_data = np.array(neighbors(seed[0], landscape_dict, AAs=AAs))
   # print(n_data)
    fits = n_data[:,-1].astype(np.float)
   # print(seed[1], fits)
    score = np.greater(float(seed[1]),fits) #check if seed fitness is greater than neighbor fitnesses//   
    if np.isin(False, score): #if there is a False (i.e. there  is a fitness greater than seed's), return False
        return False        
    else:
        return True
    
def get_nmaxima(landscape_sub, landscape_dict, experimental=False, AAs='ACDEFGHIKLMNPQRSTVWY'):
    """Given a landscape (formatted as a dictionary), returns the number of local maxima in the landscape."""

    n = 0
    as_array = [[x,y] for x, y in landscape_sub.items()]
    for i in as_array: 
        out = is_maximum(i,landscape_dict, experimental=experimental, AAs=AAs)
        if out==True:
            n+=1
    #print('Process time: {} seconds'.format(e-s))
    return n



#####################################DISTRIBUTIONS############################################3

def lomax_PDF(alpha, lamb, x): 
    """Return Lomax probability density function for random variables in x. 
    Properties of Lomax. Mean   --> lambda/(alpha-1) (defined only for alpha>1)
                         Median --> 
    
    alpha:      alpha parameter of PDF (int)
    lamb:       lambda parameter of PDF (int)
    x:          Numpy array of random variables over which PDF is calculated"""
    return (alpha*np.power(lamb, alpha))/np.power((x+lamb), alpha+1)

def lomax_CDF(alpha, lamb, x): 
    """Return Lomax cumulative distribution function for random variables in x. 
    
    alpha:      alpha parameter of CDF (int)
    lamb:       lambda parameter of CDF (int)
    x:          Numpy array of random variables over which CDF is calculated"""
    return 1-np.power((1+(x/lamb)), -alpha)

def sample_lomax(alpha, lamb, size=1, cutoff=None): 
    """Samples the Lomax distribution using the inversion method 
    (https://web.mit.edu/urban_or_book/www/book/chapter7/7.1.3.html). """  
    
    x       = np.random.uniform(size=size)
    inverse = lamb*(np.power((-x+1), -1/alpha) -1)
    
    if cutoff !=None: 
        while inverse > cutoff:
            x       = np.random.uniform(size=size)
            inverse = lamb*(np.power((-x+1), -1/alpha) -1)    
            
    return(inverse)