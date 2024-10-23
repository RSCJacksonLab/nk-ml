"""
A class and set of methods to compute various ruggedness metrics of protein sequence-fitness landscape (genotype-phenotype) data. 
"""

# Author: Mahakaran Sandhu 
# BSD 3 License 


import numpy as np
import utils
import math
from collections import Counter
from scipy import stats
from statsmodels.tsa.stattools import acf
import random
import itertools
from itertools import combinations
from multiprocessing import Pool
from tqdm.notebook import tqdm

from sklearn.linear_model import Ridge
from sklearn import metrics

from itertools import product



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





def two_chunks(line): 
    """Chunks a string into chunks of 2. Taken from StackOverflow."""
    return [line[i:i+2] for i in range(0, len(line), 2)]


class RuggednessBase: 
    """Base class for ruggedness estimators""" 

    def __init__(self, landscape, alphabet='ACDEFGHIKLMNPQRSTVWY'): 

        self.landscape = landscape 
        self.sequences = landscape[:,0]
        self.fitnesses = np.array([float(i) for i in landscape[:,1]])

        self.landscape_dict = {x:float(y) for x,y in zip(self.sequences, self.fitnesses)}
        self.alphabet  = alphabet
        self.seq_len   = len(self.sequences[0])




class NMaximaEstimator(RuggednessBase): 
    """Class implements the n-maxima method"""


    @staticmethod
    def _neighbors(sequence, landscape_dict, incomplete=False, alphabet='ACDEFGHIKLMNPQRSTVWY'):
        """Gets 1 Hamming neighbors of some sequence, returning the neighbor's sequence and fitness. 
        sequence      :     input sequence (string)
        landscape_dict:     landscape formatted as a dictionary for rapid look-up
        incomplete    :     Whether the landscape is complete (i.e. all sequences for a theoretical size len(AAs)**len(sequence) are present) or
                                  incomplete. Boolean value, default incomplete is False because we expect a complete theoretical landscape.
        AAs           :     String of accessible amino acid alphabet.
        
        
        returns       :     numpy array of shape (number_neighbours, 2), where the first column is a the sequences (astype str) and the second
                            column is the fitnesses (astype str). This is the same formatting as landscape. """
        hamming1 = hamming_circle(sequence,1, alphabet)
        #print('Num theoretical neighbours = {}'.format(len(list(hamming1))))
        if incomplete: 
            neighbors = [i for i in hamming1 if i in landscape_dict]
        else: 
            neighbors = hamming1   
        fits = []
        for i in neighbors:
            fits.append([i,landscape_dict[i]])
        return np.array(fits)
    

        
    @staticmethod
    def _is_maximum(seed, landscape_dict, incomplete=False, alphabet='ACDEFGHIKLMNPQRSTVWY'):
        """Computes whether a a given sequence, fitness pair (seed) from a landscape is a local maximum.
            
            seed          :       array-like, containing a sequence as the first element and the fitness as the second element

            landscape_dict:       landscape formatted as a dictionary for rapid look-up

            incomplete    :       Whether the landscape is complete (i.e. all sequences for a theoretical size len(AAs)**len(sequence) are present) or
                                  incomplete. Boolean value, default incomplete is False because we expect a complete theoretical landscape.

            AAs           :       String of accessible amino acid alphabet.
            
            returns       :       Boolean(True, False) whether seed is a local maximum (True) or not (False)"""
        
        neighborhood = NMaximaEstimator._neighbors(seed[0], landscape_dict, incomplete=incomplete, alphabet=alphabet)  
        #print(neighborhood)
        if len(neighborhood)==0:
            return None
        else: 
            fitnesses    = neighborhood[:,-1].astype(float)
            score        = np.greater(float(seed[1]),fitnesses) #check if seed fitness is greater than neighbor fitnesses

            if np.isin(False, score): #if there is a False (i.e. there  is a fitness greater than seed's), return False
                return False        
            else:
                return True

    def fit(self, incomplete=False, multiprocessing=False, num_workers=6, verbose=False):
        """Calculates the number of local maxima in a landscape.

        Parameters
        ------------
    
        landscape:     a fitness landscape (as dictionary), corresponding to unique sequence-fitness pairs; the keys
                      contains strings corresponding to sequences; the values are floats corresponding to fitnesses.
    
        incomplete:    whether the landscape is complete (i.e. all sequences for a theoretical size |alphabet|**sequence_length are present) or
                       incomplete. Boolean value, default incomplete is False because we expect a complete theoretical landscape.

        multiprocessing: whether to use multiprocessing (bool)

        num_workers:      number of processors to use if multiprocessing is True

                      
        AAs:            string of accessible amino acid alphabet.



        Returns
        -------
        
        number of local maxima (int)
        """ 


        if multiprocessing: 
            grid = [[entry, self.landscape_dict, incomplete, self.alphabet] for entry in self.landscape]
            #print('start')
            pool    = Pool(num_workers)    
            #print('begin')
            result  = pool.starmap(NMaximaEstimator._is_maximum, (tqdm(grid) if verbose else grid))
            #print('begin')
            self.n_maxima = np.sum(result)
            self.n_lone_nodes = 'Multiprocessing does not support lone node calculation yet'
        else: 
            n    = 0
            lone = 0
            for entry in (tqdm(self.landscape) if verbose else self.landscape): 
                out    = NMaximaEstimator._is_maximum(entry, self.landscape_dict, incomplete=incomplete, alphabet=self.alphabet)
                if out == True:
                    n+=1
                elif out == None: 
                    lone+=1
            self.n_maxima = n            
            self.n_lone_nodes = lone
        return self






class RandomWalkEstimator(RuggednessBase):
    """Implements random walk methods"""
    
    @staticmethod
    def _random_walk(landscape_dict, seed, n_steps=50, alphabet='AGC', incomplete=False): 
        """Performs a random walk from a seed sequence on a fitness graph, taking steps of 1 Hamming distance for a total of n_steps steps.
        
        landscape   :     fitness landscape, numpy array of shape (sequences, fitnesses)

        seed        :     sequence to start random walk from (str)

        n_steps     :     number of steps to take in the random walk (int)

        AAs         :     alphabet of amino acids (str)

        incomplete  :     whether the landscape has missing values or is complete (e.g. theoretical) (bool)
        
        returns     :     dict with keys corresponding to sequences encountered as steps on the random walk, and values corresponding to the
                            associated fitnesses. 
        """
        traj         = {}      #initialise dict to collect the trajectory {sequence:fitness}
        current_seq  = seed    #initialise current state as the seed sequence
        
        #supply landscape as dictionary to reduce conversion overheads. 
        if type(landscape_dict) != dict:
            try:
                landscape_dict = dict(landscape_dict)
            except: 
                print('Could not convert landscape into dictionary.')
        for _ in range(n_steps): 
            traj[current_seq]   = landscape_dict[current_seq]
            hamming1            = list(hamming_circle(current_seq, 1, alphabet)) #calculate the  Hamming neighbors of the current sequence
            
            if incomplete: 
                neighbours = [i for i in hamming1 if i in landscape_dict]   #if incomplete, check that each Hamming neighbor exists in the landscape
            else: 
                neighbours = hamming1
            current_seq         = random.choice(neighbours) #take a random step to a neighbouring sequence
            
        return traj
           


    @staticmethod
    def _walk_undulation(traj, epsilon): 
        """Calculates a string of length len(traj) (i.e. the length of the walk on a fitness landscape) consisting of 3 characters (U,D,F) 
        that correspond to whether a given step on a landscape  walk is uphill (U), downhill (D) or flat (F). If the absolute value of change in fitness for a step in the
        random walk, |dF|, is less than epsilon, the step is considered flat.
        
        traj     :        dict with keys corresponding to sequences encountered as steps on the random walk, and values corresponding to the
                            corresponding fitnesses.

        epsilon  :        threshold for |dF| flatness classification (float)
        
        returns  :        string describing undulation of the walk (U,D,F)
        """
        out       = []   #initialise list for building string
        fitnesses = [float(i) for i in list(traj.values())]
        
        for index, fitness in enumerate(fitnesses): 
            if index != 0: 
                dF = fitness - fitnesses[index-1]
            else: 
                dF = fitness-fitnesses[-1] #periodic boundary if first step of walk 
            
            if abs(dF) >= epsilon: 
                if dF < 0: #downhill step
                    out.append('D')
                elif dF > 0: #uphill step
                    out.append('U')
            else: 
                out.append('F')
        output = ''.join(out)
        return output

        
    @staticmethod
    def _walk_entropy(undulation, shannon=False): 
        """Calculates the entropy of a random walk on a fitness landscape based on the undulation of that walk. Note that this function is 
            symbol-invariant.
            
            undulation:     string of characters (U,D,F) representing the undulation of a path (str)
            
            returns   :     information entropy of the path undulation"""
        
        
        if shannon: 
            return stats.entropy(list(Counter(undulation).values()))
        else:
            periodic     = undulation + undulation[0]  #add first element to the end to implement PBC
            first_pos    = two_chunks(periodic)     #chunk starting from first element
            second_pos   = two_chunks(periodic[1:]) #chunk starting from second element
            combined     = [i for i in first_pos+second_pos if len(i)>1] #combine chunking from positions 1 and 2 to come up with all possible 
                                                                         #chunks of length 2 under PBC
            
            frequencies  = dict(Counter(combined))    #determine frequencies 
            factor       = 1.0/sum(frequencies.values()) 
            distribution = {x: y*factor for x, y in frequencies.items() } #normalise frequencies to get probability distribution 
            
            entropy      = -sum([pr*math.log(pr, 6) for k,pr in distribution.items() if k[0]!=k[1] ]) #perform the entropy calculation on the distribution
            
            return entropy

    @staticmethod        
    def _autocorrelation(traj, lag):
        """Returns the autocorrelation function of a walk on a landscape."""
        return acf(list(traj.values()), fft=False, nlags=5)[lag]





    @staticmethod
    def _single_walk_calculations(landscape_dict, sequence_list, n_steps=50, alphabet='AGC', epsilon=0.1, lag=1, incomplete=False, shannon=False ): 
        """Perform entropy and autocor calculations given a single random walk on a landscape.Supply landscape as dict to reduce conversion
        overheads."""
        
        seed       = np.random.choice(sequence_list)
        traj       = RandomWalkEstimator._random_walk(landscape_dict, seed, n_steps=n_steps, alphabet=alphabet, incomplete=incomplete)
        undulation = RandomWalkEstimator._walk_undulation(traj, epsilon=epsilon)
        results    = [RandomWalkEstimator._walk_entropy(undulation, shannon=shannon), RandomWalkEstimator._autocorrelation(traj, lag=lag)]
        return results


    def fit(self, n_steps=50, n_walks=100, epsilon=0.1, incomplete=False,
                          shannon=False, multiprocessing=False, num_workers=6, lag=1):
        """Calculates entropy and autocorrelation of an ensemble of random walks on the landscape. 

        Parameters
        ----------

        landscape:  self.landscape

        n_steps:    number of steps to take on a given random walk (int, default 50)

        n_walks:    number of random walks to take

        alphabet:        alphabet

        epsilon:    threshold for (U,F,D) classification

        incomplete: whether the landscape is complete (i.e. all sequences for a theoretical size |alphabet|**sequence_length are present) or
                       incomplete. Boolean value, default incomplete is False because we expect a complete theoretical landscape.

        shannon:    whether to calculate Shannon entropy (bool, default False). Default is entropy as presented by Vassilev et al 2000

        multiprocessing:  whether to use multiprocessing (bool)

        num_workers:      number of processors to use if multiprocessing is True

        Returns
        --------
        average_entropy:   entropy averaged over n_walks (float)

        walk_entropies:    entropies of the individual walks (array-like)
        """

        if multiprocessing: 
            grid = [[self.landscape_dict,sequences, n_steps, self.alphabet, epsilon, incomplete,shannon] for replicate in range(n_walks)]
            if __name__ == '__main__':
                pool    = Pool(num_workers)    
                result  = pool.starmap(RandomWalkEstimator._single_walk_calculations, tqdm(grid))
                self.rw_statistics = result
        else: 
            result      = [RandomWalkEstimator._single_walk_calculations(self.landscape_dict, sequence_list=self.sequences, n_steps=n_steps, alphabet=self.alphabet, 
                                                   epsilon=epsilon, incomplete=incomplete, shannon=shannon) for replicate in range(n_walks)]
            res_array = np.array(result)
            means     = np.mean(res_array, axis=0)
            stds      = np.std(res_array, axis=0)

            self.rw_raw_statistics = res_array
            self.rw_statistics     = (means, stds)
            self.rw_nsteps         = n_steps
            self.rw_nwalks         = n_walks
            self.rw_epsilon        = epsilon
            self.shannon           = shannon
        return self








class EpistasisEstimator(RuggednessBase): 
    """Epistasis model that performs a basis expansion to additive and higher-order interactions, and computes contributions via linear regression. 



    Parameters
    ----------
    landscape :     numpy array of shape (number_examples, 2); first column contains the sequences, second column the fitnesses

    alphabet  :     accessible amino acid/nucleotide/other alphabet


    Attributes
    ----------

     '"""



    @staticmethod
    def _position_interactions(seq_len, order): 
        """Returns a list of tuples, where each tuple specifies the sequence indices for a unique interaction; degenerate interactions are not 
           returned (itertools.combinations, order does not matter), and neither are self-interactions. This is equavalent to returning the diagonal
           in a sequence-sequence interaction matrix. This is the basis expansion in terms of positions. 
           """
        f = list(itertools.combinations(range(seq_len), order)) #excludes the diagonal
        return f


    @staticmethod
    def _estimate_mem_usage(seq_len, orders, alphabet='EDRKHQNSTPGCAVILMFYW'): 
        """Returns estimate of memory usage of array assuming dtype int8, one-hot encoding"""
        total_val = 0
        for i in range(orders): 
            ordern     = i+1         
            num_values = len(EpistasisEstimator._position_interactions(seq_len, ordern))*(ordern**len(alphabet))
            total_val+=num_values
        return total_val

    @staticmethod
    def _get_encoding(order, alphabet='EDRKHQNSTPGCAVILMFYW'): 
        """Returns an encoding dictionary for the sequence space over 'alphabet' and 'order' number of positions. 

        Parameters
        ----------
        order:      order at which to compute encodings (i.e. number of positions to calculate sequence space over)

        alphabet:   alphabet over which to compute sequence space

        Returns
        ----------
        encoding:   dict of format {sequence: number} where 'sequence' is a unique element in the sequence space (over order and alphabet), and 
                    'number' is a unique integer identifier for 'sequence'."""

        seq_space = [''.join(i) for i in list(itertools.product(alphabet, repeat=order))]   # calculate the sequence space over 'order' and 'alphabet'
        encoding  = {x:y for x,y in zip(seq_space, range(len(seq_space)))}                  # assign unique integer to each element of calculated sequence space
        return encoding

    @staticmethod
    def _vector_encoding(sequence, encoding, positions): 
        """Returns a one-hot vector representation of a 'sequence''s sub-sequence at 'positions' according to 'encoding'.

        Parameters
        ----------
        sequence:   string of characters from defined alphabet

        encoding:   dict of format {sequence: number} where 'sequence' is a unique element in the sequence space (over order and alphabet), and 
                    'number' is a unique integer identifier for 'sequence'.

        positions:  tuple of positions specifying the indices of the interaction

        Returns
        ---------
        vector:     numpy array (dtype=int8) of shape (len(encoding),), with all 0 entries except 1 at index corresponding to the integer identifier 
                    from encoding corresponding to the identity of the subsequence 


         """
        encoding_len         = len(list(encoding.keys())[0])
        assert encoding_len == len(positions), f'Encoding length {encoding_len} not equal to number of positions {len(positions)}'
        
        position_aas = []    
        
        for i in positions: 
            position_aas.append(sequence[i])
        
        search_key   = ''.join(position_aas)   
        #print(search_key)
        
        vector = [0 for _ in range(len(encoding))]
        vector[encoding[search_key]]=1
        
        return np.array(vector, dtype="int8")

    @staticmethod        
    def _encode_epistasis(sequences, seq_len, orders, available_memory=5, alphabet='EDRKHQNSTPGCAVILMFYW'):
        """Returns one-hot encoding of 'sequence' with a basis expansion to 'orders' over 'alphabet' and 'seq_len'

        Parameters
        ----------
        sequences:          iterable of sequences (str)

        seq_len:            length of sequences, (int)

        orders:             orders to perform the basis expansion to (int)

        available_memory:   available RAM in GiB (int/float)

        alphabet:           accessible alphabet (str)

        Returns
        -------
        tuple(encoded_arrays, encodings, order_interact)

        encoded_arrays:     dict of format {order: array} specifying the one-hot encoded arrays at a given basis expansion order

        encodings:          nested dict of format {order: {sequence: number}} where, at a given key order, 'sequence' is a unique element in the sequence space (over order and alphabet), and 
                            'number' is a unique integer identifier for 'sequence'.

        ord_interact:       dict of format {order:[(pos1, pos2), (pos2, pos3), ...], ...} where, for each order, the value is a list of tuples where each tuple specifies the sequence indices for a unique interaction; degenerate interactions are not 
                            returned (itertools.combinations, order does not matter), and neither are self-interactions. This is equavalent to returning the diagonal
                            in a sequence-sequence interaction matrix."""

        mem_predict = EpistasisEstimator._estimate_mem_usage(seq_len=seq_len, orders=orders, alphabet=alphabet)/10**8
        print(f'The encoding matrix will use ~ {mem_predict} GiB of memory')
        
        assert available_memory>mem_predict, f'Required memory {mem_predict} GiB exceeds available memory {available_memory} GiB.'
        
        
        
        encodings      = {1:0}    
        encoded_arrays = {}
        ord_interact   = {}
        
        encoded_arrays[1] = utils.one_hot(sequences, AAs=alphabet) #calculate simple one-hot encoding for the first order interactions (additive)
        
        if orders>1:                                                                                            # only execute below codeblock if orders>1 requested
            
            for i in range(orders-1):  
                
                order               = i+2                                                                       # to escpape python 0 indexing and get into 1 indexing
                encodings[order]    = EpistasisEstimator._get_encoding(order, alphabet=alphabet)                # get the encoding for this order 
                ord_interact[order] = EpistasisEstimator._position_interactions(seq_len=seq_len, order=order )  # calculate all possible non-redundant interactions for this order (the basis expansion in positions)
                all_encodings = []                                                                              # init list for collecting encoded sequences -- will contain all encoded sequences at a given order 
                
                for seq in sequences:                                                                           # iterate over sequences
                    at_position = []                                                                            # initialise list to collect vector encoding of sequence at a given interaction
                    for inter in ord_interact[order]:                                                           # iterate over all possible non-redundant interactions for this order (the basis expansion in positions)
                        vec_encode = EpistasisEstimator._vector_encoding(sequence=seq, encoding=encodings[order], positions=inter) # calculate one-hot vector for this sequence at a given interaction
                        at_position.append(vec_encode)                                                          
                    all_encodings.append(np.array(at_position, dtype='int8'))                                   # append the sequence encoding over all interactions to all_encodings
                encoded_arrays[order]=np.array(all_encodings, dtype='int8')                                     # append to encoded_arrays
                
        return encoded_arrays, encodings, ord_interact 
    
    @staticmethod
    def _flatten_concat_encoded_arrays(encoded_arrays):
        """Flattens and concatenates all arrays in encoded_arrays in a consistent way."""
        collection = [i.reshape(i.shape[0], i.shape[1]*i.shape[2]) for i in encoded_arrays.values()]
        concat     = np.concatenate(collection, axis=1) 
        return concat



    @staticmethod
    def _get_epistasis_coefficients(model, seq_len, encodings, ord_interact, alphabet='EDRKHQNSTPGCAVILMFYW'): 
        """ Returns epistasis coefficients for each order, formatted as an 'order+1'-dimensional tensor of shape (num_interactions, |alphabet|, |alphabet|,..., ) 
            with |alphabet| repeated 'order' times.

            Parameters
            ----------

            model:          fitted sklearn linear model (choice from LinearRegression, Lasso, Ridge or ElasticNet)

            seq_len:        sequence length (int)

            encodings:      nested dict of format {order: {sequence: number}} where, at a given key order, 'sequence' is a unique element in the sequence space (over order and alphabet), and 
                            'number' is a unique integer identifier for 'sequence'.

            ord_interact:   dict of format {order:[(pos1, pos2), (pos2, pos3), ...], ...} where, for each order, the value is a list of tuples where each tuple specifies the sequence indices for a unique interaction; degenerate interactions are not 
                            returned (itertools.combinations, order does not matter), and neither are self-interactions. This is equavalent to returning the diagonal
                            in a sequence-sequence interaction matrix.
            
            alphabet:           accessible alphabet (str)

            Returns
            ---------
            
            ord_coeffs:     dict of format {order:coefficient_array} containing epistasis coefficients at a given order. coefficient_array has shape
                            (num_interactions, |alphabet|, |alphabet|,..., ) with |alphabet| repeated 'order' times; each entry in the tensor corresponds 
                            to a particular interaction element in the sequence space at that position.
        """



        ord_coeffs         = {}    # initialise dict for collecting coefficients 

        all_coeffs         = model.coef_.reshape(-1)            # reshaping the model's coefficient array
        orders             = list(encodings.keys())             # obtaining a list of the orders of the basis expansion 
        first_order_coeffs = all_coeffs[0:seq_len*len(alphabet)].reshape(seq_len,len(alphabet)) # obtaining the coefficients of the first order basis by indexing seq_len*len(alphabet)
        ord_coeffs[1]      = first_order_coeffs                 # appending the first order coefficients to the output dictionary 

        prev = seq_len*len(alphabet)                            # variable for the lower bound of the current indexing of coefficients    

        for order in orders[1:]:                                # loop over second and high orders coefficients
            seq_space_size   = len(encodings[order])            # get length of encodings for order (i.e. size of the sequence space at each interaction at that order)
            num_interacts    = len(ord_interact[order])         # get the number of interactions             
            num_coefficients = seq_space_size*num_interacts     # the number of coefficients at a given order equals the sequence space size times the number of interactions 
            
            alphabet_size    = len(alphabet)                    # seq_space_size^(1/order) -- given that seq_space = |alphabet|^order, (and order is just num of positions that are interacting), this sqrt returns |alphabet|; one could equall
            r                = tuple(alphabet_size for _ in range(order)) # this returns a tuple of alphabet_size repeated order times. The idea here is to obtain a tensor which records the evaluation of that interaction for the entire sequence space  
            new_shape        = tuple((num_interacts,)) + r      # and this returns the shape -- which is essentially the number of interactions times the shape of the tensors. The issue here is whether this will be done in a consistent way 

            order_coefficients  = all_coeffs[prev:prev+num_coefficients].reshape(new_shape)   # we do the indexing to get the relevant coefficients, and then we reshape to get them how we want them         
            ord_coeffs[order]   = order_coefficients
            
            prev+=num_coefficients
            
        return ord_coeffs 


    def fit_linear_model(self, model, available_memory=5, orders=1, **model_kwargs): 
        """The primary class method. Fits a sklean linear regression model on an expanded sequence basis to obtain epistatic coefficients.

        Parameters
        ----------

        model:              sklean linear regression model (LinearRegression, Lasso, Ridge or ElasticNet)

        available_memory:   available RAM in GiB (int/float)

        orders:             order of basis expansion

        **model_kwargs:     parameters of the sklearn linear regression model"""


        seq_len   = len(self.sequences[0])
        self.is_fitted         = True
        self.linear_fit_orders = orders
        

        encoded_arrays, encodings, ord_interact = EpistasisEstimator._encode_epistasis(sequences=self.sequences, seq_len=seq_len,
                                                                   orders=orders, available_memory=available_memory,
                                                                   alphabet=self.alphabet)  

        flattened_arrays = EpistasisEstimator._flatten_concat_encoded_arrays(encoded_arrays)
        
        linear_model = model(**model_kwargs).fit(flattened_arrays, self.fitnesses.reshape(-1,1))
        
        self.model_params      = linear_model.get_params()
        self.linear_model      = linear_model
        self.flattened_arrays  = flattened_arrays
        self.encoded_arrays    = encoded_arrays
        self.encodings         = encodings
        self.ord_interact      = ord_interact
        self.model_score       = linear_model.score(flattened_arrays, self.fitnesses.reshape(-1,1))

        return self


    




class RSEstimator(EpistasisEstimator):
    """Get r/s ratio. Only really works with Ridge; Linear and others give spurious results"""

    @staticmethod
    def _extract_rs(model, X, y_true):
        predicts = model.predict(X)
        s        = np.mean(np.absolute(model.coef_))
        r        = np.sqrt(metrics.mean_squared_error(y_true, predicts))
        return r/s 

    def fit(self, model, available_memory=5, **model_kwargs):


        self.fit_linear_model(model=model, available_memory=available_memory, orders=1, **model_kwargs)
        self.rs_ratio = RSEstimator._extract_rs(self.linear_model, self.flattened_arrays, self.fitnesses.reshape(-1,1))
        return self


class FourierEstimator(EpistasisEstimator): 

    def fit(self, model, available_memory=5, **model_kwargs): 
        """Get a Fourier basis for the sequence space and fit the coefficients using linear regression."""
        
        self.orders = self.seq_len
    

        self.fit_linear_model(model=model, available_memory=available_memory, orders=self.orders, **model_kwargs)
        self.order_coefficients = self._get_epistasis_coefficients(self.linear_model, self.seq_len, self.encodings, self.ord_interact, self.alphabet)

        order_contribution = {}

        total_contribution = np.sum(np.square(self.linear_model.coef_))

        for order in self.order_coefficients.keys():
            beta = np.sum(np.square(self.order_coefficients[order]))
            contribution = beta/total_contribution
            order_contribution[order] = contribution

        self.order_contributions = order_contribution

        self.f_sum = np.sum(np.array([ i for i in list(self.order_contributions.values())[1:] ]))

        return self


        """ Things to check:
                1. consistency of _interpret_epistasis -- is the reshape working how we want it to? Remember the pytorch accident (DONE, to a reasonabe level)
                2. go through the entire epistasis estimator class line by line and ensure you understand how it works, and 
                    write out how it works; MAKE SURE IT IS CONSISTENT!! 
                3. How do we know that the linear regression is assigning variance appropriately? (DONE, but continue to learn)
                4. Add other epistasis measurements (TODO later)
                5. Add subgraphs averaging method  (TODO later)
                6. Add doctrings that are very informative 
                7. make a NK landscape class """



class NonLinearEpistasis(EpistasisEstimator): 
    """Implements the method of Sailer and Harms 2017"""
    pass 

class GlobalEpistasis(): 
    pass 

class MinimumEpistasis(): 
    pass 













        



