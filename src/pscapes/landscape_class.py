'''
Class for handling a protein dataset.

Modification of code from https://github.com/acmater/NK_Benchmarking/
    * Added optional use for benchmarking models trained on OHE rather
    than tokenization
    * Updated data splitting to be deterministic
'''
import copy
import multiprocessing as mp
import numpy as np
import pandas as pd
import pickle
import scipy.sparse as sp
import tqdm as tqdm

from colorama import Fore, Style
from functools import partial, reduce
from numpy.typing import ArrayLike
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from typing import Optional, Union

from utils import aa_to_ohe, collapse_concat, token_data_to_ohe

class ProteinLandscape():
    '''
    Class that handles a protein dataset

    Parameters:
    -----------
    data : np.array
        Numpy Array containg protein data. Expected shape is (Nx2),
        with the first column being the sequences, and the second being
        the fitnesses.

    seed_seq : str, default=None
        Enables the user to explicitly provide the seed sequence as a
        string.

    seed_id : int,default=0
        Id of seed sequences within sequences and fitness.

    csv_path : str,default=None
        Path to the csv file that should be imported using CSV loader
        function.

    custom_columns : {"x_data"    : str,
                      "y_data"    : str
                      "index_col" : int}, default=None
        First two entries are custom strings to use as column headers
        when extracting data from CSV. Replaces default values of
        "Sequence" and "Fitness". Third value is the integer to use as
        the index column. They are passed to the function as keyword
        arguments.

    amino_acids : str, default='ACDEFGHIKLMNPQRSTVWY'
        String containing all allowable amino acids for tokenization 
        functions

    saved_file : str, default=None
        Saved version of this class that will be loaded instead of
        instantiating a new one

    Attributes:
    -----------
    amino_acids : str, default='ACDEFGHIKLMNPQRSTVWY'
        String containing all allowable amino acids in tokenization
        functions.

    sequence_mutation_locations : np.array(bool)
        Array that stores boolean values with Trues indicating that the
        position is mutated relative to the seed sequence.

     mutated_positions: np.array(int)

        Numpy array that stores the integers of each position that is
        mutated.

    d_data : {distance : index_array}
        A dictionary where each distance is a key and the values are the
        indexes of nodes with that distance from the seed sequence.

    data : np.array
        Full, untokenized data. Two columns, first is sequences as 
        strings, and second is fitnesses

    tokens : {tuple(tokenized_sequence) : index}
        A dictionary that stores a tuple format of the tokenized string
        with the index of it within the data array as the value. Used to
        rapidly perform membership checks

    sequences : np.array(str)
        A numpy array containing all sequences as strings.

    seed_seq : str
        Seed sequence as a string.

    tokenized : np.array, shape(N,L+1)

        Array containing each sequence with fitness appended onto the
        end. For the shape, N is the number of samples, and L is the
        length of the seed sequence

    mutation_array : np.array, shape(L*20,L)
        Array containing all possible mutations to produce sequences 1
        amino acid away. Used by maxima generator to accelerate the 
        construction of the graph. L is sequence length.

    self.hammings : np.array(N,)
        Numpy array of length number of samples, where each value is the
        hamming distance of the species at that index from the seed
        sequence.

    max_distance : int
        The maximum distance from the seed sequence within the dataset.

    graph : {tuple(tokenized_seq) : np.array[neighbour_indices]}
        A memory efficient storage of the graph that can be passed to
        graph visualisation packages.

    num_minima : int
        The number of minima within the dataset.

    num_maxima : int
        The number of maxima within the dataset.

    extrema_ruggedness : float32
        The floating point ruggedness of the landscape calculated as
        the normalized number of maxima and minima.

    Written by Adam Mater, last revision 04-09-20   
    Updated by Dana Matthews and Mahakaran Sandhu, 11-01-25
    '''
    def __init__(self,
                 data: Optional[ArrayLike] = None,
                 seed_seq: Optional[str] = None,
                 seed_id: int = 0,
                 graph: bool = False,
                 csv_path: Optional[str] = None,
                 custom_columns: dict = {"x_data": "sequence",
                                         "y_data": "fitness",
                                         "index_col": None},
                amino_acids: str='ACDEFGHIKLMNPQRSTVWY',
                saved_file: Optional[str] = None,
                calculate_graph: bool = False, 
                calculate_ruggedness: bool = False 
                ):
        
        if saved_file is not None:
            try:
                print(saved_file.suffix)
                self.load(saved_file)
            except:
                e = "File could not be loaded."
                raise FileNotFoundError(e)
            
        # load provided data
        if csv_path:
            self.csv_path = csv_path
            self.data = self.csv_dloader(csv_path,
                                         **custom_columns)
            if data is not None:
                print("File path has been prioritized over data array.")
        
        elif data:
            self.data = data

        else:
            raise FileNotFoundError("No data provided.")
        
        # assign attributes
        self.amino_acids = amino_acids
        self.tokens = {
            x: y for x, y in zip(self.amino_acids,
                                 list(range(len(self.amino_acids))))
        }
        self.graph = None
        # data is stored with sequences in the first column and fitnesses
        self.sequences = self.data[:, 0]
        self.fitnesses = self.data[:, 1]
        self.ohe = self.return_ohe()

        # assign sequence properties
        if seed_seq:
            self.seed_seq = seed_seq
        else:
            self.seed_seq = seed_id
            self.seed_seq = self.sequences[seed_id]

        seq_len = len(self.seed_seq)
        self.tokenized = np.concatenate((self.tokenize_data(),
                                         self.fitnesses.reshape(-1,1)),
                                         axis=1)
        self.token_dict = {tuple(seq): idx 
                           for idx, seq in enumerate(self.tokenized[:,:-1])}
        self.mutated_positions = self.calc_mutated_positions()
        # stratify into hamming dists
        self.sequence_mutation_locations = self.boolean_mutant_array()

        # get neighbors for each sequence
        self.mutation_arrays = self.gen_mutation_arrays()

        subsets = {x : [] for x in range(seq_len+1)}
        self.hammings = self.hamming_array()

        for distance in range(seq_len+1):
            # store indexing array to isolate seqs with provided hamming dist
            subsets[distance] = np.equal(distance, self.hammings)
        subsets = {k : v for k,v in subsets.items() if v.any()}
        self.max_distance = max(subsets.keys())
        self.d_data = subsets


        if graph: 
            self.graph = graph
        elif calculate_graph: 
            self.graph = self.build_graph()

        if calculate_ruggedness: 
            self.num_minima,self.num_maxima = self.calculate_num_extrema()
            self.extrema_ruggedness = self.calc_extrema_ruggedness()
            (
                self.linear_slope,
                self.linear_RMSE,
                self.RS_ruggedness,
            ) = self.rs_ruggedness()
        print(self)

    def seed(self):
        return self.seed_seq
    
    def __str__(self):
        return f"""
        Protein Landscape class
            Number of Sequences : {len(self)}
            Max Distance        : {self.max_distance}
            Number of Distances : {len(self.d_data)}
            Seed Sequence       : {self.coloured_seed_string()}
                Modified positions are shown in green
            
        """
    
    def __repr__(self):
        # TODO Finish this
        return r"""
            Protein_Landscape(sequences={})
            """
    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.data[idx]
    
    def get_distance(self, 
                     dist: int, 
                     tokenize: bool = True):
        '''
        Returns all arrays at a fixed distance from the seed string

        Parameters:
        -----------
        dist : int
            The distance that you want extracted

        tokenize : Bool, False
            Whether or not the returned data will be in tokenized form
            or not.
        '''
        assert dist in self.d_data.keys(), "Distance not found in data."

        if tokenize:
            return self.tokenized[self.d_data[dist]]
        else:
            return self.data[self.d_data[dist]]

    def get_mutated_positions(self, 
                              positions: ArrayLike, 
                              tokenize: bool = False):
        '''
        Function that returns the portion of the data only where the
        provided positions have been modified.

        Parameters:
        -----------
        positions : np.array(ints)
            Numpy array of integer positions that will be used to index
            the data.

        tokenize : Bool, default=False
            Boolean that determines if the returned data will be
            tokenized or not.

        Returns:
        --------
        sequence_data : ArrayLike   
            Returns the data array with only the sequences that have 
            been modified at the provided positions. Cn provide as 
            either tokenized or raw sequence form.
        '''
        for pos in positions:
            assert pos in self.mutated_positions, (
                f"{pos} is not a valid position."
            )

        constants = np.setdiff1d(self.mutated_positions, positions)
        index_array = np.ones((len(self.seed_seq)), dtype=np.int8)
        index_array[positions] = 0
        # Check constant positions remain constant
        mutated_indexes = np.all(
            np.invert(self.sequence_mutation_locations[:,constants]), 
            axis=1
        ) 

        if tokenize:
            return self.tokenized[mutated_indexes]
        else:
            return self.data[mutated_indexes]
        
    @staticmethod
    def hamming(seq_1: str, seq_2: str):
        '''Calculates the Hamming distance between 2 strings'''
        return sum(aa_1 != aa_2 for aa_1, aa_2 in zip(seq_1, seq_2))
    
    def hamming_array(self, 
                      seq: Optional[ArrayLike] = None, 
                      data: Optional[ArrayLike] = None):
        '''
        Calculate the hamming distance of every array using vectorized
        operations.

        Function operates by building an array of the 
        (Nxlen(seed sequence)) with copies of the tokenized seed 
        sequence.

        This array is then compared elementwise with the tokenized data,
        setting all places where they don't match to False. This array 
        is then inverted, and summed, producing an integer representing 
        the difference for each string.

        Parameters:
        -----------
        seq : np.array[int], default=None
            Sequence which will be compared to the entire dataset.

        data : np.array, default=None
            Data for getting Hamming array for.
        '''
        if seq is None:
            tokenized_seq = np.array(self.tokenize(self.seed_seq))
        else:
            tokenized_seq = seq

        if data is None:
            data = self.tokenized[:, :-1]

        hammings = np.sum(np.invert(data == tokenized_seq), axis=1)

        return hammings
        
    @staticmethod
    def csv_dloader(csv_file: str,
                    x_data: str = "Sequence",
                    y_data: str = "Fitness",
                    index_col: Optional[int] = None):
        '''
        Simple helper function to load NK landscape data from CSV files
        into numpy arrays. Supply outputs to sklearn_split to tokenise 
        and split into train/test split.

        Parameters
        ----------
        csv_file : Path
            Path to CSV file that will be loaded.

        x_data : str, default="Sequence"
            String key used to extract relevant x_data column from 
            pandas dataframe of imported csv file.

        y_data : str, default="Fitness"
            String key used to extract relevant y_data column from
            pandas dataframe of imported csv file.

        index_col : int, default=None
            Interger value, if provided, will determine the column to
            use as the index column.

        returns np.array (Nx2), where N is the number of rows in the csv
        file.
            
            Returns an Nx2 array with the first column being x_data
            (sequences), and the second being y_data (fitnesses).
        '''
        data = pd.read_csv(csv_file, index_col=index_col)
        protein_data = data[[x_data, y_data]].to_numpy()
        return protein_data

    def tokenize(self, seq):
        '''
        Simple method which tokenizes an individual sequence - sequences
        are later embedded as one-hot prior to attribute assignment.
        '''
        return [self.tokens[aa] for aa in seq]
    
    def boolean_mutant_array(self):
        return np.invert(
            self.tokenized[:, :-1] == self.tokenize(self.seed_seq)
        )

    def calc_mutated_positions(self):
        '''
        Determines all positions that were modified experimentally and
        returns the indices of these modifications.

        Because the NumPy code is tricky to read, here is a quick 
        breakdown:

            self.tokenized is called, and the fitness column is removed
            by [:,:-1]. Each column is then tested against the first.
        '''
        # calculate the indices all of arrays which are modified.
        mutated_bools = np.invert(
            np.all(
                self.tokenized[:, :-1] == self.tokenize(self.seed_seq), axis=0
            )
        )
        # shift to the right so that zero can be counted as an idx
        mutated_idxs  = mutated_bools * np.arange(1,len(self.seed()) + 1)
        # shift back and return
        return mutated_idxs[mutated_idxs != 0] - 1
    
    def coloured_seed_string(self):
        '''
        Printing function that prints the original seed string and
        colours the positions that have been modified.
        '''
        strs = []
        idxs = self.mutated_positions
        for i,char in enumerate(self.seed_seq):
            if i in idxs:
                strs.append(f"{Fore.GREEN}{char}{Style.RESET_ALL}")
            else:
                strs.append("{0}".format(char))
        return "".join(strs)

    def tokenize_data(self):
        '''
        Takes an iterable of sequences provided as one amino acid
        strings and returns an array of their tokenized form.

        Note : The tokenize function is not called and the tokens value
        is regenerated as it removes a lot of function calls and speeds
        up the operation significantly.
        '''
        tokens = self.tokens
        return np.array(
            [[tokens[aa] for aa in seq] for seq in self.sequences]
        )
    
    def return_ohe(self):
        '''Convert sequence data into one-hot encodings.'''
        return [aa_to_ohe(s, self.amino_acids) for s in self.sequences]

    def get_data(self, tokenized: bool=False):
        '''
        Returns a copy of the data stored in the class.

        Parameters
        ----------
        tokenized : Bool, default=False
            Boolean value that determines if the raw or tokenized
            data will be returned.
        '''
        if tokenized:
            return copy.copy(self.data)
        else:
            return copy.copy(self.tokenized)

    def gen_mutation_arrays(self):
        leng = len(self.seed())
        xs = np.arange(leng*len(self.amino_acids))
        ys = np.array([[y for x in range(len(self.amino_acids))] 
                       for y in range(leng)]).flatten()
        modifiers = np.array([np.arange(len(self.amino_acids)) 
                              for x in range(leng)]).flatten()
        return (xs, ys, modifiers)
    
    def sklearn_data(self, 
                     data: Optional[ArrayLike] = None,
                     distance: Optional[Union[int, list]] = None,
                     positions: Optional[list] = None,
                     split: float = 0.8,
                     shuffle: bool = True,
                     random_state: Optional[int] = None,
                     convert_to_ohe: bool = False,
                     flatten_ohe: bool = False,
                     ):
        '''
        Parameters:
        -----------
        data : np.array(NxM+1), default=None

            Optional data array that will be split. Added to the 
            function to enable it to interface with lengthen sequences.

            Provided array is expected to be (NxM+1) where N is the
            number of data points, M is the sequence length, and the +1
            captures the extra column for the fitnesses.

        distance : int or [int], default=None
            The specific distance (or distances) from the seed sequence
            that the data should be sampled from.

        positions : [int], default=None
            The mutant positions that the data will be sampled from.

        split : float, default=0.8, range [0-1]
            The split point for the training - validation data.

        shuffle : Bool, default=True
            Determines if the data will be shuffled prior to returning.

        convert_to_ohe : Bool, default=False
            Determines if the data will be converted to one-hot encoding
            before being returned. Else, data will remain tokenized.

        flatten_ohe : bool, default=False
            If ohe, either flatten (n_seqs, seq_len * n_tokens) or keep in
            3D (n_seqs, seq_len, n_tokens)

        Returns:
        --------
        x_train, y_train, x_test, y_test
            All Nx1 arrays with train as the first 80% of the shuffled
            data and test as the latter 20% of the shuffled data.
        '''
        assert (0 <= split <= 1), "Split must be between 0 and 1"

        # get data
        if data is not None:
            data = data
        elif distance is not None:
            if type(distance) == int:
                data = copy.copy(self.get_distance(distance,
                                                   tokenize=True))
            elif type(distance) == list:
                data = collapse_concat(
                    [copy.copy(self.get_distance(d, tokenize=True))
                     for d in distance]
                )
            else:
                raise ValueError(
                    "Provided distance must be list of ints or int"
                )
        elif positions is not None:
            data = copy.copy(
                self.get_mutated_positions(positions, tokenize=True)
            )
        else:
            data = copy.copy(self.tokenized)

        # shuffle data prior to splitting
        if shuffle:
            # assign random state if provided
            if random_state is not None:
                np.random.seed(random_state)
            np.random.shuffle(data)

        split_point = int(len(data)*split)

        train = data[:split_point]
        test  = data[split_point:]

        # Y data selects only the last column of Data
        # X selects the rest

        x_train = train[:,:-1]
        y_train = train[:,-1]
        x_test  = test[:,:-1]
        y_test  = test[:,-1]

        # ensure train data is integer type (after tokenization)
        x_train = x_train.astype(int)
        x_test = x_test.astype(int)

        # convert tokenized data if requested - makes flat ohe
        if convert_to_ohe:
            x_train = token_data_to_ohe(x_train, 
                                        self.amino_acids, 
                                        flatten=flatten_ohe)
            x_test = token_data_to_ohe(x_test, 
                                       self.amino_acids, 
                                       flatten=flatten_ohe)

        return x_train.astype("int"), y_train.astype("float"), \
               x_test.astype("int"), y_test.astype("float")

    def gen_mutation_arrays(self):
        leng = len(self.seed())
        xs = np.arange(leng*len(self.amino_acids))
        ys = np.array(
            [[y for x in range(len(self.amino_acids))] for y in range(leng)]
        ).flatten()
        modifiers = np.array(
            [np.arange(len(self.amino_acids)) for x in range(leng)]
        ).flatten()
        return (xs, ys, modifiers)

    def lengthen_sequences(self, 
                           seq_len: int, 
                           AAs: Optional[str] = None, 
                           mut_indices: bool = False):
        '''
        Vectorized function that takes a seq len and randomly inserts
        the tokenized integers into this new sequence framework with a
        fixed length.

        Parameters
        ----------
        seq_len : int
            Interger that determines how long the new sequences will be.

        AAs : str, default=None
            The string of all possible amino acids. Defaults to the 
            landscapes amino acid list.

        mut_indices : Bool
            Whether to specify the indices at which the old sequences will
            be embedded.
        '''
        if not AAs:
            AAs = self.amino_acids

        sequences = self.tokenized[:,:-1]
        new = np.random.choice(len(AAs),seq_len)
        lengthened_data = np.zeros(
            (sequences.shape[0],seq_len),
            dtype=np.int8
        )
        for i,val in enumerate(new):
            lengthened_data[:,i] = val

        if mut_indices.__class__.__name__ == "ndarray":
            assert len(mut_indices) == sequences.shape[1], (
                "Index array must have same length as the original sequence"
            )
            idxs = mut_indices

        else:
            idxs = np.random.choice(seq_len, 
                                    sequences.shape[1],
                                    replace=False)

        for i, idx in enumerate(idxs):
            lengthened_data[:,idx] = sequences[:,i]

        fitnesses = self.tokenized[:,-1]

        return np.concatenate(
            (lengthened_data,
             fitnesses.reshape(-1,1)
        ), axis=1 )

    def return_lengthened_data(self,
                               seq_len: int, 
                               AAs: Optional[str] = None, 
                               mut_indices: bool = False,
                               split: float = 0.8,
                               shuffle: bool = True):
        '''
        Helper function that passes the result of lengthen sequences
        to sklearn_data. Argument signature is a combination of 
        self.lengthen_sequences and self.sklearn_data.
        '''
        return self.sklearn_data(
            data=(self.lengthen_sequences(seq_len,AAs,mut_indices)),
            split=split,
            shuffle=shuffle
        )

    def save(self, name=None, ext=".txt"):
        """
        Save function that stores the entire landscape so that it can be
        reused without having to recompute distances and tokenizations.

        Parameters
        ----------
        name : str, default=None
            Name that the class will be saved under. If none is provided
            it defaults to the same name as the csv file provided.

        ext : str, default=".txt"
            Extension that the file will be saved with.
        """
        if self.csv_path:
            directory, file = self.csv_path.rsplit("/",1)
            directory += "/"
            if not name:
                name = file.rsplit(".",1)[0]
            file = open(directory+name+ext,"wb")
            file.write(pickle.dumps(self.__dict__))
            file.close()

    def load(self, name):
        '''
        Functions that instantiates the landscape from a saved file if
        one is provided.

        Parameters
        ----------
        name: str
            Provides the name of the file. MUST contain the extension.
        '''
        file = open(name,"rb")
        data_pkl = file.read()
        file.close()

        self.__dict__ = pickle.loads(data_pkl)
        return True
    
    ## Ruggedness Section ##
    def is_extrema(self, idx: int, graph: Optional[dict] = None):
        '''
        Takes the ID of a sequence and determines whether or not it is
        a maxima given its neighbours.

        Parameters:
        -----------
        idx : int
            Integer index of the sequence that will be checked as a maxima.
        
        graph : dict, default=None
            Dictionary that contains the graph of the protein landscape.
        '''
        if graph is None:
            graph = self.graph

        neighbours = graph[tuple(self.tokenized[idx,:-1])]
        max_comparisons = np.greater(self.fitnesses[idx],
                                     self.fitnesses[neighbours])
        min_comparisons = np.less(self.fitnesses[idx],
                                  self.fitnesses[neighbours])
        # check if maxima
        if np.all(max_comparisons):
            return 1
        # check if minima
        elif np.all(min_comparisons):
            return -1
        else:
            return 0

    def generate_mutations(self, seq: ArrayLike):
        """
        Takes a sequence and generates all possible mutants 1 Hamming
        distance away using array substitution.

        Parameters:
        -----------
        seq : np.array[int]
            Tokenized sequence array
        """
        seed = self.seed()
        hold_array = np.zeros(((len(seed)*len(self.amino_acids)),len(seed)))
        for i,char in enumerate(seq):
            hold_array[:,i] = char
        xs, ys, mutations = self.mutation_arrays
        hold_array[(xs,ys)] = mutations
        copies = np.invert(np.all(hold_array == seq,axis=1))
        return hold_array[copies]

    def calc_neighbours(self, 
                        seq: ArrayLike, 
                        token_dict: Optional[dict] = None):
        '''
        Takes a sequence and checks all possible neighbours against the
        ones that are actually present within the dataset.

        Parameters:
        -----------
        seq : np.array[int]
            Tokenized sequence array
        '''
        if token_dict is None:
            token_dict = self.token_dict
        possible_neighbours = self.generate_mutations(seq)
        actual_neighbours = [
            token_dict[tuple(key)] 
            for key in possible_neighbours if tuple(key) in token_dict
        ]
        return seq, actual_neighbours

    def calculate_num_extrema(self, idxs: Optional[ArrayLike] = None):
        '''
        Calcaultes the number of maxima across a given dataset or array
        of indices.
        '''
        if idxs is None:
            idxs = range(len(self))
            graph = self.graph
        else:
            graph = self.build_graph(idxs=idxs)
            idxs = np.where(idxs)[0]

        print("Calculating the number of extrema")
        mapfunc = partial(self.is_extrema, graph=graph)
        results = np.array(list(map(mapfunc,tqdm.tqdm(idxs))))
        minima = -1*np.sum(results[results<0])
        maxima = np.sum(results[results>0])
        return minima, maxima

    def calc_extrema_ruggedness(self):
        '''
        Simple function that returns a normalized ruggedness value.
        '''
        ruggedness = (self.num_minima+self.num_maxima)/len(self)
        return ruggedness

    ## Graph Section ##
    def build_graph(self, idxs: Optional[ArrayLike] = None):
        '''
        Builds the graph of the protein landscape.

        Parameters:
        -----------
        idxs : np.array[int]
            An array of integers that are used to index the complete
            dataset and provide a subset to construct a subgraph of the
            full dataset.
        '''
        if idxs is None:
            print("Building Protein Graph for entire dataset")
            dataset = self.tokenized[:,:-1]
            token_dict = self.token_dict
            pool = mp.Pool(mp.cpu_count())

        else:
            print(f"Building Protein Graph For subset of length {sum(idxs)}")
            dataset = self.tokenized[:,:-1][idxs]
            int_idxs = np.where(idxs)[0]
            token_dict = {
                key: value 
                for key,value in self.token_dict.items() if value in int_idxs}
            if len(int_idxs) < 100000:
                pool = mp.Pool(4)
            else:
                pool = mp.Pool(mp.cpu_count())

        mapfunc = partial(self.calc_neighbours, token_dict=token_dict)
        results = pool.map(mapfunc, tqdm.tqdm(dataset))
        neighbours = {tuple(key) : value for key, value in results}
        return neighbours
    
    def fit_adjacency(self): 
        if self.graph == None:
            print('Computing graph...') 
            self.build_graph()
            print('Graph built. Computing adjancency...')
        
        self.adjacency = self.dict_graph_to_adjacency(self.graph)
        return self 
    
    def dict_graph_to_adjacency(self, dict_graph: dict):
        '''
        Get the adjacency matrix from the graph dictionary.
        '''
        dim = len(dict_graph)
        matrix = sp.lil_matrix((dim, dim))
        values = list(dict_graph.values())
        
        for ind, i in enumerate(values): 
            matrix[ind, i] = 1
        return matrix

    def extrema_ruggedness_subset(self,idxs):
        '''
        Calculate the extrema ruggedness based on a subset of the full
        protein graph.
        '''
        minima, maxima = self.calculate_num_extrema(idxs=idxs)
        return (minima+maxima)/sum(idxs)

    def indexing(self,distances=None,percentage=None,positions=None):
        '''
        Handles more complex indexing operations, for example combining
        multiple distance indexes or asking for a random set of indices
        of a given length relative to the overall dataset.
        '''
        if distances is not None:
            # Uses reduce from functools package and the bitwise or operation
            # to recursively combine the indexing arrays, returning a final 
            # array where all Trues are collated into one array
            return reduce(np.logical_or, [self.d_data[d] for d in distances])

        if percentage is not None:
            assert 0 <= percentage <= 1, "Percentage must be between 0 and 1"
            idxs = np.zeros((len(self)))
            for idx in np.random.choice(np.arange(len(self)),
                                        size=int(len(self)*percentage),
                                        replace=False):
                idxs[idx] = 1
            return idxs.astype(np.bool)

        if positions is not None:
            # This code uses bitwise operations to maximize speed. It first
            # uses an or gate to collect every one where the desired position
            # was modified. It then goes through each position that shouldn't
            # be changed, and uses three logic gates to switch ones where 
            # they're both on to off, returning the indexes of strings where
            # ONLY the desired positions are changed
            not_positions = [
                x for x in range(len(self.seed_seq)) if x not in positions
            ]
            working = reduce(
                np.logical_or,
                [self.sequence_mutation_locations[:,pos] for pos in positions]
            )
            for pos in not_positions:
                temp = np.logical_xor(working,
                                      self.sequence_mutation_locations[:, pos])
                working = np.logical_and(
                    temp,
                    np.logical_not(self.sequence_mutation_locations[:, pos]))
            return working


    def rs_ruggedness(self, log_transform=False, distance=None, split=1.0, n_jobs=-1):
        '''
        Returns the rs based ruggedness estimate for the landscape.

        Parameters
        ----------
        log_transform : bool, default=False
            Boolean value that determines if the base 10 log transform 
            will be applied. The application of this was suggested in 
            the work by Szengdo

        distance : int, default=None
            Determines the distance for data that will be sampled

        split : float, default=1.0, range [0-1]
            How much of the data is used to determine ruggedness
        '''
        if distance:
            x_train, y_train, _, _ = self.sklearn_data(
                split=split,
                distance=distance
            )
        else:
            x_train, y_train, _, _ = self.sklearn_data(split=split)
        if log_transform:
            y_train = np.log10(y_train)

        lin_model = LinearRegression(n_jobs=n_jobs).fit(
            x_train, 
            y_train
        )
        y_preds = lin_model.predict(x_train)
        coefs   = lin_model.coef_
        rmse_predictions = np.sqrt(mean_squared_error(y_train, y_preds))
        slope = (1/len(self.seed_seq)*sum([abs(i) for i in coefs]))
        return [slope, rmse_predictions, rmse_predictions/slope]