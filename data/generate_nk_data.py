import sys

sys.path.append('/home/mahakaran/NK-paper-12-5-24-version/nk-ml-paper2-2024/pscapes')

from pscapes.landscape_models import make_NK
from pscapes.utils import dict_to_np_array

import pandas as pd
import numpy as np


AA_ALPHABET = 'ACDEFG' #amino acid alphabet for NK landscapes
SEQ_LENGTH  = 6	#sequence length for NK models 
K_VALS   	= list(range(SEQ_LENGTH)) # K values 
REPLICATES  = 8



def main():

	for k_index, k in enumerate(K_VALS):
		for r_index in range(REPLICATES):
			landscape 	 = make_NK(N=SEQ_LENGTH, K=k, AAs=AA_ALPHABET, distribution=np.random.uniform)
			landscape_np = dict_to_np_array(landscape)
			landscape_df = pd.DataFrame(landscape_np, columns=['sequence', 'fitness'])
			landscape_df.to_csv('./nk_landscapes/k{0}_r{1}.csv'.format(k_index, r_index))
	print('All data generated.')





if __name__ == "__main__":
    answer = input("Rerunning this script as a standalone will generate all data again in the default directory, is this what is desired? y/[n]")
    if answer.lower() == "y" or answer.lower() == "yes":
        main()





