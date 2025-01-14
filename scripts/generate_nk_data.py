import numpy as np
import pandas as pd

from src.pscapes import make_NK

# define NK parameters

AA_ALPHABET = 'ACDEFG' #amino acid alphabet for NK landscapes
SEQ_LENGTH = 6	#sequence length for NK models 
K_VALS = list(range(SEQ_LENGTH)) # K values 
REPLICATES = 8

def main():
	# for each possible K value
	for k_index, k in enumerate(K_VALS):
		# for each replicate
		for r_index in range(REPLICATES):
			landscape = make_NK(N=SEQ_LENGTH, 
					   			K=k, 
								amino_acids=AA_ALPHABET, 
								distribution=np.random.uniform)
			landscape_array = np.array(
				[[key, float(value)] for key, value in landscape.items()],
				dtype='object'
			)
			landscape_df = pd.DataFrame(landscape_array, 
							   columns=['sequence', 'fitness'])
			landscape_df.to_csv(
				f'../data/nk_landscapes/k{k_index}_r{r_index}.csv'
			)
	print('All data generated.')

if __name__ == "__main__":
    answer = input("Overwrite existing data? (y/n) ")
    if answer.lower() == "y":
        main()





