# Investigating the determinants of performance in machine learning for protein fitness prediction

This repository contains the codebase used for evaluating the performance of key machine learning algorithms on synthetic NK fitness landscapes, across a number of key metrics: 

1. Interpolation;
2. Extrapolation;
3. Ablation; and
4. Positional extrapolation.

## Python environment 
All scripts were run using Python 3.10.12 and packages listed in requirements.txt. 

## Installation
After downloading this repository, please create a Python 3.10.12 virtual environment: 

`cd /path/to/nk-ml` \
`python -m venv venv` 


And then install using `setup.py`: 

`pip install -r requirements.txt`\
`pip install ./setup.py`

## Demos/Scripts
The use of source code is found in the `/scripts` directory.

## File overview 

### `data`
Contains subdirectories with the datasets used in this project, namely GB1 and NK landscapes. 

### `figures`
Contains Jupyter Notebooks, SVG and PNG files corresponding to figures in the main text of the paper. 

### `hyperopt`
Contains results of hyperparamter optimisation 

### `scripts`
Scripts used in the generation of results. 

1. `generate_nk_data.py`: generates NK landscapes in `data/nk_landscapes`
2. `run_hparam_opt_GB1.py`: performs hyperparameter optimisation on GB1 dataset, depositing results in `hyperopt/ohe/gb1_hparams`
3. `run_hparam_opt_NK.py`: performs hyperparameter optimisation on NK datasets, depositing results in `hyperopt/ohe/nk_landscape_hparams`

### `src`
The package source code directory. Root directory of the project. 

#### `analysis`
Code for analysing model data.
1. `aa_reprsentations.py`: Code for extracting amino acid reprsentations from a given model.

#### `benchmarking`
Subdirectory implementing experiments i.e. training and testing across 4 metrics. 
1. `ablation.py`: functions for running ablation experiments 
2. `extrapolation.py`: functions for running extrapolation experiments 
3. `file_proc.py`: functions for collecting hyperparameter sets and datasets into structure amenable for experiment function definitionss. 
4. `interpolation.py`: functions for running interpolation experiments 
5. `positional_extrapolation.py`: functions for running positional extrapolation experiments

#### `modelling`
Subdirectory containing files for ML archicture definition, data processing and training. 
1. `architectures.py`: definitions of ML model classes, including fit() and score() methods
2. ``data_utils.py`: definitions of helper functions used in processing data for ML models and for scoring sklearn models 
3. `hparam_space`: default hyperparameter search space for ML models training on NK landscapes and GB1 dataset
4. `hyperopt.py`: definitions of objective functions for hyperparameter optimisation using optuna, including early stopping during hyperparameter optimisation 
5. `ml_utils`: definitions for functions for parsing hyperparameters, training of ML models and early stopping. 

#### `pscapes`
Subdirectory containing ProteinLandscape class and functions to initialise synthetic NK landscapes. 
1. `landscape_class.py`: definition of ProteinLandscape class, used extensively in the present work 
2. `NK_landscape.py`: definition of functions to initialise synthetic NK landscapes