# Improving the Classification of General Public and Institutional Twitter Users with Transformers.

> This repository holds the report and code for my project in Computational Analysis of Communication, 
> in the fall semester of 2022. The course was supervised by
> [Felix Dietrich](https://felix-dietrich.de/).

## Introduction

Several classifiers exist for differentiating general public and institutional Twitter users. 
To the best of our knowledge, none of them have used transformers and transfer learning to create predictions out of text data.
In our work, we attempted to beat the performance of an [existing study](https://doi.org/10.1080/19312458.2018.1430755) using solely Twitter profile descriptions as input, for classifying users into the two groups.
To achieve this, we used  [BERTweet](https://github.com/VinAIResearch/BERTweet), a pretrained transformer model to obtain useful features. 
These were then fed into a dense, one-layer classification head.
Results were mixed, with performance improvements in many cases, but worse performance in others. A more thorough hyperparameter search might further improve the results in the future.


# Table of contents

- [Overview](#overview)
- [Usage](#usage)
  - [Environment](#env)
  - [Flags](#flags)
    - [Hyperparameter Optimization](#hpo)
    - [Training and Testing](#train_test)
  


# Overview

The project directory is organised in the following way:

| Path               | Role                                                                      |
|--------------------|---------------------------------------------------------------------------|
| `data/`            | The data used for training and analysis.                                  |
| `environment/`     | A yml containing the environment.                                         |
| `logs/`            | The logs produced by training, validating and testing.                    |
| `report/`          | The report.                                                               |
| `analysis.py`      | Some data analysis.                                                       |
| `preprocessing.py` | The data preprocessing.                                                   |
| `run_training.sh`  | File for running the hyperparameter optimization or training and testing. |
| `training.py`      | The hyperparameter optimization, training and testing.                    |


# Usage

To run the hyperparameter optimization and/or training and testing yourself, follow the steps mentioned below.

## Environment
This project was created on Ubuntu 22.04.1.

To create the conda environment used for this project on your local machine, navigate to `/environment` in your terminal
and call 

`conda env create -f last_working_env.yml`

To activate it, call:

`conda activate user_classification`

## Flags

Different flags are required to run either the HPO or training and testing.

### Hyperparameter Optimization
The Flags that have to be set to conduct an HPO are: 

| Flag         | Description                                            |
|--------------|--------------------------------------------------------|
| `--train_ds` | The training dataset.                                  |
| `--val_ds`   | The validation dataset.                                |
| `--test_ds`  | The test dataset.                                      |
| `--mode`     | The mode ("train" or "test").                          |
| `--n_trials` | The number of trials conducted by the HyperbandPruner. |


To run the HPO you have two options. 


**Option 1:** 

Call the following in your terminal to run an HPO with 50 randomly sampled hyperparameter configurations on the 
preprocessed _boston_ dataset:

`python3 training.py --train_ds=df_boston_nd_c_train --val_ds=df_boston_nd_c_val --test_ds=df_boston_nd_c_test --mode=train --n_trials=50`


**Option 2:**

Put the abovementioned command into the `run_training.sh` file and call the following in your terminal:

`sh run_training.sh`


### Training and Testing
Before running the training, you have to specify the `lr`, `momentum`, `batch_size`, and `seq_length`, in the main function of `training.py`.
The optimal values for this can be identified by running the hyperparameter optimization first.

The Flags that have to be set to conduct a normal training and test run are: 

| Flag         | Description                                            |
|--------------|--------------------------------------------------------|
| `--train_ds` | The training dataset.                                  |
| `--val_ds`   | The validation dataset.                                |
| `--test_ds`  | The test dataset.                                      |
| `--mode`     | The mode ("train" or "test").                          |


To run you again have two options. 

**Option 1:** 

Call the following in your terminal to run training with early stopping after 10 epochs without an improved validation 
loss, and subsequent testing on the preprocessed _boston_ dataset:

`python3 training.py --train_ds=df_boston_nd_c_train --val_ds=df_boston_nd_c_val --test_ds=df_boston_nd_c_test --mode=test`

**Option 2:**

Put the abovementioned command into the `run_training.sh` file and call the following in your terminal:

`sh run_training.sh`
