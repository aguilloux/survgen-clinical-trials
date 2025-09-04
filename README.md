# HI-VAE

This repository contains the implementation of our Heterogeneous Variational Autoendoder model (surv_HI-VAE) supporting survival data, enabling the model to handle time-to-event analysis with incomplete and heterogeneous data types. It has been written in Python, using PyTorch.

## Database description

There are three different datasets considered in the experiments (XXXX). Each dataset has each own folder, containing:

* **data.csv**: the dataset
* **data_types.csv**: a csv containing the types of that particular dataset. Every line is a different attribute containing three paramenters:
  	* name: real, pos (positive), cat (categorical), ord (ordinal), count, surv (log-normal) or surv_weibul (Weibull)
   	* type: real, pos (positive), cat (categorical), ord (ordinal), count
	* dim: dimension of the variable in the original dataset
	* nclass: number of categories (for cat and ord)
* **Missingxx_y.csv**: a csv containing the positions of the different missing values in the data. Each "y" mask was generated randomly, containing a "xx" % of missing values. This file may be left blank if no missing values need to be specified.

You can add your own datasets as long as they follow this structure.

## Files description

* **src.py**: Contains the HI-VAE models (factorized encoder or input dropout encoder).
* **utils**: This folder contains different scripts to support load data, compute likelihood, compute error.
* **data_preprocessing __ .ipynb**: Data reprocessing notebooks.
* **tutorial __ .ipynb**: Tutorial notebooks.


# Code Pre-requisites

First, create an conda environment compatible with Synthcity,
```console
$ cd HI-VAE
$ conda create --name hivae python=3.12.9
$ conda activate hivae
$ pip install synthcity
$ pip install -r pip_requirements.txt
```

Then, run preprocessing and tutorial on your data.






