# FACT-AI

This repository contains the code used for reproducing the paper "DECAF: Generating Fair Synthetic Data Using Causally-Aware Generative Networks" by Boris van Breugel, Trent Kyono, Jeroen Berrevoets, and Mihaela van der Schaar (2021).

To create our code base we heavily relied on the code already provided by the authors in their paper which can be found at: https://github.com/vanderschaarlab/DECAF. In particular, the file DECAF.py in the DECAF/models folder is a direct copy of the file as found in said repository.

## Installation and Requirements

To be able to run the code, we have provided an anaconda environment that contains all the necessary packages. To install the environment run the following conda command in the main directory:

    conda env create -f environment.yml
    
To activate the environment run the following command:

    conda activate decaf-repro
    
To be able to install this environment you will first have to have Anaconda installed on your device and you will need JDK 1.8 to be able to install the py-causal package.

## Contents

* DECAF/models contains the DECAF synthetic generator class and a GAN and WGAN-GP class to be used as benchmarks
* DECAF/util contains files (adult_data.py, credit_data.py, crime_data.py and data.py) used for getting the data from each dataset that we used in the correct format, a file dag.py for generating a dag using TETRAD for each dataset, a file metrics.py for calculating the fairness metrics, and a file plots.py for creating the plots for Experiment 2.
* DECAF/checkpoints contains 10 models that were pretrained with different seeds for all experiments we performed to be able to quickly reproduce our results.
* DECAF/benchmarks.py is a file for training and testing our benchmark models on the adult dataset
* DECAF/test_decaf.py is a file for training and testing the DECAF model for all datasets we used
* DECAF/experiments.ipynb is a notebook for reproducing our results as they can be found in our report

## Reproducing our Results

To reproduce the results as they can be found in our paper, we have provided the notebook DECAF/experiments.py. In this notebook we generate the synthetic data using our pretrained DECAF and benchmark models and we train a downstream classifier on the data to test the data utility. Since training the downstream models still takes some time we have given the option to generate results for one run or for the ten runs as can be found in the paper. Since the standard deviation of the metrics are not very high, running one model should still give reasonable results if you do not have the time. 

To reproduce the results simply follow the instructions in the notebook. Generating results for 10 seeds will take about an hour.
