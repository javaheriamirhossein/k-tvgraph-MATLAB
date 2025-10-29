# Description

This repository contains the MATLAB code for 
k-component time-varying graph learning (kTVGL) 
applicable to heavy-tailed financial data clustering. 
The 'Main' folder contains the principal functions 
including the implementation of the 'kTVGL' method.

The folder 'Utils' contains some utility functions 
(for random data generation and graph operations). 
The 'Data' folder also includes 'graph_types.mat' which 
contains several types of graph models to be used for 
generating synthetic data defined over time-varying graphs.

To compare with the benchmark algorithms one can edit the 
'tv_graph_learning_algorithms.m' file in the 'Main' folder by adding
some code lines for implentation of the benchmark methods.

We have already included the GSPBOX method for comparison. 
To run this method, you need to download the [GSPBOX](https://github.com/epfl-lts2/gspbox)
MATLAB toolbox and put the installation folder in the main directory.  


# Demos

## demo_tv_graph_learning:
Evaluate/visualize the time-varying graph learning performance versus the frame number
in a single experiment with random data

<img src="Demo results/Demo.png" width="100%" style="display: block; margin: auto;" />


## tvgl_graph_learn_different_experiments:    
Evaluate the average time-varying graph learning performance versus different parameters for 
several different experiments with random data. 
These parameters include:
  - sr          : the sampling rate 
  - std_n       : the noise level
  - nu          : degree of freedom parameter of the Student-t distribution
  - K           : number of clusters or components
  - d           : maximum degree of each node
  - rho         : parameter of the ADMM
  - sigma_e     : the parameter to specify the sparsity and temporal consistency of the graph
  - gamma       : regularization parameter for L1 norm penalty on the VAR model parameters


# References

Please give a star and cite the following:


• A. Javaheri, J. Ying, D. P. Palomar, and F. Marvasti, 
"Time-Varying Graph Learning for Data with Heavy-Tailed Distribution",
IEEE Transactions on Signal Processing ȧ, doi: 10.1109/TSP.2025.3588173, Jul 2025.

• A. Javaheri and D. P. Palomar, 
"Learning Time-Varying Graphs for Heavy-Tailed Data Clustering", 
2024 European Signal Processing Conference (EUSIPCO 2024), Lyon, France, Aug 2024.

