# k-tvgraph
===============================================================

This repository contains the MATLAB code for 
k-component time-varying graph learning (kTVGL) 
applicable to heavy-tailed financial data clustering. 
The 'Main' folder contains the principal functions 
including the implementation of the 'kTVGL' method proposed 
in  the following paper: 

A. Javaheri, J. Ying, D. P. Palomar, and F. Marvasti, 
"Time-Varying Graph Learning for Data with Heavy-Tailed Distribution",
IEEE Transactions on Signal Processing ȧ, doi: 10.1109/TSP.2025.3588173, Jul 2025.

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

===============================================================

Run the following demos:
-----------------------------------
* demo_tv_graph_learning:
&nbsp;&nbsp;&nbsp; evaluate/visualize the time-varying graph learning performance versus the frame number
in a single experiment with random data


* tvgl_graph_learn_different_experiments:    
&nbsp;&nbsp;&nbsp; evaluate the average time-varying graph learning performance versus different parameters for 
several different experiments with random data. 
These parameters include:
&nbsp;&nbsp;&nbsp; ** sr          : the sampling rate 
&nbsp;&nbsp;&nbsp; ** std_n       : the noise level
&nbsp;&nbsp;&nbsp; ** nu          : degree of freedom parameter of the Student-t distribution
&nbsp;&nbsp;&nbsp; ** K           : number of clusters or components
&nbsp;&nbsp;&nbsp; ** d           : maximum degree of each node
&nbsp;&nbsp;&nbsp; ** rho         : parameter of the ADMM
&nbsp;&nbsp;&nbsp; ** sigma_e     : the parameter to specify the sparsity and temporal consistency of the graph
&nbsp;&nbsp;&nbsp; ** gamma       : regularization parameter for L1 norm penalty on the VAR model parameters


===============================================================

Please give a star and cite the following:


• A. Javaheri, J. Ying, D. P. Palomar, and F. Marvasti, 
"Time-Varying Graph Learning for Data with Heavy-Tailed Distribution",
IEEE Transactions on Signal Processing ȧ, doi: 10.1109/TSP.2025.3588173, Jul 2025.

• A. Javaheri and D. P. Palomar, 
"Learning Time-Varying Graphs for Heavy-Tailed Data Clustering", 
2024 European Signal Processing Conference (EUSIPCO 2024), Lyon, France, Aug 2024.
A. Javaheri, A. Amini, F. Marvasti and D. P. Palomar, "Joint Signal Recovery and Graph Learning from Incomplete Time-Series," ICASSP 2024 - 2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), Seoul, Korea, Republic of, 2024, pp. 13511-13515, doi: 10.1109/ICASSP48485.2024.10448021.

