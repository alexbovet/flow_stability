# Flow stability for dynamic community detection
Dynamic community detection in temporal networks using the flow stability framework described in 

Alexandre Bovet, Jean-Charles Delvenne & Renaud Lambiotte,
[Flow stability for dynamic community detection](https://www.science.org/doi/10.1126/sciadv.abj3063),
Sci. Adv., 8 (19), eabj3063. DOI: 10.1126/sciadv.abj3063

# Requirements:
- Python3 (>=3.6)
- pandas
- scipy
- numpy
- Cython (optional but highly recommended)
- [sparse_dot_mkl](https://github.com/flatironinstitute/sparse_dot) (optional, allows to perform multithreaded sparse matrix multiplication)

# Installation

`git clone git@github.com:alexbovet/flow_stability.git`

`cd flow_stability`

`python setup.py build_ext --inplace` to compile the cython functions.

# Usage

The main classes are:
- `ContTempNetwork` in the module `TemporalNetwork` which is used to store and save temporal networks and to compute inter-event transition matrices.
- `FlowIntegralClustering` in the module `FlowStability` which is used to computed the flow stability (integral of covariance) and to find the best forward and backward partition using the Louvain algorithm.

Additional interesting classes and functions are:
- `Clustering` and `SparseClustering` in `FlowStability` can be used to directly cluster covariances or intergrals of covariances.
- `static_clustering` in `FlowStability` is an helper function to cluster static networks using Markov Stability.
- `run_multi_louvain` in `FlowStability` helper function to run the Louvain multiple times on the same covariance in order to check the robustness of the partition.
- `avg_norm_var_information` in `FlowStability` computes the average Normalized Variation of Information of list of cluster lists obtained with `run_multi_louvain`.
- `compute_parallel_clustering` in `parallel_clustering`, same than `run_multi_louvain` but in parallel.
- the `parallel_expm` module contains functions to compute the matrix exponential of very large matrices using different strategies.

A jupyter notebook reproducing the example from Fig. 2 of the paper is available in [`asymmetric_example.ipynb`](https://github.com/alexbovet/flow_stability/blob/main/asymmetric_example.ipynb).




[![DOI](https://zenodo.org/badge/330739659.svg)](https://zenodo.org/badge/latestdoi/330739659)


