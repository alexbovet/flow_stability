## flowstab - Flow stability for dynamic community detection
Python package for the dynamic community detection in temporal networks implementing the flow stability framework described in 

Alexandre Bovet, Jean-Charles Delvenne & Renaud Lambiotte,
[Flow stability for dynamic community detection](https://www.science.org/doi/10.1126/sciadv.abj3063),
Sci. Adv., 8 (19), eabj3063. DOI: 10.1126/sciadv.abj3063

<!--- quickstart --->
## Requirements:
- Python3 (>=3.9)
- pandas
- scipy
- numpy
- Cython (optional but highly recommended)
- [sparse_dot_mkl](https://github.com/flatironinstitute/sparse_dot) (optional, allows to perform multithreaded sparse matrix multiplication)

## Installation

You can pip install `flowstab` directly from this repository into your virtual
environment. Simply run:

    pip install git+https://github.com/alexbovet/flow_stability.git

## Usage

After installation you can access the relevant classes and methods by importing
`flowstab` into your python scripts or via command line (see [CLI scripts](#cli-scripts) for details).

If you want to use the `FlowIntegralClustering` class, for example, you would
want to add the following line in your script:

```python

from flowstab.flow_stability import FlowIntegralClustering

# forw_flow = FlowIntegralClustering(...
```

Refer to the [examples](./examples/) folder more detailed usage examples.

### CLI scripts

`flowstab` provides also several command line hooks that can be run
directly in the terminal after installation without the need of opening a
python shell:

**run\_clusterings**

This command requires
[sparse\_dot\_mkl](https://github.com/flatironinstitute/sparse_dot) which relies
on the closed-source `libmkl_rt.so` library. In Ubuntu you might need to fetch
this library with `apt-get install libmkl-rt`.

**run\_cov\_integrals**

**run\_laplacians\_transmats**

## Content

The main classes are:
- `ContTempNetwork` in the module `temporal_network` which is used to store and save temporal networks and to compute inter-event transition matrices.
- `FlowIntegralClustering` in the sub-module `network_clustering` which is used to computed the flow stability (integral of covariance) and to find the best forward and backward partition using the Louvain algorithm.

Additional interesting classes and functions are:
- `Clustering` and `SparseClustering` in the `network_clustering` sub-module can be used to directly cluster covariances or integrals of covariances.
- `static_clustering` in `flow_stability` is an helper function to cluster static networks using Markov Stability.
- `run_multi_louvain` in `network_clustering` helper function to run the Louvain multiple times on the same covariance in order to check the robustness of the partition.
- `avg_norm_var_information` in `network_clustering` computes the average Normalized Variation of Information of list of cluster lists obtained with `run_multi_louvain`.
- `compute_parallel_clustering` in `parallel_clustering`, same than `run_multi_louvain` but in parallel.
- the `parallel_expm` module contains functions to compute the matrix exponential of very large matrices using different strategies.

A jupyter notebook reproducing the example from Fig. 2 of the paper is available in [`asymmetric_example.ipynb`](https://github.com/alexbovet/flow_stability/blob/main/asymmetric_example.ipynb).


[![DOI](https://zenodo.org/badge/330739659.svg)](https://zenodo.org/badge/latestdoi/330739659)


