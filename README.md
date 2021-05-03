# Flow stability for dynamic community detection
Dynamic community detection in temporal networks using the flow stability framework described in  https://arxiv.org/abs/2101.06131. 

# Requirements:
- Python3 (>=3.6)
- pandas
- scipy
- numpy
- Cython (optional but highly recommended)
- [sparse_dot_mkl](https://github.com/flatironinstitute/sparse_dot) (optional, allows to perform multithreaded sparse matrix multiplication)
- graph_tool (optional, only used to aggregate temporal networks in a static network)

# Usage

A jupyter notebook reproducing the example from Fig. 1 of the paper is available in [`non_stationary_example.ipynb`](https://github.com/alexbovet/flow_stability/blob/main/non_stationary_example.ipynb).

