import pytest

from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, csc_matrix

@pytest.fixture(scope='function')
def get_csr_matrix_small():
    """Creat an exemplary csr matrix that can be used for testing
    """
    row = np.array([0, 0, 1, 2, 3, 3])
    col = np.array([0, 2, 2, 1, 3, 4])
    data = np.array([1, 2, 3, 5, 1, 6])
    return csr_matrix((data, (row, col)), shape=(5, 5))

@pytest.fixture(scope='function')
def get_csr_matrix_large():
    """Creat an exemplary csr matrix that can be used for testing
    """
    size = 10000
    nbr_non_zeros = 1000
    row = np.sort(np.random.randint(0, size, size=nbr_non_zeros))
    col = np.sort(np.random.randint(0, size, size=nbr_non_zeros))
    data = np.random.randint(0,100, size=nbr_non_zeros)
    density = nbr_non_zeros / size
    return csr_matrix((data, (row, col)), shape=(size, size)), density

@pytest.fixture(scope='function')
def cs_matrix_creator():
    """Creat an exemplary csr matrix that can be used for testing
    """
    size = 1000000
    nbr_non_zeros = 10000

    def _get_matrix(nbr:int=1,size:int=size, nbr_non_zeros:int=nbr_non_zeros, mode='r'):
        matrices = []
        assert mode in ['r', 'c']
        if mode == 'r':
            matrix_gen = csr_matrix
        else:
            matrix_gen = csc_matrix

        for _ in range(nbr):
            row = np.sort(np.random.randint(0, size, size=nbr_non_zeros))
            col = np.sort(np.random.randint(0, size, size=nbr_non_zeros))
            data = np.random.random(size=nbr_non_zeros)
            a_csr = matrix_gen((data, (row, col)), shape=(size, size))
            a_csr.indptr = a_csr.indptr.astype(np.int32, copy=False)
            a_csr.indices = a_csr.indices.astype(np.int32, copy=False)
            matrices.append(a_csr)
        return tuple(matrices)
    return _get_matrix

@pytest.fixture(scope='function')
def SSM_matrix_creator():
    """Creat an exemplary csr matrix that can be used for testing
    """
    from flowstab.sparse_stoch_mat import SparseStochMat as SSM
    size = 1000000
    nbr_non_zeros = 1000
    def _get_matrix(nbr:int=1,size:int=size, nbr_non_zeros:int=nbr_non_zeros):
        matrices = []
        for _ in range(nbr):
            row = np.sort(np.random.randint(0, size, size=nbr_non_zeros))
            col = np.sort(np.random.randint(0, size, size=nbr_non_zeros))
            data = np.random.randint(0,1, size=nbr_non_zeros)
            _a_csr = csr_matrix((data, (row, col)), shape=(size, size))
            _a_csr.data = _a_csr.data.astype(np.float64, copy=False)
            _a_csr.indices = _a_csr.indices.astype(np.int32, copy=False)
            matrices.append(SSM.from_full_csr_matrix(_a_csr))
        return tuple(matrices)
    return _get_matrix

@pytest.fixture(scope='session')
def compare_alike():
    def compare_sparse_matrice(A, B):
        """Checks if two csr matrices describe the same matrix

        csr notation can deviate in that data and indices can be re-arranged
        within a indptr slice.
        """
        assert len(A.indptr) == len(B.indptr)
        for i in range(len(A.indptr) - 1):
            A_s, A_e = A.indptr[i:i+2]
            B_s, B_e = B.indptr[i:i+2]
            B_sorted = B.indices[B_s: B_e].argsort()
            A_sorted = A.indices[A_s: A_e].argsort()
            np.testing.assert_equal(A.indices[A_s:A_e][A_sorted], B.indices[B_s:B_e][B_sorted])
            np.testing.assert_equal(B.data[B_s:B_e][B_sorted], A.data[A_s:A_e][A_sorted])
    return compare_sparse_matrice

@pytest.fixture(scope='session')
def compare_SSM_args():
    def compare_SSM_args(ssm_args1, ssm_args2):
        """Checks if two csr matrices describe the same matrix

        csr notation can deviate in that data and indices can be re-arranged
        within a indptr slice.
        """
        # size
        assert ssm_args1[0] == ssm_args2[0]
        # data
        np.testing.assert_equal(ssm_args1[1], ssm_args2[1])
        # indices
        np.testing.assert_equal(ssm_args1[2], ssm_args2[2])
        # indptr
        np.testing.assert_equal(ssm_args1[3], ssm_args2[3])
        # diag val
        np.testing.assert_equal(ssm_args1[4], ssm_args2[4])
    return compare_SSM_args

@pytest.fixture(scope='session')
def probabilities_transition():
    """Create exemplary densities and transition probabilities
    """
    nbr_non_zeros = 1000
    p1 = np.ones(shape=(nbr_non_zeros), dtype=np.float64) / nbr_non_zeros
    T = np.zeros(shape=(nbr_non_zeros, nbr_non_zeros), dtype=np.float64)
    for i in range(nbr_non_zeros):
        if np.random.rand() >= 0.5:
            _t = np.random.dirichlet(np.ones(nbr_non_zeros),size=1)
        else:
            _t = np.zeros(shape=(nbr_non_zeros,), dtype=np.float64)
        T[:,i] = _t

    p2 = p1 @ T
    return p1, p2, T

@pytest.fixture(scope='session')
def propa_transproba_creator():
    """Creat an exemplary csr matrix that can be used for testing
    """
    size = 1000
    zero_col_density = 0.05
    def _get_p_tp(nbr:int=1, size:int=size,
                  zero_col_density:float=zero_col_density):
        """Generates a tuple of p1,p2,T triplets for a network of `size` nodes
        """
        ptps = []
        for _ in range(nbr):
            p1 = np.ones(shape=(size), dtype=np.float64) / size 
            T = np.zeros(shape=(size, size), dtype=np.float64)
            for i in range(size):
                if np.random.rand() < 1 - zero_col_density:
                    _t = np.random.dirichlet(np.ones(size),size=1)
                else:
                    _t = np.zeros(shape=(size,), dtype=np.float64)
                T[:,i] = _t
            p2 = p1 @ T
            ptps.append((p1,p2,T))
        return tuple(ptps)
    return _get_p_tp

def temporal_network_to_df(network: SimpleNamespace):
    """Convert a network from a namespace ot a data frame
    """
    as_df = pd.DataFrame({
        "source_nodes": network.source_nodes,
        "target_nodes": network.target_nodes,
        "starting_times": network.starting_times,
    })
    ending_times = getattr(network, 'ending_times', None)
    if ending_times is not None:
        as_df['ending_times'] = ending_times
    return as_df

@pytest.fixture(scope='session')
def get_temporal_network_df_minimal():
    simple = SimpleNamespace()
    # we assume 10 nodes, and each starting a connection in order
    simple.source_nodes = list(range(0, 10))
    # target nodes are also in order
    simple.target_nodes = list(range(1,10)) + [1]
    simple.starting_times = [0, 0.5, 1, 2, 3, 4, 4, 5, 5, 5]
    simple.ending_times =   [3, 1, 2, 7, 4, 5, 6, 6, 6, 7]
    return temporal_network_to_df(network=simple)

@pytest.fixture(scope='function')
def minimal_temp_network():
    # This fixture should return a ContTempNetwork instance as expected by FlowStability.
    # You may need to adjust this if ContTempNetwork needs specific arguments.
    from flowstab.temporal_network import ContTempNetwork
    import pandas as pd
    df = pd.DataFrame({"source_nodes": [0],
                       "target_nodes": [1],
                       "starting_times": [0],
                       "ending_times": [1]})
    return ContTempNetwork(events_table=df)
