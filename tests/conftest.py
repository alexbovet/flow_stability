import pytest

import numpy as np
from scipy.sparse import csr_matrix

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

@pytest.fixture(scope='session')
def csr_matrix_creator():
    """Creat an exemplary csr matrix that can be used for testing
    """
    size = 10000000
    nbr_non_zeros = 100000

    def _get_matrix(nbr:int=1,size:int=size, nbr_non_zeros:int=nbr_non_zeros):
        matrices = []
        for _ in range(nbr):
            row = np.sort(np.random.randint(0, size, size=nbr_non_zeros))
            col = np.sort(np.random.randint(0, size, size=nbr_non_zeros))
            data = np.random.random(size=nbr_non_zeros)
            a_csr = csr_matrix((data, (row, col)), shape=(size, size))
            a_csr.indptr = a_csr.indptr.astype(np.int32, copy=False)
            a_csr.indices = a_csr.indices.astype(np.int32, copy=False)
            matrices.append(a_csr)
        return tuple(matrices)
    return _get_matrix

@pytest.fixture(scope='session')
def SSM_matrix_creator():
    """Creat an exemplary csr matrix that can be used for testing
    """
    from flowstab.SparseStochMat import sparse_stoch_mat
    size = 1000000
    nbr_non_zeros = 1000
    def _get_matrix(nbr:int=1,size:int=size, nbr_non_zeros:int=nbr_non_zeros):
        matrices = []
        for _ in range(nbr):
            row = np.sort(np.random.randint(0, size, size=nbr_non_zeros))
            col = np.sort(np.random.randint(0, size, size=nbr_non_zeros))
            data = np.random.randint(0,1, size=nbr_non_zeros)
            _a_csr = csr_matrix((data, (row, col)), shape=(size, size))
            _a_csr.indptr = _a_csr.indptr.astype(np.int32, copy=False)
            _a_csr.indices = _a_csr.indices.astype(np.int64, copy=False)
            matrices.append(sparse_stoch_mat.from_full_csr_matrix(_a_csr))
        return tuple(matrices)
    return _get_matrix
