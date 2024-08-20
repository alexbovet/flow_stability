import pytest

import numpy as np
from scipy.sparse import csr_matrix

@pytest.fixture(scope='function')
def get_csr_matrix():
    """Creat an exemplary csr matrix that can be used for testing
    """
    row = np.array([0, 0, 1, 2, 3, 3])
    col = np.array([0, 2, 2, 1, 3, 4])
    data = np.array([1, 2, 3, 5, 1, 6])
    return csr_matrix((data, (row, col)), shape=(5, 5))
