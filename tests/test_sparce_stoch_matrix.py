import numpy as np
from scipy.sparse import csr_matrix, coo_matrix

def test_SSM(get_csr_matrix):
    """Bacic operations with the 'spares_stoch_mat' class
    """
    from flowstab.SparseStochMat import sparse_stoch_mat as SSM
    # Inits
    # ###
    # inti from scipy.sparse.csr_matrix
    A_csr = get_csr_matrix.copy()
    # print(A_csr.toarray())
    ssm = SSM.from_full_csr_matrix(A_csr)
    # print(ssm.toarray())
    
    




def test_SPA():
    """Basic operations with the `SpasreStochMat.SPA` class"""
    from flowstab.SparseStochMat import SPA
    # TODO
