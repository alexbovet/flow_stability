"""
This module provides drop-in replacement functions in case cython is not installed.
"""
import numpy as np

def sparse_stoch_from_full_csr(nz_rowcols, Tf_data, Tf_indices, Tf_indptr, diag_val):
    """Pure python implementation of the sparce_stoch_mat
    """
    T_s_nnz = Tf_data.shape[0] - Tf_indptr.shape[0] + 1 + nz_rowcols.shape[0]

    T_s_data = np.zeros(T_s_nnz, dtype=np.float64)
    T_s_indices = -1*np.ones(T_s_nnz, dtype=np.int32)
    T_s_indptr = -1*np.ones(nz_rowcols.size+1, dtype=np.int32)

    #map indices from big to small T
    BtoS = {v:k for k,v in enumerate(nz_rowcols)}

    its = 0
    T_s_indptr[0] = 0
    for tsrow, tbrow in enumerate(nz_rowcols):
        nzr = 0
        for k in range(Tf_indptr[tbrow],Tf_indptr[tbrow+1]):
            T_s_data[its] = Tf_data[k]
            T_s_indices[its] = BtoS[Tf_indices[k]]
            its += 1
            nzr += 1
        T_s_indptr[tsrow+1] = T_s_indptr[tsrow] + nzr
    return (Tf_indptr.shape[0] - 1,
            T_s_data,
            T_s_indices,
            T_s_indptr,
            nz_rowcols,
            diag_val)
