import pytest

import numpy as np
from scipy.sparse import (
    eye,
    )

def test_sparse_stoch_from_full_csr(cs_matrix_creator):
    """
    """
    from flowstab._cython_subst import sparse_stoch_from_full_csr as ssffc_subst
    from flowstab.SparseStochMat import _css
    A_csr = cs_matrix_creator(nbr=1, size=100000, nbr_non_zeros=1000)[0]
    diag_val = 1.0
    nz_rows, nz_cols = (A_csr - diag_val * eye(A_csr.shape[0], format="csr")).nonzero()
    nz_rowcols = np.union1d(nz_rows, nz_cols)
    ssm_args = _css.sparse_stoch_from_full_csr(nz_rowcols, A_csr.data,
                                                   A_csr.indices,
                                                   A_csr.indptr, diag_val)
    ssm_args_subst = ssffc_subst(nz_rowcols, A_csr.data, A_csr.indices,
                                     A_csr.indptr, diag_val)
    # size
    assert ssm_args[0] == ssm_args_subst[0]
    # data
    np.testing.assert_equal(ssm_args_subst[1], ssm_args[1])
    # indices
    np.testing.assert_equal(ssm_args_subst[2], ssm_args[2])
    # indptr
    np.testing.assert_equal(ssm_args_subst[3], ssm_args[3])
    # diag val
    np.testing.assert_equal(ssm_args_subst[4], ssm_args[4])
