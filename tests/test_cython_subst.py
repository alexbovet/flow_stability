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


def test_inplace_csr_row_normalize(cs_matrix_creator):
    """
    """
    from flowstab._cython_subst import inplace_csr_row_normalize as icrn_subst
    from flowstab.SparseStochMat import _css
    A_csr = cs_matrix_creator(nbr=1, size=100000, nbr_non_zeros=1000)[0]
    B_csr = A_csr.copy()
    n_row = 333
    row_sum = 1
    A_data_old = A_csr.data.copy()
    _css.inplace_csr_row_normalize(A_csr.data, A_csr.indptr.astype(np.int64), n_row, row_sum)
    icrn_subst(B_csr.data, B_csr.indptr, n_row, row_sum)
    # make sure it changed
    assert not np.array_equal(A_csr.data, A_data_old)
    # make sure non-cython is equivalent
    np.testing.assert_equal(A_csr.data, B_csr.data)
    

def test_stoch_mat_add(SSM_matrix_creator, compare_SSM_args):
    """
    """
    from flowstab._cython_subst import stoch_mat_add as sma_subst
    from flowstab.SparseStochMat import _css
    size=100000
    full_size = 10*size
    A, B = SSM_matrix_creator(nbr=2, size=100000, nbr_non_zeros=1000)

    ssm_args = _css.stoch_mat_add(
        size=A.size,
        Adata=A.T_small.data,
        Aindices=A.T_small.indices,
        Aindptr=A.T_small.indptr,
        Anz_rowcols=A.nz_rowcols,
        Adiag_val=A.diag_val,
        Bdata=B.T_small.data,
        Bindices=B.T_small.indices,
        Bindptr=B.T_small.indptr,
        Bnz_rowcols=B.nz_rowcols,
        Bdiag_val=B.diag_val,
    )
    ssm_args_subst = sma_subst(
        size=A.size,
        Adata=A.T_small.data,
        Aindices=A.T_small.indices,
        Aindptr=A.T_small.indptr,
        Anz_rowcols=A.nz_rowcols,
        Adiag_val=A.diag_val,
        Bdata=B.T_small.data,
        Bindices=B.T_small.indices,
        Bindptr=B.T_small.indptr,
        Bnz_rowcols=B.nz_rowcols,
        Bdiag_val=B.diag_val,
    )
    compare_SSM_args(ssm_args, ssm_args_subst)
    

def test_stoch_mat_sub(SSM_matrix_creator, compare_SSM_args):
    """
    """
    from flowstab._cython_subst import stoch_mat_sub as sma_subst
    from flowstab.SparseStochMat import _css
    size=100000
    full_size = 10*size
    A, B = SSM_matrix_creator(nbr=2, size=100000, nbr_non_zeros=1000)

    ssm_args = _css.stoch_mat_sub(
        size=A.size,
        Adata=A.T_small.data,
        Aindices=A.T_small.indices,
        Aindptr=A.T_small.indptr,
        Anz_rowcols=A.nz_rowcols,
        Adiag_val=A.diag_val,
        Bdata=B.T_small.data,
        Bindices=B.T_small.indices,
        Bindptr=B.T_small.indptr,
        Bnz_rowcols=B.nz_rowcols,
        Bdiag_val=B.diag_val,
    )
    ssm_args_subst = sma_subst(
        size=A.size,
        Adata=A.T_small.data,
        Aindices=A.T_small.indices,
        Aindptr=A.T_small.indptr,
        Anz_rowcols=A.nz_rowcols,
        Adiag_val=A.diag_val,
        Bdata=B.T_small.data,
        Bindices=B.T_small.indices,
        Bindptr=B.T_small.indptr,
        Bnz_rowcols=B.nz_rowcols,
        Bdiag_val=B.diag_val,
    )
    compare_SSM_args(ssm_args, ssm_args_subst)

def test_rebuild_nnz_rowcol(SSM_matrix_creator, compare_SSM_args):
    """
    """
    from flowstab._cython_subst import rebuild_nnz_rowcol as rnr_subst
    from flowstab.SparseStochMat import _css
    size=100000
    full_size = 10*size
    A = SSM_matrix_creator(nbr=2, size=100000, nbr_non_zeros=1000)[0]
    (data, indices, indptr, size) = _css.rebuild_nnz_rowcol(
        T_data=A.T_small.data,
        T_indices=A.T_small.indices.astype(np.int64),
        T_indptr=A.T_small.indptr.astype(np.int64),
        nonzero_indices=A.nz_rowcols.astype(np.int64),
        size=A.size,
        diag_val=A.diag_val
    )
    ssm_args = (size, np.asarray(data), np.asarray(indices), indptr, A.size)
    (data_, indices_, indptr_, size_) = rnr_subst(
        T_data=A.T_small.data,
        T_indices=A.T_small.indices.astype(np.int64),
        T_indptr=A.T_small.indptr.astype(np.int64),
        nonzero_indices=A.nz_rowcols.astype(np.int64),
        size=A.size,
        diag_val=A.diag_val
    )
    ssm_args_subst = (size_, data_, indices_, indptr_, A.size)
    compare_SSM_args(ssm_args, ssm_args_subst)

def test_get_submat_sum(cs_matrix_creator):
    """
    """
    from flowstab._cython_subst import get_submat_sum as gss_subst
    from flowstab.SparseStochMat import _css
    A_csr = cs_matrix_creator(nbr=1, size=100000, nbr_non_zeros=1000)[0]
    row_idx=np.arange(20, 200).astype(np.int32)
    col_idx=np.arange(5,500).astype(np.int32)
    subm_sum = _css.get_submat_sum(
        Adata=A_csr.data,
        Aindices=A_csr.indices,
        Aindptr=A_csr.indptr,
        row_idx=row_idx,
        col_idx=col_idx
    )
    subm_sum_subst = gss_subst(
        Adata=A_csr.data,
        Aindices=A_csr.indices,
        Aindptr=A_csr.indptr,
        row_idx=row_idx,
        col_idx=col_idx
    )
    assert subm_sum == subm_sum_subst
