import numpy as np
import tracemalloc

from copy import copy

from scipy.sparse import (
    eye,
)

def test_timing(capfd):
    """Bacic operations with the 'spares_stoch_mat' class
    """
    from time import sleep
    from flowstab.SparseStochMat import timing
    
    @timing
    def sleep_some(some=0.3, **params):
        return sleep(some)
    log_message = "END"
    sleep_some(verbose=True, log_message=log_message)
    out, err = capfd.readouterr()
    assert ~log_message.startswith("END")
    assert log_message in out


def test_SSM_small(get_csr_matrix_small):
    """Bacic operations with the 'spares_stoch_mat' class
    """
    from flowstab.SparseStochMat import sparse_stoch_mat as SSM
    # Inits
    # ###
    # inti from scipy.sparse.csr_matrix
    A_csr = get_csr_matrix_small
    ssm = SSM.from_full_csr_matrix(A_csr)
    np.testing.assert_equal(A_csr.toarray(), ssm.toarray(), strict=False)
    # crete a diagonal matrix
    _ = SSM.create_diag(size=100, diag_val=0.3)
    # convert it to a full csr
    full_A = A_csr.to_full_mat()
    # ...

    

def test_SSM_large(get_csr_matrix_large):
    """Make sure an SSM does not get expanded during creation
    """
    from flowstab.SparseStochMat import sparse_stoch_mat as SSM
    # Inits
    # ###
    # inti from scipy.sparse.csr_matrix
    A_csr, density = get_csr_matrix_large
    tracemalloc.start()
    for _ in range(100):
        _ = SSM.from_full_csr_matrix(A_csr)
    first_size, first_peak = tracemalloc.get_traced_memory()
    tracemalloc.reset_peak()

    for _ in range(100):
        _ = SSM.from_full_csr_matrix(A_csr).toarray()
    second_size, second_peak = tracemalloc.get_traced_memory()
    assert first_size < density * second_size
    assert first_peak < density * second_peak


def test_SSM_from_full_csr_cython_memory(get_csr_matrix_large):
    """Check the cython implementation
    """
    from flowstab.SparseStochMat import (
        _css
    )
    A_csr, _ = get_csr_matrix_large
    A_csr_data = A_csr.data.astype(np.float64)
    diag_val = 1.0
    nz_rows, nz_cols = (
        A_csr - diag_val * eye(A_csr.shape[0], format="csr")
    ).nonzero()
    nz_rowcols = np.union1d(nz_rows, nz_cols)
    for _ in range(100):
        _ = _css.sparse_stoch_from_full_csr(
            np.array(nz_rowcols, dtype=np.int32),
            A_csr_data,
            A_csr.indices,
            A_csr.indptr,
            diag_val
        )


def test_SSM_from_full_csr_nocython_memory(get_csr_matrix_large):
    """Check the python substitue
    """
    from flowstab._cython_sparse_stoch_subst import (
        sparse_stoch_from_full_csr as sparse_stoch_from_full_csr
    )
    A_csr, density = get_csr_matrix_large
    A_csr_data = A_csr.data.astype(np.float64)
    diag_val = 1.0
    nz_rows, nz_cols = (
        A_csr - diag_val * eye(A_csr.shape[0], format="csr")
    ).nonzero()
    nz_rowcols = np.union1d(nz_rows, nz_cols)
    for _ in range(100):
        _ = sparse_stoch_from_full_csr(
            np.array(nz_rowcols, dtype=np.int32),
            A_csr_data,
            A_csr.indices,
            A_csr.indptr,
            diag_val
        )

def test_SSM_from_full_csr_equivalence(get_csr_matrix_large):
    """Check if both cython and native python implementations match
    """
    from flowstab.SparseStochMat import (
        _css
    )
    from flowstab._cython_sparse_stoch_subst import (
        sparse_stoch_from_full_csr as sparse_stoch_from_full_csr
    )
    A_csr, _ = get_csr_matrix_large
    A_csr_data = A_csr.data.astype(np.float64)
    diag_val = 1.0
    nz_rows, nz_cols = (
        A_csr - diag_val * eye(A_csr.shape[0], format="csr")
    ).nonzero()
    nz_rowcols = np.union1d(nz_rows, nz_cols)
    (
        c_size, c_data, c_indices,
        c_indptr, c_nz_rowcols, c_diag_val
    ) = _css.sparse_stoch_from_full_csr(
            np.array(nz_rowcols, dtype=np.int32),
            A_csr_data,
            A_csr.indices.astype(np.int64),
            A_csr.indptr,
            diag_val
        )
    (
        nc_size, nc_data, nc_indices,
        nc_indptr, nc_nz_rowcols, nc_diag_val
    ) = sparse_stoch_from_full_csr(
            np.array(nz_rowcols, dtype=np.int32),
            A_csr_data,
            A_csr.indices,
            A_csr.indptr,
            diag_val
        )
    assert nc_size == c_size
    assert nc_diag_val == c_diag_val
    np.testing.assert_array_equal(nc_data, c_data)
    np.testing.assert_array_equal(nc_indices, c_indices)
    np.testing.assert_array_equal(nc_indptr, c_indptr)
    np.testing.assert_array_equal(nc_nz_rowcols, c_nz_rowcols)

def test_SSM_inplace_row_normalize_equivalence(get_SSM_matrix_large):
    """Make sure the cython and pure python implementations are equivalent
    """
    from flowstab.SparseStochMat import (
        _css
    )
    from flowstab._cython_sparse_stoch_subst import (
        inplace_csr_row_normalize
    )
    A_ssm1 = get_SSM_matrix_large
    A_ssm1_data = copy(A_ssm1.T_small.data)
    A_ssm2 = copy(A_ssm1)
    A_ssm2_data = copy(A_ssm2.T_small.data)
    # the cython implementation
    _css.inplace_csr_row_normalize(A_ssm1.T_small.data, A_ssm1.T_small.indptr, A_ssm1.T_small.shape[0], 1.0)
    # pure python
    inplace_csr_row_normalize(A_ssm2.T_small.data, A_ssm2.T_small.indptr, A_ssm2.T_small.shape[0], 1.0)
    # test change
    np.testing.assert_array_equal(A_ssm1_data, A_ssm1.T_small.data)
    np.testing.assert_array_equal(A_ssm2_data, A_ssm2.T_small.data)
    # test equivalence
    np.testing.assert_array_equal(A_ssm1.data, A_ssm2.T_small.data)

def test_csr_operations(get_SSM_matrix_large):
    """Check the csr_* funcions

    .. note::
      This function adds csr_matrix objects, so it should be compared to the built in addition

    """
    from flowstab.SparseStochMat import (
        csr_add, 
        csr_matmul,
        csr_csc_matmul,
        csr_csrT_matmul,
    )
    A_ssm1 = get_SSM_matrix_large
    # A_ssm2 = A_ssm1 + 2
    diff_csr = csr_add(A_ssm1, A_ssm1)
    np.testing.assert_equal(diff_csr.toarray(), np.full(shape=diff_csr.shape, fill_value=0))

def test_SparseAutocovMatrixCSR():
    """Check basic operations on sparse_autocov_csr_mat"""
    from flowstab.SparseStochMat import sparse_autocov_csr_mat as SAMCSR
    pass

def test_SparseAutocovMatrix():
    """Check basic operations on sparse_autocov_mat"""
    from flowstab.SparseStochMat import sparse_autocov_mat as SAM
    pass

def test_sparse_matmul_memory(get_csr_matrix_pair):
    """
    """
    A, B = get_csr_matrix_pair
    for _ in range(5000):
        _ = A @ B

def test_sparse_matmul_mkl_memory(get_csr_matrix_pair):
    """
    """
    from sparse_dot_mkl import dot_product_mkl as mkl_matmul
    A, B = get_csr_matrix_pair
    for _ in range(5000):
        _ = mkl_matmul(A, B)
