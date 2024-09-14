import numpy as np
import tracemalloc

from copy import copy

from scipy.sparse import (
    eye,
)

from flowstab.SparseStochMat import (
    csr_add, 
    csr_matmul,
    csr_csc_matmul,
    csr_csrT_matmul,
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

def test_SSM_inplace_row_normalize_equivalence(SSM_matrix_creator):
    """Make sure the cython and pure python implementations are equivalent
    """
    from flowstab.SparseStochMat import (
        _css
    )
    from flowstab._cython_sparse_stoch_subst import (
        inplace_csr_row_normalize
    )
    A_ssm1 = SSM_matrix_creator(nbr=1)[0]
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

# ###
# Testing the csr operations
# ###

def test_csr_add_compare(cs_matrix_creator):
    """Compare the csr_add to a native implementation"""
    A_ssm1, A_ssm2 = cs_matrix_creator(nbr=2, size=1000000,
                                       nbr_non_zeros=20000, mode='r')
    sum_csr_native = A_ssm1 + A_ssm2
    sum_csr = csr_add(A_ssm1, A_ssm2)
    np.testing.assert_allclose(sum_csr_native.data, sum_csr.data)
    np.testing.assert_equal(sum_csr_native.indices, sum_csr.indices)
    np.testing.assert_equal(sum_csr_native.indptr, sum_csr.indptr)

def test_csr_add_memory(cs_matrix_creator):
    """Check the csr_add function for timing and memory consumption"""
    A_ssm1, A_ssm2 = cs_matrix_creator(nbr=2, size=1000000,
                                       nbr_non_zeros=20000, mode='r')
    _ = csr_add(A_ssm1, A_ssm2)

def test_csr_add_native_memory(cs_matrix_creator):
    """Check the csr native addition for timing and memory consumption"""
    A_ssm1, A_ssm2 = cs_matrix_creator(nbr=2, size=1000000,
                                       nbr_non_zeros=20000, mode='r')
    _ = A_ssm1 + A_ssm2

def test_csr_matmul_compare(cs_matrix_creator):
    """Compare the csr_matmul to a native implementation"""
    A_ssm1, A_ssm2 = cs_matrix_creator(nbr=2, size=10000000,
                                       nbr_non_zeros=200000, mode='r')
    mmul_csr_native = A_ssm1 @ A_ssm2
    mmul_csr = csr_matmul(A_ssm1, A_ssm2)
    np.testing.assert_allclose(mmul_csr_native.data, mmul_csr.data)
    np.testing.assert_equal(mmul_csr_native.indices, mmul_csr.indices)
    np.testing.assert_equal(mmul_csr_native.indptr, mmul_csr.indptr)

def test_csr_matmul_native_memory(cs_matrix_creator):
    """Check the csr_matmul function for timing and memory consumption"""
    A_ssm1, A_ssm2 = cs_matrix_creator(nbr=2, size=10000000,
                                       nbr_non_zeros=200000, mode='r')
    _ = A_ssm1 @ A_ssm2

def test_csr_matmul_memory(cs_matrix_creator):
    """Check the csr native @ function for timing and memory consumption"""
    A_ssm1, A_ssm2 = cs_matrix_creator(nbr=2, size=10000000,
                                       nbr_non_zeros=200000, mode='r')
    _ = csr_matmul(A_ssm1, A_ssm2)

def test_csr_csc_matmul_compare(cs_matrix_creator):
    """Compare the csr_csc_matmul to a native implementation"""
    A_csr, = cs_matrix_creator(nbr=1, mode='r')
    A_csc, = cs_matrix_creator(nbr=1, mode='c')
    mmul_csr_native = A_csr @ A_csc.tocsr()
    mmul_csr = csr_csc_matmul(A_csr, A_csc)
    np.testing.assert_allclose(mmul_csr_native.data, mmul_csr.data)
    np.testing.assert_equal(mmul_csr_native.indices, mmul_csr.indices)
    np.testing.assert_equal(mmul_csr_native.indptr, mmul_csr.indptr)

def test_csr_csc_matmul_native_memory(cs_matrix_creator):
    """Check the csr_matmul function for timing and memory consumption"""
    A_csr, = cs_matrix_creator(nbr=1, mode='r')
    A_csc, = cs_matrix_creator(nbr=1, mode='c')
    _ = A_csr @ A_csc.tocsr()

def test_csr_csc_matmul_memory(cs_matrix_creator):
    """Check the csr native @ function for timing and memory consumption"""
    A_csr, = cs_matrix_creator(nbr=1, mode='r')
    A_csc, = cs_matrix_creator(nbr=1, mode='c')
    _ = csr_csc_matmul(A_csr, A_csc)

def test_csr_csrT_matmul_compare(cs_matrix_creator):
    """Compare the csr_csrT_matmul to a native implementation"""
    A_csr1, = cs_matrix_creator(nbr=1, mode='r')
    A_csr2, = cs_matrix_creator(nbr=1, mode='c')
    mmul_csr_native = A_csr1 @ A_csr2.tocsr().T
    mmul_csr = csr_csrT_matmul(A_csr1, A_csr2)
    np.testing.assert_allclose(mmul_csr_native.data, mmul_csr.data)
    np.testing.assert_equal(mmul_csr_native.indices, mmul_csr.indices)
    np.testing.assert_equal(mmul_csr_native.indptr, mmul_csr.indptr)

def test_csr_csrT_matmul_native_memory(cs_matrix_creator):
    """Check the csr_csrT_matmul function for timing and memory consumption"""
    A_csr, = cs_matrix_creator(nbr=1, mode='r')
    A_csc, = cs_matrix_creator(nbr=1, mode='c')
    _ = A_csr @ A_csc.tocsr().T

def test_csr_csrT_matmul_memory(cs_matrix_creator):
    """Check the csr_csrT native @ function for timing and memory consumption"""
    A_csr, = cs_matrix_creator(nbr=1, mode='r')
    A_csc, = cs_matrix_creator(nbr=1, mode='c')
    _ = csr_csrT_matmul(A_csr, A_csc)

# ###

def test_SparseAutocovMatrixCSR():
    """Check basic operations on sparse_autocov_csr_mat"""
    from flowstab.SparseStochMat import sparse_autocov_csr_mat as SAMCSR
    pass

def test_SparseAutocovMatrix():
    """Check basic operations on sparse_autocov_mat"""
    from flowstab.SparseStochMat import sparse_autocov_mat as SAM
    pass

def test_sparse_matmul_mkl_memory(csr_matrix_creator):
    """
    """
    from sparse_dot_mkl import dot_product_mkl as mkl_matmul
    A, B = csr_matrix_creator(nbr=2)
    for _ in range(1000):
        _ = mkl_matmul(A, B)

def test_sparse_matmul_memory(csr_matrix_creator):
    """
    """
    A, B = csr_matrix_creator(nbr=2)
    for _ in range(1000):
        _ = A @ B
