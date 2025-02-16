import pytest
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
    from flowstab.sparse_stoch_mat import timing
    
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
    from flowstab.sparse_stoch_mat import SparseStochMat as SSM
    # Inits
    # ###
    # inti from scipy.sparse.csr_matrix
    A_csr = get_csr_matrix_small
    print(f"{A_csr[1,2]=}")
    ssm = SSM.from_full_csr_matrix(A_csr)
    np.testing.assert_equal(A_csr.toarray(), ssm.toarray(), strict=False)
    # crete a diagonal matrix
    _ = SSM.create_diag(size=100, diag_val=0.3)
    # convert it to a full csr
    full_A = ssm.to_full_mat()
    # ...

    

def test_SSM_large(get_csr_matrix_large):
    """Make sure an SSM does not get expanded during creation
    """
    from flowstab.sparse_stoch_mat import SparseStochMat as SSM
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
    from flowstab.sparse_stoch_mat import (
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
    from flowstab.sparse_stoch_mat import (
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
    from flowstab.sparse_stoch_mat import (
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

def test_rebuild_nnz_rowcol(cs_matrix_creator, compare_alike):
    """Test conversions from ssm to csr and back
    """
    from flowstab.sparse_stoch_mat import SparseStochMat as SSM
    A_csr = cs_matrix_creator(nbr=1, size=100000, nbr_non_zeros=1000)[0]
    A_ssm = SSM.from_full_csr_matrix(Tcsr=A_csr)
    A_rebuild = A_ssm.to_full_mat()
    compare_alike(A_csr, A_rebuild)

# ### Testing the csr operations
# ###

def test_csr_add_native_memory(cs_matrix_creator):
    """Check the csr native addition for timing and memory consumption"""
    A_ssm1, A_ssm2 = cs_matrix_creator(nbr=2, size=1000000,
                                       nbr_non_zeros=20000, mode='r')
    _ = A_ssm1 + A_ssm2

def test_csr_csc_matmul_native_memory(cs_matrix_creator):
    """Check the csr_matmul function for timing and memory consumption"""
    A_csr, = cs_matrix_creator(nbr=1, mode='r')
    A_csc, = cs_matrix_creator(nbr=1, mode='c')
    _ = A_csr @ A_csc.tocsr()

def test_csr_csrT_matmul_native_memory(cs_matrix_creator):
    """Check the csr_csrT_matmul function for timing and memory consumption"""
    A_csr, = cs_matrix_creator(nbr=1, mode='r')
    A_csc, = cs_matrix_creator(nbr=1, mode='c')
    _ = A_csr @ A_csc.tocsr().T

# ###

def test_inplace_diag_matmul_csr(cs_matrix_creator):
    """Check the inplace diagonal multiplication for csr and csc"""
    from flowstab.sparse_stoch_mat import (
        inplace_csr_matmul_diag,
        inplace_diag_matmul_csr
    )
    size = 1000
    nnz = 100
    A_csr, = cs_matrix_creator(nbr=1, size=size, nbr_non_zeros=nnz, mode='r')
    A_csc, = cs_matrix_creator(nbr=1, size=size, nbr_non_zeros=nnz, mode='c')
    Acsr_array = A_csr.toarray()
    Acsc_array = A_csc.toarray()
    diag_array = np.random.randint(0, 10, size=size)
    # test the csr sparse matrix column resacling
    inplace_csr_matmul_diag(A_csr, diag_array)
    Diag = np.diagflat(diag_array)
    Acsr_rescaled = Acsr_array @ Diag
    np.testing.assert_equal(A_csr.toarray(), Acsr_rescaled)
    # now rescale the rows
    inplace_diag_matmul_csr(A_csr, diag_array)
    Acsr_rescaled_row =  Diag @ Acsr_rescaled
    np.testing.assert_equal(A_csr.toarray(), Acsr_rescaled_row)
    # test the csc sparse matrix column rescaling
    inplace_csr_matmul_diag(A_csc, diag_array)
    Acsc_rescaled = Acsc_array @ Diag
    np.testing.assert_equal(A_csc.toarray(), Acsc_rescaled)
    # now rescale the rows
    inplace_diag_matmul_csr(A_csc, diag_array)
    Acsc_rescaled_row =  Diag @ Acsc_rescaled
    np.testing.assert_equal(A_csc.toarray(), Acsc_rescaled_row)


# ###
# Testing the autocovaraince matrix class
# ###

@pytest.mark.parametrize("p1, p2, size",
                         [(np.random.random(size=1000),
                           np.random.random(size=1000), 1000),
                          (0.2, 0.3, 1000),
                          (0.4, None, 1000),
                          (None, 0.1, 1000),
                          (None, None, 1000)])
def test_SAM_init(p1, p2, size, cs_matrix_creator):
    """Check basic operations on SparseAutocovMat"""
    from flowstab.sparse_stoch_mat import SparseAutocovMat as SAM
    from flowstab.sparse_stoch_mat import (
        inplace_diag_matmul_csr
    )
    T = cs_matrix_creator(nbr=1, size=size, nbr_non_zeros=1000)[0]
    PT = T.copy()
    if p1 is not None:
        if not isinstance(p1, np.ndarray):
            _p1 = np.full(shape=size, fill_value=p1)
        else:
            _p1 = p1
        inplace_diag_matmul_csr(PT, _p1)
    # testing various init methods with from_T
    sam = SAM(PT=PT, p1=p1, p2=p2)
    sam_copy = sam.copy()
    sam_array = sam.toarray()

@pytest.mark.parametrize("p1, p2, size",
                         [(np.random.random(size=100000),
                           np.random.random(size=100000), 100000),
                          (0.2, 0.3, 100000),
                          (0.4, None, 100000),
                          (None, 0.1, 100000),
                          (None, None, 100000)])
def test_SAM_from_T(p1, p2, size, cs_matrix_creator):
    """Check basic operations on SparseAutocovMat"""
    from flowstab.sparse_stoch_mat import SparseAutocovMat as SAM
    from flowstab.sparse_stoch_mat import (
        inplace_diag_matmul_csr
    )
    T = cs_matrix_creator(nbr=1, size=size, nbr_non_zeros=1000)[0]
    PT = T.copy()
    if p1 is not None:
        if not isinstance(p1, np.ndarray):
            _p1 = np.full(shape=size, fill_value=p1)
        else:
            _p1 = p1
        inplace_diag_matmul_csr(PT, _p1)
    # testing various init methods with from_T
    np.testing.assert_equal(
        SAM(PT=PT, p1=p1, p2=p2).PT.data, 
        SAM.from_T(T=T, p1=p1, p2=p2).PT.data
    )

@pytest.mark.parametrize("p1, p2, size",
                         [(np.random.random(size=100000),
                           np.random.random(size=100000), 100000),
                          (0.2, 0.3, 100000),
                          (0.4, None, 100000),
                          (None, 0.1, 100000),
                          (None, None, 100000)])
def test_SAM_from_T_forward(p1, p2, size, cs_matrix_creator):
    """Check basic operations on SparseAutocovMat"""
    from flowstab.sparse_stoch_mat import SparseAutocovMat as SAM
    from flowstab.sparse_stoch_mat import (
        inplace_diag_matmul_csr
    )
    T = cs_matrix_creator(nbr=1, size=size, nbr_non_zeros=1000)[0]
    PT = T.copy()
    if p1 is not None:
        if not isinstance(p1, np.ndarray):
            _p1 = np.full(shape=size, fill_value=p1)
        else:
            _p1 = p1
        inplace_diag_matmul_csr(PT, _p1)
    # testing various init methods with  from_T_forward
    np.testing.assert_equal(
        SAM(PT=PT, p1=p1, p2=p2).PT.data, 
        SAM.from_T_forward(T=T, p1=p1, p2=p2).PT.data
    )

def test_sparse_matmul_mkl_memory(cs_matrix_creator):
    """
    """
    from sparse_dot_mkl import dot_product_mkl as mkl_matmul
    A, B = cs_matrix_creator(nbr=2)
    for _ in range(1000):
        _ = mkl_matmul(A, B)

def test_sparse_matmul_memory(cs_matrix_creator):
    """
    """
    A, B = cs_matrix_creator(nbr=2)
    for _ in range(1000):
        _ = A @ B
