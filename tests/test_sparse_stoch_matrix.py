import numpy as np
import tracemalloc

from scipy.sparse import (
    eye,
)

def test_SSM_small(get_csr_matrix_small):
    """Bacic operations with the 'spares_stoch_mat' class
    """
    from flowstab.SparseStochMat import sparse_stoch_mat as SSM
    # Inits
    # ###
    # inti from scipy.sparse.csr_matrix
    A_csr = get_csr_matrix_small
    for i in range(10000):
        ssm = SSM.from_full_csr_matrix(A_csr)
        np.testing.assert_equal(A_csr.toarray(), ssm.toarray(), strict=False)
    

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
    A_csr, density = get_csr_matrix_large
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


def test_SPA():
    """Basic operations with the `SpasreStochMat.SPA` class"""
    from flowstab.SparseStochMat import SPA
    # TODO
