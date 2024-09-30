import pytest

import numpy as np

def test_sum_Sto():
    """
    """
    from _cython_fast_funcs import (
        sum_Sto
    )
    # Define a sample S array and ix_cf list for testing purposes.
    S = np.array([[1,2,3], [4,5,6], [7,8,9]], dtype=np.float64)
    k = 0
    ix_cf = [1, 2]
    # Calculate the expected result manually.
    expected_result = S[k, ix_cf[0]] + S[ix_cf[0], k] + S[k, ix_cf[1]] + \
            S[ix_cf[1], k] + S[k,k]
    # Compare the expected result to the actual output from sum_Sto.
    assert np.testing.allclose(expected_result,sum_Sto(S, k, ix_cf))

def test_sum_Sout():
    """
    """
    from _cython_fast_funcs import (
        sum_Sout
    )
    # Define a sample S array and ix_ci list for testing purposes.
    S = np.array([[1,2,3], [4,5,6], [7,8,9]], dtype=np.float64)
    k = 0
    ix_ci = [1, 2]

    # Calculate the expected result manually.
    expected_result = -S[k, ix_ci[0]] - S[ix_ci[0], k] - S[k, ix_ci[1]] - \
            S[ix_ci[1], k] + S[k,k]

    # Compare the expected result to the actual output from sum_Sout.
    assert np.testing.allclose(expected_result,sum_Sout(S, k, ix_ci))

def test_compute_S(probabilities_transition):
    """
    """
    from _cython_fast_funcs import (
        compute_S
    )
    p1, p2, T = probabilities_transition
    S = compute_S(p1=p1, p2=p2, T=T)
    S_nonC = np.diag(p1) @ T - np.outer(p1, p2)
    np.testing.assert_equal(S, S_nonC)
