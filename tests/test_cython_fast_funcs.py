import pytest

import numpy as np

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
