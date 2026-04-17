import numpy as np

def make_diagonal(v):
    """
    Returns: (n, n) NumPy array with v on the main diagonal
    """
    # Write code here
    n = len(v)
    if n < 1:
        return None

    Diag_m = np.zeros((n, n))
    for i in range(n):
        Diag_m[i, i] = v[i]

    return Diag_m