import numpy as np

def matrix_trace(A):
    """
    Compute the trace of a square matrix (sum of diagonal elements).
    """
    # Write code here
    if not isinstance(A, np.ndarray):
        A_np = np.array(A)

    (m, n) = A_np.shape
    if m / n > 1:
        return None

    trace_A = 0
    for i in range(m):
        trace_A += A_np[i, i]

    return trace_A