import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    A_np = A if isinstance(A, np.ndarray) else np.array(A)
    m, n = A_np.shape[0], A_np.shape[1]
    A_T = np.zeros((n, m))

    for j in range(n):
        A_T[j, :] = A_np[:, j]

    return A_T
