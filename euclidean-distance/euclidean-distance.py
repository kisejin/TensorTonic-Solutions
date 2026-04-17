import numpy as np

def euclidean_distance(x, y):
    """
    Compute the Euclidean (L2) distance between vectors x and y.
    Must return a float.
    """
    # Write code here
    if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
        x_np = np.array(x)
        y_np = np.array(y)
    euc_d = np.sqrt(
        np.sum(
            (x_np - y_np)**2
        )
    )

    return euc_d