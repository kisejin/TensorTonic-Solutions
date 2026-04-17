import numpy as np

def dot_product(x, y):
    """
    Compute the dot product of two 1D arrays x and y.
    Must return a float.
    """
    # Write code here
    if len(x) != len(y):
        raise ValueError()
    if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
        x_np = np.array(x)
        y_np = np.array(y)

    sum = 0
    for xi, yi in zip(x_np, y_np):
        sum += xi*yi

    return sum