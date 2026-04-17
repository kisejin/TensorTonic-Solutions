import numpy as np

def manhattan_distance(x, y):
    """
    Compute the Manhattan (L1) distance between vectors x and y.
    Must return a float.
    """
    # Write code here
    if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
        x_np = np.array(x)
        y_np = np.array(y)

    m_d = int(np.sum(np.abs(x_np - y_np)))
    
    return m_d