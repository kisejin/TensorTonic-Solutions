import numpy as np

def cosine_similarity(a, b):
    """
    Compute cosine similarity between two 1D NumPy arrays.
    Returns: float in [-1, 1]
    """
    if not isinstance(a, np.ndarray) or not isinstance(b, np.ndarray):
        a_np = np.array(a)
        b_np = np.array(b)

    
    num_sim = a_np@b_np
    dec_sim = np.sqrt(np.sum(a_np**2)) * np.sqrt(np.sum(b_np**2)) + 1e-10
    return num_sim / dec_sim