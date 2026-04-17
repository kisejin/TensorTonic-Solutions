import numpy as np

def covariance_matrix(X):
    """
    Compute covariance matrix from dataset X.
    """
    # Write code here
    if not isinstance(X, np.ndarray):
        X_np = np.array(X)
    
    if len(X_np) <= 1 or len(X_np.shape) <= 1:
        return None
    

    if not isinstance(X, np.ndarray):
        X_np = np.array(X)
    
    X_mu = np.mean(X_np, axis=0)

    X_centered = X_np - X_mu
    m, n = X_np.shape[0], X_np.shape[1]
    X_cov = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            X_cov[i, j] = X_centered[:, i] @ X_centered[:, j]

    
    return  (1/(m - 1)) * X_cov
    