import numpy as np

def PCA(X, components=2, cov = True):
    # centering the data matrix X for each direction
    X_mean = X.mean(axis=0)
    X = X - X_mean
    
    # transformation matrix
    S = None
    if cov:
        S = np.cov(X, rowvar=False)
    else:
        S = np.corrcoef(X, rowvar=False)

    eigenvalues, eigenvectors = np.linalg.eig(S)

    return X @ eigenvectors[:,range(0,components)]