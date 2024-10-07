import numpy as np

def PCA(X, n_components = 2, covariance = True):
    X_mean = X.mean(axis=0)
    X = X - X_mean
    S = None
    if covariance:
        S = np.cov(X, rowvar=False)
    else:
        S = np.corrcoef(X, rowvar=False)
    eigen_values, eigen_vectors = np.linalg.eig(S)
    sorted_index = np.argsort(eigen_values)[::-1]
    sorted_eigenvalue = eigen_values[sorted_index]
    sorted_eigenvectors = eigen_vectors[:, sorted_index]
    return (X @ sorted_eigenvectors)[:, :n_components]
