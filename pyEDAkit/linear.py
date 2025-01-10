import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF as skNMF


def PCA(X, n_components = 2, covariance = True, plot_scree_plot = False):
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

    if plot_scree_plot:
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(sorted_eigenvalue) + 1), sorted_eigenvalue, marker='o', linestyle='-')
        plt.plot(n_components, sorted_eigenvalue[n_components - 1], 'ro', label = 'n_components')
        plt.title('Scree Plot')
        plt.xlabel('Eigenvalue Index')
        plt.ylabel('Eigenvalue Magnitude')
        plt.legend()
        plt.grid(True)
        plt.show()

    return (X @ sorted_eigenvectors)[:, :n_components]

def SVD(X, plot_singular_values = False):
    # It provides a way to find the PCs without explicitly calculating 
    #   the covariance matrix.
    # The plot of the singular values is similar to the scree plot in PCA.
    U, S, Vt = np.linalg.svd(X)
    
    if plot_singular_values:
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(S) + 1), S, marker='o', linestyle='-', label='Singular Values')
        plt.title('Singular Value Decomposition')
        plt.xlabel('Index')
        plt.ylabel('Singular Value')
        plt.grid(True)
        plt.legend()
        plt.show()

    return U, S, Vt

def NMF(X, rank, plot_components_contribution = False):
    # is X non-negative?
    if np.any(X < 0):
        print('Error: X contains negative values.')
        return None

    nmf_model = skNMF(n_components=rank, init='random', random_state=42) 
    W = nmf_model.fit_transform(X)
    H = nmf_model.components_

    X_reconstructed = np.dot(W, H)

    if plot_components_contribution:
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, H.shape[0] + 1), np.linalg.norm(H, axis=1), marker='o', linestyle='-', label='Component Norms')
        plt.title('NMF Component Contributions')
        plt.xlabel('Component Index')
        plt.ylabel('Norm of Component')
        plt.grid(True)
        plt.legend()
        plt.show()

    return W, H