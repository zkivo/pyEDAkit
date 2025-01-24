from sklearn.manifold import MDS  as skMDS
from sklearn.manifold import TSNE as skTSNE
from sklearn.manifold import LocallyLinearEmbedding, Isomap
from minisom import MiniSom
import numpy as np

def MDS(X, d, metric=True):
    """
    Multidimensional Scaling (MDS) is a technique that reduces the 
    dimensionality of the data while preserving the distances between the 
    data points as much as possible.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        The data to be transformed.
    d : int
        The number of dimensions to reduce to.
    metric : bool, default=True
        If True, use metric MDS; otherwise, use nonmetric MDS.
    
    Returns
    -------
    array-like, shape (n_samples, d)
        The transformed data.
    """

    mds = skMDS(n_components=d, random_state=42, metric=metric)
    return mds.fit_transform(X)

def LLE(X, d, K=12):
    """
    Locally Linear Embedding (LLE) is a technique that reduces the
    dimensionality of the data while preserving the local relationships
    between the data points.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        The data to be transformed.
    d : int
        The number of dimensions to reduce to.

    Returns
    -------
    array-like, shape (n_samples, d)
        The transformed data.
    """
    lle = LocallyLinearEmbedding(n_neighbors=K, n_components=d, method='standard')
    return lle.fit_transform(X)

def ISOMAP(X, d, K):
    """
    Isometric Mapping (ISOMAP) is a technique that reduces the dimensionality
    of the data while preserving the geodesic distances between the data points.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        The data to be transformed.
    d : int
        The number of dimensions to reduce to.
    K : int
        The number of nearest neighbors to consider.

    Returns
    -------
    array-like, shape (n_samples, d)
        The transformed data.
    """
    isomap = Isomap(n_neighbors=K, n_components=d)
    return isomap.fit_transform(X)

def HLLE(X, d, K):
    """
    Hessian Locally Linear Embedding (HLLE) is a technique that reduces the
    dimensionality of the data while preserving the local relationships between
    the data points.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        The data to be transformed.
    d : int
        The number of dimensions to reduce to.
    K : int
        The number of nearest neighbors to consider.

    Returns
    -------
    array-like, shape (n_samples, d)
        The transformed
    """    
    hlle = LocallyLinearEmbedding(n_neighbors=K, n_components=d, method='hessian')
    return hlle.fit_transform(X)

def SOM(X):
    """
    Self-Organizing Map (SOM) is a type of artificial neural network that is
    trained using unsupervised learning to produce a low-dimensional
    representation of the input space.

    It maps high-dimensional data to a discrete and finite 2D plane

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        The data to be transformed.

    Returns
    -------
    array-like, shape (n_samples, d)
        The transformed data.
    """
    som_size = (10, 10)
    som = MiniSom(som_size[0], som_size[1], X.shape[1], sigma=1.0, 
        learning_rate=0.5, topology='rectangular', 
        neighborhood_function='gaussian')
    som.random_weights_init(X)
    som.train_random(X, 100000, verbose=True)

    U_matrix = som.distance_map()
    Z = np.array([som.winner(x) for x in X])  # Get neuron indices for each data point
    return Z, U_matrix

def TSNE(X, d):
    """
    t-Distributed Stochastic Neighbor Embedding (t-SNE) is a technique that
    reduces the dimensionality of the data while preserving the local
    relationships between the data points.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        The data to be transformed.
    d : int
        The number of dimensions to reduce to.

    Returns
    -------
    array-like, shape (n_samples, d)
        The transformed data.
    """
    tsne = skTSNE(n_components=d, random_state=42, perplexity=30)
    return tsne.fit_transform(X)