from sklearn.manifold import MDS as skMDS
from sklearn.manifold import LocallyLinearEmbedding, Isomap



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