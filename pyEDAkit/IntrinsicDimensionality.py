import numpy as np
from scipy.special import gammaln
from numpy.polynomial.polynomial import Polynomial
from scipy.spatial.distance import pdist, squareform
from scipy.spatial.distance import cdist

def find_nn(X, k):
    """
    Compute the pairwise distance matrix and return the distance matrix
    with only the k-nearest neighbors retained (others set to 0).
    """
    n = X.shape[0]
    distances = cdist(X, X, 'euclidean')  # Pairwise Euclidean distances
    nearest_neighbors = np.zeros_like(distances)

    for i in range(n):
        # Find indices of k smallest distances (excluding the diagonal element)
        neighbor_indices = np.argsort(distances[i, :])[1:k+1]
        nearest_neighbors[i, neighbor_indices] = distances[i, neighbor_indices]

    return nearest_neighbors

def packing_numbers(X):
    # Parameters for the estimation
    r = [0.1, 0.5]
    epsilon = 0.01
    max_iter = 20
    done = False
    l = 0

    # Input X assumed to be a NumPy array
    # X = np.array(...)  # Define your input here

    # Perform iterations (until 'convergence')
    L = np.zeros((2, max_iter))
    while not done:
        l += 1
        perm = np.random.permutation(X.shape[0])

        # Compute L for two radii (size of C is packing number)
        for k in range(2):
            C = []
            for i in range(X.shape[0]):
                is_far = True
                for j in C:
                    if np.sqrt(np.sum((X[perm[i]] - X[j]) ** 2)) < r[k]:
                        is_far = False
                        break
                if is_far:
                    C.append(perm[i])
            L[k, l - 1] = np.log(len(C))  # maximum cardinality of an r(k)-separated subset of X

        # Estimate of intrinsic dimension
        no_dims = -((np.mean(L[1, :l]) - np.mean(L[0, :l])) / (np.log(r[1]) - np.log(r[0])))

        # Stop condition
        if l > 10:
            variance_sum = np.var(L[0, :l]) ** 2 + np.var(L[1, :l]) ** 2
            if 1.65 * (np.sqrt(variance_sum) / (np.sqrt(l) * (np.log(r[1]) - np.log(r[0])))) < no_dims * ((1 - epsilon) / 2):
                done = True
        if l >= max_iter:
            done = True
    
    return no_dims

def MLE(X):
    # Set neighborhood range to search in
    k1 = 6
    k2 = 12

    # Transpose X to match MATLAB input format
    X = X.T
    d, n = X.shape

    # Compute X^2 for distance calculation
    X2 = np.sum(X ** 2, axis=0)
    knnmatrix = np.zeros((k2, n))

    if n < 3000:
        # Compute pairwise distances
        distance = np.add.outer(X2, X2) - 2 * np.dot(X.T, X)
        # Sort distances for nearest neighbors
        distance = np.sort(distance, axis=0)
        knnmatrix = 0.5 * np.log(distance[1:k2 + 1, :])
    else:
        for i in range(n):
            # Compute distances for a single point
            distance = np.sort(X2[i] + X2 - 2 * np.dot(X[:, i], X))
            knnmatrix[:, i] = 0.5 * np.log(distance[1:k2 + 1])

    # Compute the ML estimate
    S = np.cumsum(knnmatrix, axis=0)
    indexk = np.arange(k1, k2 + 1).reshape(-1, 1)  # Reshape for broadcasting
    indexk = np.tile(indexk, (1, n))
    dhat = -(indexk - 2) / (S[k1 - 1:k2, :] - knnmatrix[k1 - 1:k2, :] * indexk)

    # Average over estimates and values of k
    no_dims = np.mean(dhat)
    
    return no_dims

def corr_dim(X):
    # Compute correlation dimension estimation
    n = X.shape[0]

    # Compute distance matrix with k-nearest neighbors
    D = find_nn(X, 5)

    # Extract non-zero elements from the distance matrix D
    indices = np.nonzero(D)
    val = D[indices]

    r1 = np.median(val)
    r2 = np.max(val)

    s1 = 0
    s2 = 0

    # Transpose X for easier broadcasting
    X = X.T

    XX = np.sum(X ** 2, axis=0)  # Equivalent of MATLAB's sum(X .^ 2)
    onez = np.ones(n)

    for i in range(n):
        p = X[:, i]
        xx = XX[i]
        xX = np.dot(p, X)  # Dot product

        # Calculate pairwise distances and handle numerical precision issues
        dist = xx * onez + XX - 2 * xX
        dist = np.maximum(dist, 0)  # Ensure non-negative distances
        dist = np.sqrt(dist)
        dist = dist[i + 1:n]

        s1 += np.sum(dist < r1)
        s2 += np.sum(dist < r2)

    Cr1 = (2 / (n * (n - 1))) * s1
    Cr2 = (2 / (n * (n - 1))) * s2

    # Estimate intrinsic dimensionality
    no_dims = (np.log(Cr2) - np.log(Cr1)) / (np.log(r2) - np.log(r1))

    return no_dims

def id_pettis(X):
    """
    Estimate the intrinsic dimensionality using the Pettis, 
        Bailey, Jain, and Dubes algorithm.

    Parameters:
    X (array): Data matrix.

    Returns:
    float: Estimate of intrinsic dimensionality.
    """

    # Get the distances using the pdist function
    ydists = pdist(X)
    ydists_matrix = squareform(ydists)
    
    n = X.shape[0]

    K = 5
    kdist = np.zeros((n, K))

    for i in range(n):
        tmp = ydists_matrix[i, :]
        tmp_sorted = np.sort(tmp)
        kdist[i, :] = tmp_sorted[1:(K + 1)]  # Skip the first element (distance to itself)

    # kmax corresponds to the last column
    mmax = np.mean(kdist[:, K - 1])
    smax = np.sqrt(np.var(kdist[:, K - 1]))
    k = np.arange(1, K + 1)

    # Get the averages for the estimate but remove the outliers
    kcell = []
    logrk = []

    for i in range(K):
        ind = np.where(kdist[:, i] <= (mmax + smax))[0]
        kcell.append(kdist[ind, i])
        logrk.append(np.log(np.mean(kcell[i])))

    logrk = np.array(logrk)
    logk = np.log(k)

    # Initial value for d
    p_coeffs = Polynomial.fit(logk, logrk, 1).convert().coef

    dhat = 1 / p_coeffs[1]
    dhatold = np.inf
    maxiter = 100
    epstol = 0.01
    i = 0

    while abs(dhatold - dhat) >= epstol and i < maxiter:
        logGRk = (1 / dhat) * logk + gammaln(k) - gammaln(k + 1 / dhat)
        p_coeffs = Polynomial.fit(logk, logrk + logGRk, 1).convert().coef
        dhatold = dhat
        dhat = 1 / p_coeffs[1]
        i += 1

    return dhat