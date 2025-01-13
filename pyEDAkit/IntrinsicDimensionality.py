import numpy as np
from scipy.special import gammaln
from numpy.polynomial.polynomial import Polynomial
from scipy.stats import linregress
from scipy.spatial.distance import pdist, squareform


def id_pettis(X):
    """
    Estimate the intrinsic dimensionality using the Pettis, Bailey, Jain, and Dubes algorithm.

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