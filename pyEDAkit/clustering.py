import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import linkage as scipy_linkage
from scipy.cluster.hierarchy import fcluster, inconsistent
from scipy.spatial.distance import pdist, squareform, cdist
from scipy.special import gammaln
from numpy.polynomial.polynomial import Polynomial
from sklearn.decomposition import FactorAnalysis
from sklearn.decomposition import NMF as skNMF
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.random_projection import GaussianRandomProjection

def linkage(X,
            method='single',
            metric='euclidean',
            *args,
            **kwargs):
    """
    A wrapper around scipy.cluster.hierarchy.linkage that emulates the syntax
    of MATLAB's linkage function.

    Parameters
    ----------
    X : ndarray
        - If X is 2D, interpret as data (n_samples x n_features).
        - If X is 1D, interpret as a condensed distance matrix
          of length n_samples*(n_samples-1)//2.
    method : str, optional
        Linkage method. One of:
          'single', 'complete', 'average', 'weighted', 'centroid',
          'median', 'ward'.
        Default is 'single'.
    metric : str, optional
        Distance metric to use if X is observation data. One of:
          'euclidean', 'squaredeuclidean', 'seuclidean', 'fasteuclidean',
          'fastsquaredeuclidean', 'fastseuclidean', 'mahalanobis', 'cityblock',
          'minkowski', 'chebychev', 'cosine', 'correlation', 'hamming',
          'jaccard', 'spearman'.
        Default is 'euclidean'.

    *args:
        Positional arguments that might appear in MATLAB calls, e.g.
        Z = linkage(X,method,metric,'savememory',value).
        These are ignored or parsed for MATLAB-compatibility only.
    **kwargs:
        Additional keyword arguments, for example:
          - 'V' (std dev array) if metric='seuclidean'
          - 'VI' or 'C' (covariance or inverse covariance) if metric='mahalanobis'
          - 'p' (exponent) if metric='minkowski'
        Or 'savememory' for MATLAB-compatibility.

    Returns
    -------
    Z : ndarray of shape (n-1, 4)
        A linkage matrix encoding the hierarchical clustering. Each row
        represents a merge of two clusters. The columns are:
          1. idx1 : int
          2. idx2 : int
          3. dist : float
          4. size : int
        where 'size' is the total number of original observations in
        the newly formed cluster.

    Notes
    -----
    - If 'centroid', 'median', or 'ward' is chosen, it is assumed that the
      distances are Euclidean. If a condensed distance vector is passed,
      SciPy will attempt to verify it or compute centroid/median/Ward
      distances accordingly.
    - 'fasteuclidean', 'fastsquaredeuclidean', and 'fastseuclidean' are
      mapped internally to their standard counterparts: 'euclidean',
      'squaredeuclidean', and 'seuclidean'.
    - 'savememory' is a no-op in this Python wrapper.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.random.rand(10, 3)
    >>> Z = linkage(X, 'ward')
    >>> Z
    array([[ 0.        ,  5.        ,  0.07603864,  2.        ],
           [ 2.        ,  7.        ,  0.21194439,  2.        ],
           ...
           [ 8.        , 10.        ,  0.56467721, 10.        ]])
    """

    # --- 1) Parse 'savememory' if present (MATLAB compatibility). ---
    #     We do not do anything with it in SciPy, but let's remove it from kwargs.
    if 'savememory' in kwargs:
        _ = kwargs.pop('savememory', None)  # no-op, just pop it

    # --- 2) Map any "fast*" metric to normal ones for SciPy. ---
    metric_map = {
        'fasteuclidean': 'euclidean',
        'fastsquaredeuclidean': 'squaredeuclidean',
        'fastseuclidean': 'seuclidean'
    }
    if metric in metric_map:
        metric = metric_map[metric]

    # --- 3) If the user provided a condensed distance matrix (1D),
    #         then call scipy_linkage directly. ---
    X = np.asarray(X)
    if X.ndim == 1:
        # We assume X is already a condensed distance matrix
        # SciPy's linkage can handle it directly if method is valid
        Z = scipy_linkage(X, method=method)
        return Z

    # --- 4) If X is a 2D observation matrix, compute distances using pdist. ---
    #     We pass any additional parameters in **kwargs to pdist if relevant.
    #     For example: pdist(X, metric='mahalanobis', VI=..., ...).
    if X.ndim == 2:
        # For 'squaredeuclidean', SciPy expects metric='euclidean' with an extra
        # note that the squared distances will be handled differently.
        # Alternatively, we can compute them ourselves or pass a custom function.
        # The most straightforward approach is:
        if metric == 'squaredeuclidean':
            # SciPy doesn't have a direct 'squaredeuclidean' metric name.
            # We'll compute the Euclidean distances and square them.
            # Then call linkage on that condensed matrix:
            dist_array = pdist(X, metric='euclidean', **kwargs)
            # Square the distances:
            dist_array = dist_array**2
            Z = scipy_linkage(dist_array, method=method)
            return Z
        else:
            # For all other valid metrics, we can pass them directly to pdist.
            dist_array = pdist(X, metric=metric, **kwargs)
            Z = scipy_linkage(dist_array, method=method)
            return Z

    # --- 5) If none of the above, raise an error (e.g., user provided a
    #         3D array or something invalid). ---
    raise ValueError(
        "Input X must be either a 2D observation matrix or a 1D condensed distance "
        "matrix. Got shape: {}".format(X.shape)
    )

##############################################################################
# MATLAB-style cluster function (wrapper around SciPy fcluster)
##############################################################################
def cluster(Z, *args, **kwargs):
    """
    A wrapper around SciPy's fcluster and inconsistent that emulates
    MATLAB's cluster function for hierarchical clustering.

    Syntax:
    --------
    T = cluster(Z,'Cutoff',C)
    T = cluster(Z,'Cutoff',C,'Depth',D)
    T = cluster(Z,'Cutoff',C,'Criterion',criterion)
    T = cluster(Z,'MaxClust',N)

    Parameters
    ----------
    Z : ndarray
        The hierarchical clustering encoded as a linkage matrix (output of
        the `linkage` function). Typically shape (m-1, 4) in SciPy.

    *args:
        Positional arguments used to mimic MATLAB name-value pairs:
         - 'Cutoff', C
         - 'Depth', D
         - 'Criterion', criterion
         - 'MaxClust', N
         Each can also be passed in multiple-value form (e.g., C or N can be arrays).

    **kwargs:
        Same as *args but in key=value style. e.g. cluster(Z, Cutoff=..., Depth=...).

    Returns
    -------
    T : ndarray
        A numeric vector or matrix of cluster assignments for each observation.
        - If C or N is a scalar, T is 1D of length m (where m is #observations).
        - If C or N is a length L array, T is an m-by-L matrix, each column
          containing cluster assignments for the corresponding C or N value.

    Notes
    -----
    - 'inconsistent' is the default criterion for 'Cutoff', with Depth=2 by default.
    - If 'Criterion'='distance', cluster uses a distance threshold in fcluster.
    - If 'Criterion'='inconsistent', cluster uses the "inconsistent" threshold in fcluster.
    - If 'MaxClust' is provided, it uses criterion='maxclust' in fcluster,
      forcing a maximum of N clusters. This uses a distance-based cut.
    - For multiple values of C or N, the output T is a matrix with one column
      per threshold or cluster count.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.random.rand(10, 3)
    >>> Z = linkage(X, 'ward')
    >>> # Cutoff by distance at 1.0
    >>> T = cluster(Z, 'Cutoff', 1.0, 'Criterion', 'distance')
    >>> # Force exactly 3 clusters
    >>> T2 = cluster(Z, 'MaxClust', 3)
    >>> # Multiple cutoffs
    >>> T3 = cluster(Z, 'Cutoff', [0.7, 1.2, 1.5], 'Criterion', 'distance')
    """
    # Default values
    cutoff = None
    maxclust = None
    criterion = 'inconsistent'  # default in MATLAB
    depth = 2                   # default Depth in MATLAB

    # --- 1) Parse the *args in MATLAB-like name-value pairs ---
    i = 0
    while i < len(args):
        if i+1 >= len(args):
            raise ValueError(f"Parameter '{args[i]}' is missing its value.")
        name = str(args[i]).lower()
        val = args[i+1]
        i += 2

        if name == 'cutoff':
            cutoff = val
        elif name == 'maxclust':
            maxclust = val
        elif name == 'criterion':
            criterion = str(val).lower()
        elif name == 'depth':
            depth = int(val)
        else:
            raise ValueError(f"Unrecognized parameter: '{args[i]}'")

    # --- 2) Parse **kwargs similarly ---
    for k, v in kwargs.items():
        k_lower = k.lower()
        if k_lower == 'cutoff':
            cutoff = v
        elif k_lower == 'maxclust':
            maxclust = v
        elif k_lower == 'criterion':
            criterion = str(v).lower()
        elif k_lower == 'depth':
            depth = int(v)
        else:
            raise ValueError(f"Unrecognized parameter: '{k}'")

    # Convert to NumPy arrays if they are not already
    if cutoff is not None and not np.isscalar(cutoff):
        cutoff = np.asarray(cutoff).ravel()
    if maxclust is not None and not np.isscalar(maxclust):
        maxclust = np.asarray(maxclust).ravel()

    # We need the number of observations, m, to shape the output.
    # SciPy linkage has shape (m-1, 4). So m = Z.shape[0] + 1
    m = Z.shape[0] + 1

    # --- 3) If 'MaxClust' is provided, do an fcluster for each N in maxclust. ---
    if maxclust is not None:
        # If maxclust is scalar, just do one call.
        if np.isscalar(maxclust):
            N = int(maxclust)
            T = fcluster(Z, t=N, criterion='maxclust')
            return T
        else:
            # We have a vector of N values => produce an m-by-l matrix
            T_out = np.zeros((m, len(maxclust)), dtype=int)
            for iN, N in enumerate(maxclust):
                cluster_assign = fcluster(Z, t=N, criterion='maxclust')
                T_out[:, iN] = cluster_assign
            return T_out

    # --- 4) If 'Cutoff' is provided, do an fcluster for each c in cutoff. ---
    if cutoff is not None:
        # If the user did not specify criterion, it remains 'inconsistent' by default.

        # If cutoff is scalar, do one call. If array, do multiple columns.
        if np.isscalar(cutoff):
            c_val = float(cutoff)
            if criterion == 'distance':
                T = fcluster(Z, t=c_val, criterion='distance')
                return T
            elif criterion == 'inconsistent':
                T = fcluster(Z, t=c_val, criterion='inconsistent', depth=depth)
                return T
            else:
                raise ValueError(
                    "Only 'inconsistent' or 'distance' are valid cluster criteria with Cutoff."
                )
        else:
            # cutoff is an array => produce an m-by-l matrix
            T_out = np.zeros((m, len(cutoff)), dtype=int)
            for ic, c_val in enumerate(cutoff):
                if criterion == 'distance':
                    cluster_assign = fcluster(Z, t=float(c_val), criterion='distance')
                elif criterion == 'inconsistent':
                    cluster_assign = fcluster(Z, t=float(c_val),
                                              criterion='inconsistent',
                                              depth=depth)
                else:
                    raise ValueError(
                        "Only 'inconsistent' or 'distance' are valid cluster criteria with Cutoff."
                    )
                T_out[:, ic] = cluster_assign
            return T_out

    # --- 5) If neither 'MaxClust' nor 'Cutoff' is specified, error out. ---
    raise ValueError(
        "No valid clustering instruction found. Use 'Cutoff',C or 'MaxClust',N."
    )

